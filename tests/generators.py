"""Tensor and routing generators shared by the W4A16 tests.

``PERFORMANCE_CASES`` below is the single place where benchmark shapes are
declared.  Adding or removing a case is an edit to that table; the test module
only iterates over what it finds here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


Projection = Literal["gate_up", "down"]
Distribution = Literal["random", "balanced"]


@dataclass(frozen=True)
class IndexedCase:
    """One indexed MoE shape used by correctness or performance checks.

    ``projection`` selects the routing contract of an MoE expert GEMM:

    ``gate_up``
        The gate/up projection reads one row per token and writes one row per
        route, so ``input`` has ``m`` rows and ``output`` has ``m * top_k``.
        The kernel receives the real ``top_k`` and divides each sorted id by it
        to recover the source token row.

    ``down``
        The down projection consumes the routed activations produced by the
        gate/up projection, so ``input`` and ``output`` both have ``m * top_k``
        rows.  The kernel receives ``top_k=1`` so the same id division becomes
        an identity mapping and each routed row indexes itself.  Routing
        metadata is still generated with the real ``top_k``.
    """

    profile: str
    m: int
    n: int
    k: int
    num_experts: int
    top_k: int
    seed: int
    projection: Projection = "gate_up"
    distribution: Distribution = "random"

    @property
    def routed_m(self) -> int:
        """Number of routed rows, i.e. the output row count."""

        return self.m * self.top_k

    @property
    def input_rows(self) -> int:
        """Row count of the activation tensor handed to the kernel."""

        return self.routed_m if self.projection == "down" else self.m

    @property
    def kernel_top_k(self) -> int:
        """``top_k`` value passed to the kernel for this projection."""

        return 1 if self.projection == "down" else self.top_k

    @property
    def is_tensor_parallel(self) -> bool:
        """True when the profile shards ``moe_intermediate`` instead of experts."""

        return "_tp8" in self.profile

    @property
    def is_decode(self) -> bool:
        return "decode" in self.profile

    @property
    def is_prefill(self) -> bool:
        """True when the sweep is a whole-system chunk rather than a decode step.

        TP8 counts as prefill-shaped here even though its profile is ``mix``.
        """

        return not self.is_decode

    @property
    def token_count(self) -> int:
        """Token count in the unit this phase is naturally specified in.

        Prefill-shaped sweeps are quoted for the whole system, because a chunk is
        split across all ranks; decode runs with DP8, so its count is already per
        GPU.  Reporting the raw ``m`` for prefill would invite reading a
        system-wide number as a per-GPU one.

        The two shard axes recover it differently: under EP8 a rank owns 48 of the
        384 experts, so exactly ``routed_m == T`` routes land on it, while under
        TP8 every route is local and the count is ``routed_m / top_k``, i.e. ``m``.
        """

        if self.is_decode or self.is_tensor_parallel:
            return self.m
        return self.routed_m

    @property
    def token_count_label(self) -> str:
        return "tokens/GPU" if self.is_decode else "tokens_total"

    @property
    def label(self) -> str:
        return (
            f"{self.profile} {self.projection} m={self.m} n={self.n} k={self.k} "
            f"E={self.num_experts} top_k={self.top_k} {self.distribution}"
        )


@dataclass(frozen=True)
class IndexedTensors:
    profile: object
    kernel_config: object
    activation: torch.Tensor
    logical_weight: torch.Tensor
    scale: torch.Tensor
    prepared_weight: object
    topk_ids: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_padded: torch.Tensor
    output: torch.Tensor

    @property
    def num_active_experts(self) -> int:
        """Count experts actually reached by routing, for DRAM traffic math."""

        return int(torch.unique(self.expert_ids).numel())


def generate_topk_ids(
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    distribution: Distribution = "random",
) -> torch.Tensor:
    """Pick ``top_k`` distinct experts per token.

    ``random`` mirrors the upstream Humming benchmark: score every expert and
    take the top ``top_k``, which yields the skewed per-expert token counts a
    real router produces.  ``balanced`` spreads routes evenly so each expert
    receives near-identical work, which keeps block padding minimal.
    """

    if top_k > num_experts:
        raise ValueError(
            f"top_k={top_k} cannot exceed num_experts={num_experts}"
        )
    if distribution == "balanced":
        # Draw an even number of routes per expert, then let the scatter below
        # turn them into the highest scores for their token.
        tokens_per_expert = math.ceil(num_tokens * top_k / num_experts)
        expert_scores = torch.randn(
            (tokens_per_expert, num_experts),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        expert_index = torch.argsort(expert_scores, dim=1).reshape(-1)
        expert_index = expert_index[: num_tokens * top_k].view(num_tokens, top_k)
        scores = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        scores.scatter_(1, expert_index, 1000.0)
    elif distribution == "random":
        scores = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    return scores.topk(top_k, dim=1)[1].to(torch.int32)


def align_routing_blocks(
    topk_ids: torch.Tensor,
    *,
    block_m: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort routes by expert into ``block_m`` blocks padded with a sentinel.

    This is the CPU equivalent of vLLM's ``moe_align_block_size``: every active
    block holds valid routed ids as a contiguous prefix followed only by
    sentinel values, and each block belongs to exactly one expert.
    """

    device = topk_ids.device
    flat_ids = topk_ids.reshape(-1)
    sentinel = flat_ids.numel()
    sorted_parts: list[torch.Tensor] = []
    expert_blocks: list[int] = []
    for expert in range(num_experts):
        rows = torch.where(flat_ids == expert)[0]
        if rows.numel() == 0:
            # An expert with no routes contributes no block, matching the way a
            # router hands only active blocks to the kernel.
            continue
        num_blocks = math.ceil(rows.numel() / block_m)
        rows = torch.nn.functional.pad(
            rows, (0, num_blocks * block_m - rows.numel()), value=sentinel
        )
        sorted_parts.append(rows)
        expert_blocks.extend([expert] * num_blocks)

    if not sorted_parts:
        raise ValueError("routing produced no active expert blocks")
    sorted_ids = torch.cat(sorted_parts).to(torch.int32).contiguous()
    expert_ids = torch.tensor(expert_blocks, dtype=torch.int32, device=device)
    num_tokens_padded = torch.tensor(
        sorted_ids.numel(), dtype=torch.int32, device=device
    )
    return sorted_ids, expert_ids, num_tokens_padded


def generate_indexed_routing(
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    block_m: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    distribution: Distribution = "random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build vLLM-style sorted routing metadata for one case."""

    topk_ids = generate_topk_ids(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        device=device,
        generator=generator,
        distribution=distribution,
    )
    sorted_ids, expert_ids, num_tokens_padded = align_routing_blocks(
        topk_ids, block_m=block_m, num_experts=num_experts
    )
    return topk_ids, sorted_ids, expert_ids, num_tokens_padded


def reference_indexed(
    activation: torch.Tensor,
    logical_weight: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    *,
    projection: Projection = "gate_up",
    max_rows: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Reference W4A16 GEMM written in plain PyTorch, for engineering error only.

    Dequantization happens in BF16 with a BF16 group scale, exactly as the kernel
    does it, and only the accumulation is FP32 -- matching the kernel's FP32
    accumulator.  Comparing against this isolates engineering error (layout,
    routing, pipelining) from the accuracy cost of 4-bit quantization itself: a
    reference that dequantized in FP32 would fold the quantization algorithm's
    own error into the number and mask real kernel bugs behind it.

    For ``gate_up`` each routed row reads source token ``route // top_k``; for
    ``down`` the activation is already routed, so row ``i`` reads itself.
    """

    route_experts = topk_ids.reshape(-1)
    if projection == "down":
        routed_inputs = activation
    else:
        routed_inputs = activation.repeat_interleave(top_k, dim=0)
    rows = (
        route_experts.numel()
        if max_rows is None
        else min(max_rows, route_experts.numel())
    )
    rows = max(rows, 1)
    route_experts = route_experts[:rows]
    routed_inputs = routed_inputs[:rows]
    reference = torch.empty(
        (rows, logical_weight.size(1)), dtype=torch.float32, device=activation.device
    )
    # Dequantize a few routes at a time so a production-sized expert tensor never
    # has to be materialized in BF16 all at once.
    for start in range(0, rows, 4):
        end = min(start + 4, rows)
        expert = route_experts[start:end]
        # Unsigned code -> signed INT4 value in BF16, then the group-32 scale.
        grouped = (logical_weight[expert] - 8).to(torch.bfloat16).reshape(
            end - start,
            logical_weight.size(1),
            logical_weight.size(2) // 32,
            32,
        )
        dequantized = (grouped * scale[expert].unsqueeze(-1)).reshape(
            end - start, logical_weight.size(1), logical_weight.size(2)
        )
        # FP32 accumulation over BF16 inputs, matching the kernel's accumulator.
        reference[start:end] = torch.einsum(
            "rk,rnk->rn",
            routed_inputs[start:end].float(),
            dequantized.float(),
        )
    return reference, rows


def pack_checkpoint_uint4(weight: torch.Tensor) -> torch.Tensor:
    """Pack eight little-endian 4-bit codes into each INT32 checkpoint word."""

    words = weight.to(torch.int64).reshape(*weight.shape[:-1], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=weight.device) * 4
    return (words << shifts).sum(dim=-1).to(torch.int32)


def generate_indexed_case(case: IndexedCase, device: torch.device) -> IndexedTensors:
    """Generate and pack all tensors required by one indexed W4A16 case."""

    from chord_kernels.operator import (
        IndexedLayerMeta,
        pack_w4a16,
        select_indexed_profile,
    )

    profile = select_indexed_profile(case.profile, device=device)
    meta = IndexedLayerMeta(
        shape_n=case.n,
        shape_k=case.k,
        num_experts=case.num_experts,
        profile=profile,
    )
    # Block-M is selected from the routed row count for both projections: the
    # kernel schedules over routed rows regardless of the input row count.
    kernel_config = meta.kernel_config(case.routed_m)
    generator = torch.Generator(device=device).manual_seed(case.seed)
    activation = torch.randn(
        (case.input_rows, case.k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    logical_weight = torch.randint(
        0,
        16,
        (case.num_experts, case.n, case.k),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    scale = (
        torch.rand(
            (case.num_experts, case.n, case.k // 32),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.04
        + 0.005
    ).to(torch.bfloat16)
    prepared_weight = pack_w4a16(logical_weight, scale, layout=profile.layout)
    # Routing reproduces the upstream bench_humming.py draw exactly: that
    # benchmark seeds the global CUDA RNG with the token count immediately
    # before scoring experts, and a fresh Generator with the same seed yields
    # the same stream.  Kernel time is sensitive to the draw — per-expert
    # padding makes the active block count vary by several percent between
    # seeds — so matching the draw keeps the printed `us` directly comparable
    # with the Humming tables for the same token count.
    routing_generator = torch.Generator(device=device).manual_seed(case.m)
    topk_ids, sorted_ids, expert_ids, num_tokens_padded = generate_indexed_routing(
        num_tokens=case.m,
        top_k=case.top_k,
        num_experts=case.num_experts,
        block_m=kernel_config.block_m,
        device=device,
        generator=routing_generator,
        distribution=case.distribution,
    )
    output = torch.empty((case.routed_m, case.n), dtype=torch.bfloat16, device=device)
    return IndexedTensors(
        profile=profile,
        kernel_config=kernel_config,
        activation=activation,
        logical_weight=logical_weight,
        scale=scale,
        prepared_weight=prepared_weight,
        topk_ids=topk_ids,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        output=output,
    )


# Model provenance for the production shapes below.
#
# All of them come from the Kimi K2.5 MoE architecture, a main open-weight LLM
# family shipping INT4 weights (K2.5/K2.6/K2.7), and so the reason a W4A16 kernel
# exists at all.  Its relevant config is:
#
#     num_experts          384     (routed experts, excluding the shared expert)
#     top_k                  8
#     hidden_size         7168
#     moe_intermediate    2048
#
# "EP8" is expert parallelism over 8 GPUs: the 384 routed experts are split into
# 8 groups, so each GPU owns 384 / 8 = 48 of them.  It does NOT mean the tensor
# is sliced 8 ways -- every expert stays whole, which is why the per-expert N and
# K below are the model's full dimensions:
#
#     gate/up   N = 2 * moe_intermediate = 4096   K = hidden_size      = 7168
#     down      N = hidden_size          = 7168   K = moe_intermediate = 2048
#
# gate/up emits 2x the intermediate width because the gate and up projections are
# fused into one GEMM before SwiGLU; down then consumes the 2048-wide result.
_EP8_GATE_UP = {"n": 4096, "k": 7168}
_EP8_DOWN = {"n": 7168, "k": 2048}
_EP8 = {"num_experts": 48, "top_k": 8}

# TP8 shards the same model over the same 8 GPUs, but along the other axis.
# Instead of splitting the expert list, every rank keeps all 384 routed experts
# and slices `moe_intermediate` 2048 / 8 = 256 ways:
#
#     gate/up   N = 2 * (moe_intermediate / 8) = 512    K = hidden_size          = 7168
#     down      N = hidden_size               = 7168    K = moe_intermediate / 8 = 256
#
# So TP8 narrows exactly the dimension EP8 leaves whole: gate/up loses output
# width and down loses K depth.  The expert count is the full 384 rather than 48.
_TP8_GATE_UP = {"n": 512, "k": 7168}
_TP8_DOWN = {"n": 7168, "k": 256}
_TP8 = {"num_experts": 384, "top_k": 8}

# Chunked-prefill sizes for the whole EP8 system, not per GPU: serving splits a
# long prompt into chunks, and 16384 is the largest chunk in use ("chunked 16k").
#
# Under EP8 a token's top_k=8 routes are spread over all 384 experts, and this
# GPU owns 48 of them, so the routes landing locally are
#
#     T * top_k * (48 / 384) = T * 8 / 8 = T
#
# i.e. the local routed row count happens to equal the system token count.  The
# kernel is handed those local rows directly, so a case built from T uses
# m = T / top_k with the real top_k, giving routed_m = T.  Passing m = T instead
# would model eight times the work that actually reaches one GPU.
PREFILL_SYSTEM_TOKENS = (1024, 2048, 4096, 8192, 16384)

# Decode runs EP8 together with DP8, so each GPU receives its own token batch and
# these counts are already per GPU: bs_per_gpu * (mtp + 1), where every sequence
# contributes its own token plus its MTP speculative tokens.
DECODE_TOKENS_PER_GPU = (20, 30, 40, 50)


def _profile_cases(
    profile: str,
    routed_rows: tuple[int, ...],
    *,
    seed_base: int,
    shard: dict | None = None,
    gate_up_shape: dict | None = None,
    down_shape: dict | None = None,
) -> tuple[IndexedCase, ...]:
    """Sweep local routed row counts for one profile, gate/up first then down.

    ``routed_rows`` is the number of rows the kernel writes on this GPU.  Each
    case therefore uses ``m = routed / top_k`` so that ``routed_m == routed``,
    which is what selects the tuning bracket.  Keeping the real ``top_k`` also
    keeps the gate/up fan-out under test rather than degenerating the index map.

    The two projections stay in separate runs of rows so each one's scaling
    across the sweep can be read down a single column.

    ``shard``/``gate_up_shape``/``down_shape`` default to the EP8 expert count and
    shapes; the TP8 profile passes its own, which keeps this sweep the single
    place a token count turns into cases regardless of the shard axis.
    """

    shard = _EP8 if shard is None else shard
    gate_up_shape = _EP8_GATE_UP if gate_up_shape is None else gate_up_shape
    down_shape = _EP8_DOWN if down_shape is None else down_shape
    top_k = shard["top_k"]
    gate_up = [
        IndexedCase(
            profile, m=routed // top_k, seed=seed_base + index, **shard, **gate_up_shape
        )
        for index, routed in enumerate(routed_rows)
    ]
    down = [
        IndexedCase(
            profile,
            m=routed // top_k,
            seed=seed_base + len(routed_rows) + index,
            projection="down",
            **shard,
            **down_shape,
        )
        for index, routed in enumerate(routed_rows)
    ]
    return (*gate_up, *down)


# Prefill: local routed rows equal the system token count (see the note above).
# Decode: the per-GPU token count fans out to top_k routes on the local experts,
# because DP8 gives this GPU its own batch while EP8 keeps all 384 experts
# reachable from it.
_PREFILL_ROUTED_ROWS = PREFILL_SYSTEM_TOKENS
_DECODE_ROUTED_ROWS = tuple(
    tokens * _EP8["top_k"] for tokens in DECODE_TOKENS_PER_GPU
)

# TP8 sweeps the same token counts as EP8 prefill, to compare the two shard axes
# at equal serving load, but nothing is split by expert here: every rank holds a
# slice of all 384 experts, so every route is local and routed_m is 8x the EP8
# figure at the same token count (16384 tokens -> 131072 rows).  That is why the
# tuning table has to reach far past EP8's 16384-row maximum.
_TP8_ROUTED_ROWS = tuple(
    tokens * _TP8["top_k"] for tokens in PREFILL_SYSTEM_TOKENS
)


PERFORMANCE_CASES: tuple[IndexedCase, ...] = (
    *_profile_cases("h200_prefill_ep8", _PREFILL_ROUTED_ROWS, seed_base=20260900),
    *_profile_cases(
        "h200_tp8",
        _TP8_ROUTED_ROWS,
        seed_base=20260920,
        shard=_TP8,
        gate_up_shape=_TP8_GATE_UP,
        down_shape=_TP8_DOWN,
    ),
    *_profile_cases("h200_decode_ep8", _DECODE_ROUTED_ROWS, seed_base=20260940),
    *_profile_cases("blackwell_decode_ep8", _DECODE_ROUTED_ROWS, seed_base=20260980),
)


_PROFILE_CAPABILITY = {
    "h200_prefill_ep8": {(9, 0)},
    "h200_tp8": {(9, 0)},
    "h200_decode_ep8": {(9, 0)},
    "blackwell_decode_ep8": {(10, 0), (10, 3)},
}


def select_cases(capability: tuple[int, int]) -> list[IndexedCase]:
    """Return the declared cases this compute capability can actually run."""

    return [
        case
        for case in PERFORMANCE_CASES
        if capability in _PROFILE_CAPABILITY.get(case.profile, set())
    ]


__all__ = [
    "DECODE_TOKENS_PER_GPU",
    "PERFORMANCE_CASES",
    "PREFILL_SYSTEM_TOKENS",
    "Distribution",
    "IndexedCase",
    "IndexedTensors",
    "Projection",
    "align_routing_blocks",
    "generate_indexed_case",
    "generate_indexed_routing",
    "generate_topk_ids",
    "pack_checkpoint_uint4",
    "reference_indexed",
    "select_cases",
]
