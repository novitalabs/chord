# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

"""Small indexed-only W4A16 layer adapter.

The layer deliberately keeps the framework contract narrow: callers provide the
already aligned routing tensors produced by a MoE router, while this module owns
checkpoint weight unpacking, operator packing, and profile selection.  It is not a
replacement for a framework quantization method.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
from collections.abc import Mapping
from typing import Any, Literal

import torch

from chord_kernels.operator import dtypes
from chord_kernels.operator.api import (
    IndexedKernelConfig,
    _compatible_block_n,
    w4a16_indexed,
)
from chord_kernels.operator.packing import (
    WeightLayout,
    PreparedWeight,
    SUPPORTED_INDEXED_COMPUTE_CAPABILITIES,
    pack_w4a16,
)

IndexedMode = Literal["prefill", "decode"]

# P/D role bit for SM90 profile selection, mirroring upstream Humming's
# HUMMING_INT_SM90_DECODE (default OFF -> the prefill/WGMMA path).  A serving
# framework such as vLLM launches prefill and decode instances from the same
# code path, so the instance role has to come from the environment: the decode
# launcher sets CHORD_SM90_DECODE=1; prefill launchers leave it unset or set 0.
# The variable only fills in the role when neither an explicit profile name nor
# a mode/role argument decides it, and only on SM90 — Blackwell publishes a
# decode profile only, so there is no role to select there.  It is read when
# auto selection resolves against a CUDA device, which happens before the
# weight is packed; the packed layout cannot switch at runtime.
_SM90_DECODE_ENV = "CHORD_SM90_DECODE"


def indexed_mode_from_env() -> IndexedMode | None:
    """Read the SM90 P/D role bit from ``CHORD_SM90_DECODE``.

    Returns ``"decode"`` for ``1``, ``"prefill"`` for ``0``, and ``None`` when
    the variable is unset or empty (callers then apply the upstream default of
    prefill).  Any other value is rejected instead of being silently treated
    as one of the roles.
    """
    value = os.getenv(_SM90_DECODE_ENV)
    if value is None or not value.strip():
        return None
    value = value.strip()
    if value == "1":
        return "decode"
    if value == "0":
        return "prefill"
    raise ValueError(
        f"{_SM90_DECODE_ENV} must be '1' (decode) or '0' (prefill), got {value!r}"
    )


@dataclasses.dataclass(frozen=True)
class IndexedLayerProfile:
    """A fixed weight layout and launch schedule for one serving regime."""

    name: str
    device_major: int
    mode: IndexedMode
    expert_parallel_size: int
    layout: WeightLayout
    swap_ab: bool
    block_m: int
    compute_capabilities: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("profile name must be a non-empty string")
        if (
            isinstance(self.device_major, bool)
            or not isinstance(self.device_major, int)
            or self.device_major not in (9, 10)
        ):
            raise ValueError(
                "indexed W4A16 profiles must target SM90 or SM100/SM103, got "
                f"device_major={self.device_major!r}"
            )
        capabilities = self.compute_capabilities
        if not isinstance(capabilities, tuple) or not capabilities or any(
            not isinstance(capability, tuple)
            or len(capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in capability
            )
            or capability[0] != self.device_major
            or capability not in SUPPORTED_INDEXED_COMPUTE_CAPABILITIES
            for capability in capabilities
        ):
            raise ValueError(
                "profile compute_capabilities must contain supported (major, minor) "
                f"pairs matching device_major={self.device_major}, got {capabilities!r}"
            )
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("profile compute_capabilities must not contain duplicates")
        if self.mode not in ("prefill", "decode"):
            raise ValueError(
                f"profile mode must be 'prefill' or 'decode', got {self.mode!r}"
            )
        if (
            isinstance(self.expert_parallel_size, bool)
            or not isinstance(self.expert_parallel_size, int)
            or self.expert_parallel_size != 8
        ):
            raise ValueError(
                "indexed W4A16 profiles are published only for expert_parallel_size=8"
            )
        if self.layout not in ("mma", "wgmma"):
            raise ValueError(
                f"profile layout must be 'mma' or 'wgmma', got {self.layout!r}"
            )
        if self.layout == "wgmma" and self.swap_ab:
            raise ValueError("WGMMA profiles cannot use swap_ab")
        if (
            isinstance(self.block_m, bool)
            or not isinstance(self.block_m, int)
            or self.block_m <= 0
            or self.block_m % 8
        ):
            raise ValueError(
                f"profile block_m must be a positive multiple of 8, got {self.block_m!r}"
            )

    @property
    def compute_capability(self) -> tuple[int, int]:
        return self.compute_capabilities[0]

    @property
    def role(self) -> IndexedMode:
        return self.mode

    def validate_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError(
                f"profile {self.name!r} requires a CUDA device, got {device}"
            )
        capability = torch.cuda.get_device_capability(device)
        if capability not in self.compute_capabilities:
            gpu = (
                "Hopper SM90"
                if self.device_major == 9
                else "Blackwell SM100/SM103"
            )
            supported = ", ".join(
                f"{major}.{minor}"
                for major, minor in self.compute_capabilities
            )
            raise RuntimeError(
                f"profile {self.name!r} requires {gpu} compute capability "
                f"{supported}, got {capability[0]}.{capability[1]} on {device}"
            )


@dataclasses.dataclass(frozen=True)
class IndexedLayerMeta:
    """Small, serializable metadata object used by framework adapters.

    A framework keeps this object on ``layer.humming_metas`` and reads its
    logical dimensions and tensor-name properties while preparing a layer.  The
    operator has one fixed BF16/INT4/group-32 schema, so the fields below carry
    only what indexed MoE integration needs.
    """

    shape_n: int
    shape_k: int
    num_experts: int
    profile: IndexedLayerProfile
    sublayer_name: str = ""
    pad_shape_n: int = 0
    pad_shape_k: int = 0
    a_dtype: Any = dtypes.bfloat16
    b_dtype: Any = dtypes.uint4
    c_dtype: Any = dtypes.bfloat16
    bs_dtype: Any = dtypes.bfloat16
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 32
    weight_scale_group_size_n: int = 1
    weight_scale_type: str = "group"
    has_bias: bool = False
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    use_int_weight_scale: bool = False
    use_fused_e8m0_scale: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("shape_n", self.shape_n),
            ("shape_k", self.shape_k),
            ("num_experts", self.num_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        if self.shape_n <= 0 or self.shape_k <= 0 or self.num_experts <= 0:
            raise ValueError("shape_n, shape_k, and num_experts must be positive")
        if self.pad_shape_n < 0 or self.pad_shape_k < 0:
            raise ValueError("pad_shape_n and pad_shape_k must be non-negative")

    @property
    def name_prefix(self) -> str:
        return f"{self.sublayer_name}_" if self.sublayer_name else ""

    @property
    def weight_name(self) -> str:
        return self.name_prefix + "weight"

    @property
    def weight_scale_name(self) -> str:
        return self.name_prefix + "weight_scale"

    @property
    def zero_point_name(self) -> str:
        return self.name_prefix + "zero_point"

    @property
    def global_scale_name(self) -> str:
        return self.name_prefix + "global_scale"

    @property
    def bias_name(self) -> str:
        return self.name_prefix + "bias"

    @property
    def param_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def layout(self) -> WeightLayout:
        return self.profile.layout

    @property
    def swap_ab(self) -> bool:
        return self.profile.swap_ab

    @property
    def block_m(self) -> int:
        return self.profile.block_m

    @property
    def block_shape(self) -> tuple[int, int, int]:
        """The profile's fallback ``(M, N, K)`` tile.

        Consistent with :attr:`block_m` and :attr:`swap_ab`: this is the
        conservative profile tile, not a routed-M tuning row.  Use
        :meth:`kernel_config` with a real routed-M for the launch schedule.
        """
        return (
            self.profile.block_m,
            _compatible_block_n(self.shape_n),
            64,
        )

    def kernel_config(self, valid_shape_m: int = 0) -> IndexedKernelConfig:
        return _select_indexed_kernel_config(self, valid_shape_m)

    @property
    def mma_type(self) -> str:
        return self.profile.layout

    @property
    def compute_capability(self) -> tuple[int, int]:
        return self.profile.compute_capability

    @property
    def compute_capabilities(self) -> tuple[tuple[int, int], ...]:
        return self.profile.compute_capabilities

    @property
    def role(self) -> IndexedMode:
        return self.profile.mode

    @property
    def weight_nbytes(self) -> int:
        # INT4 weights plus BF16 group-32 scales, across all local experts.
        # This follows the upstream adapter's weight_nbytes contract and is used
        # by framework memory estimators when sizing an expert parameter shard.
        per_expert = self.shape_n * self.shape_k // 2
        per_expert += self.shape_n * (self.shape_k // 32) * 2
        return per_expert * self.num_experts

    def estimate_bound_min_shape_m(self, use_f16_accum: bool = False) -> int:
        del use_f16_accum
        return 0

    def to_str(self) -> str:
        """Return the JSON form expected by Humming-style kernel adapters."""
        return json.dumps(
            {
                "shape_n": self.shape_n,
                "shape_k": self.shape_k,
                "pad_shape_n": self.pad_shape_n,
                "pad_shape_k": self.pad_shape_k,
                "num_experts": self.num_experts,
                "a_dtype": str(self.a_dtype),
                "b_dtype": str(self.b_dtype),
                "c_dtype": str(self.c_dtype),
                "bs_dtype": str(self.bs_dtype),
                "input_scale_group_size": self.input_scale_group_size,
                "weight_scale_group_size": self.weight_scale_group_size,
                "weight_scale_group_size_n": self.weight_scale_group_size_n,
                "weight_scale_type": self.weight_scale_type,
                "has_bias": self.has_bias,
                "has_zero_point": self.has_zero_point,
                "is_fp_zero_point": self.is_fp_zero_point,
                "sublayer_name": self.sublayer_name,
            }
        )


# Names are intentionally explicit: changing a profile changes the physical
# weight layout, so an implicit architecture guess would be unsafe at load time.
_PROFILES: dict[str, IndexedLayerProfile] = {
    "h200_prefill_ep8": IndexedLayerProfile(
        name="h200_prefill_ep8",
        device_major=9,
        mode="prefill",
        expert_parallel_size=8,
        layout="wgmma",
        swap_ab=False,
        block_m=16,
        compute_capabilities=((9, 0),),
    ),
    "h200_decode_ep8": IndexedLayerProfile(
        name="h200_decode_ep8",
        device_major=9,
        mode="decode",
        expert_parallel_size=8,
        layout="mma",
        swap_ab=True,
        block_m=8,
        compute_capabilities=((9, 0),),
    ),
    "blackwell_decode_ep8": IndexedLayerProfile(
        name="blackwell_decode_ep8",
        device_major=10,
        mode="decode",
        expert_parallel_size=8,
        layout="mma",
        swap_ab=True,
        block_m=8,
        compute_capabilities=((10, 0), (10, 3)),
    ),
}

# Public constants make profile selection convenient in framework registries while
# keeping the actual values immutable.  The dictionary is intentionally a copy so
# callers cannot mutate the module's internal lookup table.
H200_PREFILL_EP8 = _PROFILES["h200_prefill_ep8"]
H200_DECODE_EP8 = _PROFILES["h200_decode_ep8"]
BLACKWELL_DECODE_EP8 = _PROFILES["blackwell_decode_ep8"]
INDEXED_PROFILES = dict(_PROFILES)


def _indexed_config(
    meta: IndexedLayerMeta,
    *,
    block_m: int,
    block_n: int,
    block_k: int = 64,
    num_stages: int = 4,
    num_ctas_per_sm: int = 1,
    swap_ab: bool | None = None,
    use_stream_k: bool = False,
) -> IndexedKernelConfig:
    if swap_ab is None:
        swap_ab = meta.profile.swap_ab
    warp_n = 32 if meta.layout == "wgmma" else 64
    # Stream-K (K-dimension split with cross-CTA reduction) is opt-in per config;
    # decode and the small prefill tiles stay one-pass.
    return IndexedKernelConfig(
        block_shape=(block_m, block_n, block_k),
        warp_shape=(block_m, warp_n, min(block_k, 64)),
        num_stages=num_stages,
        num_ctas_per_sm=num_ctas_per_sm,
        swap_ab=swap_ab,
        use_stream_k=use_stream_k,
    )


# WGMMA accumulator register ceiling: block_m=176 already costs ~255 regs per
# thread, so this is the largest tile that stays off the spill cliff.
_H200_PREFILL_MAX_BLOCK_M = 176


def _min_block_count_block_m(valid_shape_m: int, num_experts: int) -> int:
    """Pick the block-M that minimizes total blocks over a modelled routing.

    Each expert's rows are padded to a whole block on its own, so the total
    block count is ``sum(ceil(count_e * 1.1 / block_m))`` rather than
    ``ceil(routed_m / block_m)``.  A single fixed random routing is sampled so
    the estimate tracks a real router rather than an even split.  The seed and
    RNG match upstream Humming's
    ``np.random.RandomState(seed=0).randint(0, num_experts, size=shape_m)``, so
    the chosen block-M is identical for the same shape.
    """

    import numpy as np

    experts = max(num_experts, 1)
    rows = max(valid_shape_m, 1)
    samples = np.random.RandomState(seed=0).randint(0, experts, size=rows)
    counts = np.bincount(samples, minlength=experts)

    best_block_m = 8
    best_blocks = None
    for block_m in range(8, _H200_PREFILL_MAX_BLOCK_M + 1, 8):
        blocks = int(np.ceil(counts * 1.1 / block_m).sum())
        if best_blocks is None or blocks < best_blocks:
            best_blocks, best_block_m = blocks, block_m
    return best_block_m


def _h200_prefill_block_m(valid_shape_m: int, num_experts: int, shape_k: int) -> int:
    """Size prefill block-M from routed tokens per expert.

    The governing quantity is tokens-per-expert, ``tok_e = routed_m /
    num_experts``, not ``routed_m`` alone: each expert's rows are padded up to a
    whole block independently, so two layers with the same routed_m but
    different expert counts want different tiles.

    For the EP-scale projections (N >= 4096, K > 512) the output is wide enough
    that ``N / block_n >= 16`` tiles keep the grid full even at large block-M, so
    per-expert M-padding dominates: split each expert's padded rows into the
    fewest blocks that fit under the 176 register ceiling, then size block-M to
    just cover them.  Past two blocks, deep-K shapes get one more 176 window
    because the long K loop amortizes the register spill, while short-K shapes
    settle on 128.

    Below ``tok_e`` 80 the padding model above would undershoot, so block-M comes
    from minimizing the total block count instead -- the regime there is block
    count and occupancy, not per-expert padding.
    """

    tokens_per_expert = valid_shape_m / max(num_experts, 1)
    if tokens_per_expert < 80:
        return _min_block_count_block_m(valid_shape_m, num_experts)

    # 1.1x covers the per-expert overshoot a random router leaves in each block.
    padded = tokens_per_expert * 1.1
    num_blocks = math.ceil(padded / _H200_PREFILL_MAX_BLOCK_M)
    if num_blocks <= 2:
        return math.ceil(padded / num_blocks / 8) * 8
    # Only the deep-K gate/up projection wins from a third 176 window; the
    # short-K down projection regresses there, so it goes straight to 128.
    if tokens_per_expert <= 352 and shape_k >= 4096:
        return _H200_PREFILL_MAX_BLOCK_M
    return 128


def _h200_prefill_use_stream_k(valid_shape_m: int, shape_n: int, shape_k: int) -> bool:
    """Decide whether prefill splits the K dimension across CTAs.

    Stream-K helps the deep-K gate/up projection at every prefill size, but the
    mid-K down projection (512 < K < 4096) regresses once routed_m is large: the
    M*N tiles already fill the grid, so the K-split only adds reduction and lock
    overhead.  Below that crossover stream-K still balances the tail, so it
    stays on.
    """

    if shape_n >= 4096 and 512 < shape_k < 4096 and valid_shape_m >= 5120:
        return False
    return True


def _select_indexed_kernel_config(
    meta: IndexedLayerMeta, valid_shape_m: int = 0
) -> IndexedKernelConfig:
    """Resolve the small, published schedule table for one routed M.

    The table deliberately covers only the three target profiles.  Unrecognised
    projection shapes use the conservative profile fallback instead of silently
    claiming a tuning result for a different model.
    """
    if isinstance(valid_shape_m, bool) or not isinstance(valid_shape_m, int):
        raise TypeError(
            f"valid_shape_m must be an int, got {type(valid_shape_m).__name__}"
        )
    if valid_shape_m < 0:
        raise ValueError(f"valid_shape_m must be non-negative, got {valid_shape_m}")
    m = max(valid_shape_m, 1)
    profile = meta.profile.name
    n, k, experts = meta.shape_n, meta.shape_k, meta.num_experts

    if profile == "h200_prefill_ep8" and (n, k) in {
        (4096, 7168),
        (7168, 2048),
    }:
        block_m = _h200_prefill_block_m(m, experts, k)
        use_stream_k = _h200_prefill_use_stream_k(m, n, k)
        if block_m <= 32:
            # Small blocks split by output width: the narrow gate/up
            # (N <= 4096) trades tile width for K depth, while the wide down
            # projection keeps block-N 256 and only doubles K.  Swapping the
            # two costs the down projection 13-16% at routed_m 112..856.
            if n <= 4096:
                return _indexed_config(
                    meta, block_m=block_m, block_n=128, block_k=256,
                    use_stream_k=use_stream_k,
                )
            return _indexed_config(
                meta, block_m=block_m, block_n=256, block_k=128,
                use_stream_k=use_stream_k,
            )
        # Above block-M 32 both projections use the wide 256x64 tile.  For
        # gate/up block-M 40..64 a generic N<=4096 heuristic would pick
        # 128x128, but 256x64 with the 2-CTA window below measures 12-15%
        # faster on H200 for these EP8 shapes (routed_m 863..2073).
        #
        # Forcing 2 CTAs/SM for the 40..80 block-M window at block_n=256
        # doubles resident warps to hide the cp.async + dequant latency.
        num_ctas_per_sm = 2 if 40 <= block_m <= 80 else 1
        return _indexed_config(
            meta, block_m=block_m, block_n=256, block_k=64,
            num_ctas_per_sm=num_ctas_per_sm,
            use_stream_k=use_stream_k,
        )

    if profile == "h200_decode_ep8" and (n, k) in {
        (4096, 7168),
        (7168, 2048),
    }:
        tokens_per_expert = m / max(experts, 1) / 0.9
        block_m = (
            8 if tokens_per_expert <= 6 else (16 if tokens_per_expert <= 13 else 24)
        )
        return _indexed_config(
            meta,
            block_m=block_m,
            block_n=256,
            num_ctas_per_sm=4,
            swap_ab=True,
        )

    if profile == "blackwell_decode_ep8" and (n, k, experts) in {
        (4096, 7168, 48),
        (7168, 2048, 48),
    }:
        # SM100 and SM103 share this table.  The two projections are tuned
        # independently; a host that shares one routing across both passes the
        # routing's block-M through the ``block_m`` forward argument, which
        # overrides the row's M tile while keeping the projection's N/K tile
        # (see _config_with_block_m).
        if (n, k) == (7168, 2048):
            # down is short-K (32 blocks of 64): the swap-AB padding-skip
            # tiles stay one-pass, and the K-split only repays on the large
            # non-swap tiles.
            if m <= 272:
                return _indexed_config(
                    meta, block_m=8, block_n=256, num_ctas_per_sm=4,
                    swap_ab=True,
                )
            if m <= 848:
                return _indexed_config(
                    meta, block_m=16, block_n=256, num_ctas_per_sm=4,
                    swap_ab=True,
                )
            if m <= 1424:
                return _indexed_config(
                    meta, block_m=32, block_n=512, swap_ab=False,
                    use_stream_k=True,
                )
            return _indexed_config(
                meta, block_m=48, block_n=512, swap_ab=False,
                use_stream_k=True,
            )

        # gate/up is deep-K (112 blocks of 64), which leaves a long scheduling
        # tail, so stream-K repays its lock overhead at every decode size.
        if m <= 320:
            # A block holds one populated 8-token mma-N tile; the small
            # swap-AB tile at 4 CTAs/SM keeps the machine latency-bound
            # rather than padding-bound.
            return _indexed_config(
                meta, block_m=8, block_n=256, num_ctas_per_sm=4,
                swap_ab=True, use_stream_k=True,
            )
        if m <= 736:
            # The second token tile fills; the wide non-swap tile with a
            # deep K block wins over both the swap rows and (16,512,64).
            return _indexed_config(
                meta, block_m=16, block_n=512, block_k=128,
                swap_ab=False, use_stream_k=True,
            )
        if m <= 1424:
            return _indexed_config(
                meta, block_m=32, block_n=512, swap_ab=False,
                use_stream_k=True,
            )
        return _indexed_config(
            meta, block_m=48, block_n=512, swap_ab=False,
            use_stream_k=True,
        )

    return IndexedKernelConfig.default(
        layout=meta.layout,
        block_m=meta.profile.block_m,
        block_n=_compatible_block_n(meta.shape_n),
        swap_ab=meta.profile.swap_ab,
    )


def _indexed_tuning_rows(meta: IndexedLayerMeta) -> list[tuple[int, int, dict]]:
    """Return ``(min_m, max_m, config)`` rows exactly matching the resolver.

    Every routed-M is swept through the same resolver the forward path uses and
    runs of equal configs are merged into rows.  Boundaries are therefore
    discovered, never hand-maintained, and a framework that aligns routing to a
    row's block-M is guaranteed the forward path launches that same tile for
    every routed-M inside the row.
    """
    rows = [
        (lower, upper, dict(config))
        for lower, upper, config in _swept_tuning_rows(meta)
    ]
    return rows


@functools.lru_cache(maxsize=64)
def _swept_tuning_rows(
    meta: IndexedLayerMeta,
) -> tuple[tuple[int, int, dict], ...]:
    # The last schedule change across the published profiles is prefill's
    # 176 -> 128 block-M step at tok_e = 352, i.e. routed_m = 352 * experts;
    # decode settles by tok_e ~ 13 and Blackwell by routed_m 1424.  The
    # sentinels then assert the final row really is constant.
    sweep_bound = max(4096, 368 * max(meta.num_experts, 1))

    rows: list[list] = []
    last_config: IndexedKernelConfig | None = None
    for m in range(1, sweep_bound + 1):
        config = _select_indexed_kernel_config(meta, m)
        if config == last_config:
            rows[-1][1] = m
        else:
            rows.append([m - 1, m, config])
            last_config = config
    rows[-1][1] = 1 << 30

    for sentinel in (2 * sweep_bound, 1 << 20, 1 << 29):
        if _select_indexed_kernel_config(meta, sentinel) != last_config:
            raise RuntimeError(
                "indexed tuning sweep bound is too small: the schedule still "
                f"changes at routed_m={sentinel} for profile "
                f"{meta.profile.name!r} (N={meta.shape_n}, K={meta.shape_k})"
            )

    frozen: list[tuple[int, int, dict]] = []
    for lower, upper, config in rows:
        row = config.to_dict()
        row.update({"block_m": config.block_m, "layout": meta.layout})
        frozen.append((lower, upper, row))
    return tuple(frozen)


def _validate_indexed_compute_config(compute_config: object) -> None:
    """Reject a framework compute config that selects another GEMM family."""
    value = compute_config
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("compute_config must be valid JSON") from exc
    if isinstance(value, Mapping):
        gemm_type = value.get("gemm_type")
        unsupported = [
            name
            for name in ("use_f16_accum", "use_batch_invariant")
            if bool(value.get(name, False))
        ]
    else:
        gemm_type = getattr(value, "gemm_type", None)
        unsupported = [
            name
            for name in ("use_f16_accum", "use_batch_invariant")
            if bool(getattr(value, name, False))
        ]
    if unsupported:
        raise ValueError(
            "indexed W4A16 does not implement compute option(s): "
            + ", ".join(unsupported)
        )
    if gemm_type is None:
        return
    gemm_type = getattr(gemm_type, "value", gemm_type)
    if str(gemm_type).lower() != "indexed":
        raise ValueError(
            "the indexed W4A16 layer only accepts compute_config gemm_type='indexed'"
        )


def _resolve_tuning_config(
    meta: IndexedLayerMeta,
    selected_shape_m: int,
    tuning_config: object | None,
    *,
    default: IndexedKernelConfig,
) -> IndexedKernelConfig:
    """Parse the small tuning format used by Humming/vLLM."""
    if tuning_config is None:
        return default
    value = tuning_config
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tuning_config must be valid JSON") from exc
    if isinstance(value, Mapping):
        row = dict(value)
        row.setdefault("swap_ab", meta.swap_ab)
        if row.get("layout", meta.layout) != meta.layout:
            raise ValueError("tuning_config layout does not match the packed weight")
        return IndexedKernelConfig.from_dict(row)
    if isinstance(value, list):
        for lower, upper, row in value:
            if selected_shape_m > int(lower) and selected_shape_m <= int(upper):
                row = dict(row)
                row.setdefault("swap_ab", meta.swap_ab)
                if row.get("layout", meta.layout) != meta.layout:
                    raise ValueError(
                        "tuning_config layout does not match the packed weight"
                    )
                return IndexedKernelConfig.from_dict(row)
        raise ValueError(
            f"no indexed tuning row covers valid_shape_m={selected_shape_m}"
        )
    raise TypeError(
        "tuning_config must be a JSON string, dict, list, or None; "
        f"got {type(tuning_config).__name__}"
    )


def _normalise_mode(mode: str | None) -> IndexedMode | None:
    if mode is None:
        return None
    if mode not in ("prefill", "decode"):
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
    return mode  # type: ignore[return-value]


def select_indexed_profile(
    profile: str | IndexedLayerProfile | None = "auto",
    *,
    mode: IndexedMode | None = None,
    role: IndexedMode | None = None,
    device: torch.device | None = None,
    expert_parallel_size: int = 8,
) -> IndexedLayerProfile:
    """Resolve an explicit profile or select one from device capability.

    ``auto`` is resolved only when this function is called, so layer
    construction remains safe on CPU/meta devices; a serving framework can
    construct a layer first and resolve it after moving weights to CUDA.  On
    SM90 the P/D role comes from an explicit ``mode``/``role`` argument, else
    ``CHORD_SM90_DECODE``, else the prefill default; Blackwell resolves to its
    only published profile (decode) regardless of the variable.
    """
    selected_mode = _normalise_mode(mode)
    if role is not None:
        role_mode = _normalise_mode(role)
        if selected_mode is not None and selected_mode != role_mode:
            raise ValueError("mode and role must agree when both are provided")
        selected_mode = role_mode

    if isinstance(profile, IndexedLayerProfile):
        resolved = profile
    else:
        if profile is not None and not isinstance(profile, str):
            raise TypeError(
                "profile must be a profile name, IndexedLayerProfile, or None; "
                f"got {type(profile).__name__}"
            )
        name = "auto" if profile is None else profile

        if name == "auto":
            if device is not None and torch.device(device).type != "cuda":
                raise RuntimeError(
                    f"profile='auto' requires a CUDA device for capability detection, got {device}"
                )
            capability = torch.cuda.get_device_capability(
                None if device is None else torch.device(device)
            )
            if capability in ((10, 0), (10, 3)):
                # Blackwell publishes a decode profile only, so the SM90 role
                # bit does not apply and CHORD_SM90_DECODE is ignored here; an
                # explicit mode='prefill' is still rejected by the mode check
                # below.
                name = "blackwell_decode_ep8"
            elif capability == (9, 0):
                # SM90 has both roles.  An explicit mode wins; otherwise the
                # CHORD_SM90_DECODE bit decides, defaulting to prefill/WGMMA
                # exactly like upstream HUMMING_INT_SM90_DECODE unset/0.
                sm90_role = selected_mode or indexed_mode_from_env() or "prefill"
                name = f"h200_{sm90_role}_ep8"
            else:
                raise RuntimeError(
                    "no indexed W4A16 profile for compute capability "
                    f"{capability[0]}.{capability[1]}"
                )
        try:
            resolved = _PROFILES[name]
        except KeyError as exc:
            valid = ", ".join(sorted(_PROFILES))
            raise ValueError(
                f"unknown indexed profile {profile!r}; choose one of {valid}"
            ) from exc

    if selected_mode is not None and resolved.mode != selected_mode:
        raise ValueError(
            f"profile {resolved.name!r} is for {resolved.mode}, not {selected_mode}"
        )

    if (
        isinstance(expert_parallel_size, bool)
        or not isinstance(expert_parallel_size, int)
        or expert_parallel_size != 8
    ):
        raise ValueError(
            "the published indexed W4A16 profiles are for expert_parallel_size=8, "
            f"got {expert_parallel_size!r}"
        )
    if resolved.expert_parallel_size != expert_parallel_size:
        raise ValueError(
            f"profile {resolved.name!r} expects expert_parallel_size="
            f"{resolved.expert_parallel_size}, got {expert_parallel_size}"
        )
    if device is not None and torch.device(device).type == "cuda":
        resolved.validate_device(torch.device(device))
    return resolved


def unpack_packed_uint4(
    weight: torch.Tensor, shape_k: int | None = None
) -> torch.Tensor:
    """Unpack Humming/vLLM INT32 checkpoint words into one 4-bit code per INT32.

    Each INT32 word contains eight nibbles in little-endian bit order.  The
    returned tensor has the same leading dimensions and a final dimension eight
    times larger than the input.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if weight.dtype != torch.int32:
        raise TypeError(f"packed weight must be torch.int32, got {weight.dtype}")
    if weight.ndim < 1 or not weight.is_contiguous():
        raise ValueError(
            "packed weight must be contiguous and have at least one dimension"
        )
    if shape_k is not None:
        if isinstance(shape_k, bool) or not isinstance(shape_k, int) or shape_k <= 0:
            raise ValueError(f"shape_k must be a positive integer, got {shape_k!r}")
        if shape_k != weight.shape[-1] * 8:
            raise ValueError(
                f"packed weight has K={weight.shape[-1] * 8}, expected shape_k={shape_k}"
            )

    # Convert through int64 so right shifts of negative signed INT32 words are
    # well-defined after masking to the original unsigned 32 bits.
    words = weight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64, device=weight.device) * 4
    codes = (words.unsqueeze(-1) >> shifts) & 0xF
    return codes.reshape(*weight.shape[:-1], weight.shape[-1] * 8).to(torch.int32)


def _replace_parameter(
    module: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    parameters = getattr(module, "_parameters", None)
    old_parameter = parameters.get(name) if isinstance(parameters, dict) else None
    parameter = torch.nn.Parameter(tensor, requires_grad=False)
    # vLLM attaches loader/sharding metadata directly to Parameters. Preserve
    # that metadata when the compact checkpoint tensor is replaced by a packed
    # kernel tensor; ordinary PyTorch Parameters simply have an empty dict.
    if old_parameter is not None:
        for key, value in getattr(old_parameter, "__dict__", {}).items():
            setattr(parameter, key, value)
    if isinstance(parameters, dict) and name in parameters:
        parameters[name] = parameter
    else:
        setattr(module, name, parameter)


class IndexedW4A16Layer(torch.nn.Module):
    """Indexed-only W4A16 layer suitable for a vLLM weight-loader adapter.

    ``weight`` is accepted in the usual Humming checkpoint form ``[E, N, K/8]``
    (eight 4-bit codes per INT32) or as unpacked codes ``[E, N, K]``.  ``transform``
    converts it to the physical Humming layout.  Construction and CPU/meta
    checkpoint loading do not touch CUDA; the first CUDA transform performs the
    actual JIT-backed repack.
    """

    def __init__(
        self,
        shape_n: int | None = None,
        shape_k: int | None = None,
        num_experts: int = 1,
        *,
        n: int | None = None,
        k: int | None = None,
        profile: str | IndexedLayerProfile | None = "auto",
        mode: IndexedMode | None = None,
        role: IndexedMode | None = None,
        expert_parallel_size: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if shape_n is None:
            shape_n = n
        elif n is not None and n != shape_n:
            raise ValueError("shape_n and n must agree")
        if shape_k is None:
            shape_k = k
        elif k is not None and k != shape_k:
            raise ValueError("shape_k and k must agree")
        if shape_n is None or shape_k is None:
            raise TypeError("shape_n/shape_k (or n/k) are required")
        for name, value in (
            ("shape_n", shape_n),
            ("shape_k", shape_k),
            ("num_experts", num_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if shape_n % 128 != 0:
            raise ValueError(f"shape_n must be divisible by 128, got {shape_n}")
        if shape_k % 64 != 0:
            raise ValueError(f"shape_k must be divisible by 64, got {shape_k}")
        if torch_dtype != torch.bfloat16:
            raise ValueError(
                "the indexed W4A16 layer currently supports BF16 activation/scale, "
                f"got torch_dtype={torch_dtype}"
            )

        self.shape_n = shape_n
        self.shape_k = shape_k
        self.num_experts = num_experts
        self.expert_parallel_size = expert_parallel_size
        self.torch_dtype = torch_dtype
        self._profile_spec = profile
        # Only an explicit mode/role is recorded; an auto layer leaves the role
        # open so resolution can pick it per architecture (SM90 reads
        # CHORD_SM90_DECODE with a prefill default, Blackwell is decode-only).
        self._mode = _normalise_mode(mode)
        if role is not None:
            role_mode = _normalise_mode(role)
            if self._mode is not None and self._mode != role_mode:
                raise ValueError("mode and role must agree when both are provided")
            self._mode = role_mode
        # Resolve explicit profiles now (without querying CUDA); leave auto until
        # the first real CUDA weight is transformed.
        if isinstance(profile, IndexedLayerProfile):
            self.profile = select_indexed_profile(
                profile,
                mode=self._mode,
                expert_parallel_size=expert_parallel_size,
            )
        elif profile not in (None, "auto"):
            self.profile = select_indexed_profile(
                profile,
                mode=None if mode is None and role is None else self._mode,
                expert_parallel_size=expert_parallel_size,
            )
        else:
            self.profile = None

        alloc_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.weight = torch.nn.Parameter(
            torch.empty(
                (num_experts, shape_n, shape_k // 8),
                dtype=torch.int32,
                device=alloc_device,
            ),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.empty(
                (num_experts, shape_n, shape_k // 32),
                # The indexed kernel currently consumes BF16 group scales.  The
                # activation dtype remains configurable independently below.
                dtype=torch.bfloat16,
                device=alloc_device,
            ),
            requires_grad=False,
        )
        self._weight_format = "checkpoint_packed"
        self._prepared_weight: PreparedWeight | object | None = None
        # Framework adapters read this mapping to discover the logical shape and
        # fixed profile.  The attribute name matches the one Humming-derived
        # adapters already look for.
        self.humming_metas: dict[str, IndexedLayerMeta] = {}
        self._set_humming_meta("")

    @property
    def indexed_profile(self) -> IndexedLayerProfile:
        if self.profile is None:
            if not self.weight.is_cuda:
                # A CPU/meta layer cannot query hardware.  Use the matching
                # Hopper schedule as a provisional layout — the SM90 role
                # comes from the explicit mode, else CHORD_SM90_DECODE, else
                # the prefill default.  An auto layer moved to Blackwell must
                # be recreated with the explicit Blackwell profile so weights
                # are never silently repacked for another layout.
                role = self._mode or indexed_mode_from_env() or "prefill"
                return select_indexed_profile(
                    f"h200_{role}_ep8",
                    expert_parallel_size=self.expert_parallel_size,
                )
            self.profile = select_indexed_profile(
                "auto",
                mode=self._mode,
                device=self.weight.device,
                expert_parallel_size=self.expert_parallel_size,
            )
        return self.profile

    @property
    def layout(self) -> WeightLayout:
        return self.indexed_profile.layout

    @property
    def block_m(self) -> int:
        return self.indexed_profile.block_m

    def _set_humming_meta(self, sublayer_name: str = "") -> IndexedLayerMeta:
        if not isinstance(sublayer_name, str):
            raise TypeError(
                f"sublayer_name must be a string, got {type(sublayer_name).__name__}"
            )
        meta = IndexedLayerMeta(
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            num_experts=self.num_experts,
            profile=self.indexed_profile,
            sublayer_name=sublayer_name,
        )
        self.humming_metas[sublayer_name] = meta
        return meta

    def _refresh_humming_metas(self) -> None:
        profile = self.indexed_profile
        if not self.humming_metas:
            self._set_humming_meta("")
            return
        for name, meta in tuple(self.humming_metas.items()):
            self.humming_metas[name] = dataclasses.replace(
                meta,
                shape_n=self.shape_n,
                shape_k=self.shape_k,
                num_experts=self.num_experts,
                profile=profile,
            )

    def load_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        packed: bool | None = None,
        transform: bool = True,
    ) -> "IndexedW4A16Layer":
        """Load checkpoint tensors and optionally prepare the kernel layout."""
        if not isinstance(weight, torch.Tensor) or not isinstance(
            weight_scale, torch.Tensor
        ):
            raise TypeError("weight and weight_scale must be torch.Tensor objects")
        if weight.device != weight_scale.device:
            raise ValueError(
                "weight and weight_scale must be on the same device, got "
                f"{weight.device} and {weight_scale.device}"
            )
        if weight.dtype != torch.int32:
            raise TypeError(f"weight must be torch.int32, got {weight.dtype}")
        if weight_scale.dtype != torch.bfloat16:
            raise TypeError(
                f"weight_scale must be torch.bfloat16, got {weight_scale.dtype}"
            )
        if weight.ndim == 2 and self.num_experts == 1:
            weight = weight.unsqueeze(0)
        if weight_scale.ndim == 2 and self.num_experts == 1:
            weight_scale = weight_scale.unsqueeze(0)
        if (
            weight.ndim != 3
            or weight.shape[0] != self.num_experts
            or weight.shape[1] != self.shape_n
        ):
            raise ValueError(
                f"weight must have leading shape ({self.num_experts}, {self.shape_n}, _), "
                f"got {tuple(weight.shape)}"
            )
        if weight_scale.shape != (self.num_experts, self.shape_n, self.shape_k // 32):
            raise ValueError(
                "weight_scale must have shape "
                f"{self.num_experts, self.shape_n, self.shape_k // 32}, "
                f"got {tuple(weight_scale.shape)}"
            )
        if packed is None:
            packed = weight.shape[-1] == self.shape_k // 8
        if packed:
            if weight.shape[-1] != self.shape_k // 8:
                raise ValueError(
                    f"packed weight must have K/8={self.shape_k // 8} words, got {weight.shape[-1]}"
                )
            weight_format = "checkpoint_packed"
        else:
            if weight.shape[-1] != self.shape_k:
                raise ValueError(
                    f"unpacked weight must have K={self.shape_k} codes, got {weight.shape[-1]}"
                )
            weight_format = "unpacked"

        weight_device = (
            weight.device
            if self.weight.is_meta
            or (self.weight.device.type == "cpu" and weight.is_cuda)
            else self.weight.device
        )
        scale_device = (
            weight_scale.device
            if self.weight_scale.is_meta
            or (self.weight_scale.device.type == "cpu" and weight_scale.is_cuda)
            else self.weight_scale.device
        )
        _replace_parameter(self, "weight", weight.contiguous().to(weight_device))
        _replace_parameter(
            self,
            "weight_scale",
            weight_scale.contiguous().to(scale_device),
        )
        self._weight_format = weight_format
        self._prepared_weight = None
        if transform:
            self.transform()
        return self

    def load_from_unquantized(
        self,
        weight_uint4: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        transform: bool = True,
    ) -> "IndexedW4A16Layer":
        return self.load_weight(
            weight_uint4, weight_scale, packed=False, transform=transform
        )

    def transform(self) -> "IndexedW4A16Layer":
        """Convert checkpoint tensors to the profile-specific Humming layout."""
        if self._prepared_weight is not None:
            return self
        if not self.weight.is_cuda:
            # Keep checkpoint-packed weights compact while a framework constructs
            # or loads the module on CPU/meta. The first CUDA forward (or an
            # explicit process_weights_after_loading call) performs both steps.
            return self

        if (
            self._weight_format == "checkpoint_packed"
            and self.weight.shape[-1] == self.shape_k
        ):
            # Framework loaders sometimes assign an already-unpacked tensor
            # directly to ``layer.weight``.  Recognize that form without making
            # the caller set a private format flag.
            source_weight = self.weight
            source_is_packed = False
        elif self._weight_format == "checkpoint_packed":
            source_weight = self.weight
            source_is_packed = True
        elif self._weight_format == "unpacked":
            source_weight = self.weight
            source_is_packed = False
        elif self._weight_format == "kernel":
            return self
        else:
            raise RuntimeError(f"unknown weight format {self._weight_format!r}")
        # Profile resolution is deliberately deferred for auto layers.  Explicit
        # profiles are checked only once a CUDA tensor is actually available.
        profile = self.indexed_profile
        self._refresh_humming_metas()
        profile.validate_device(source_weight.device)

        prepared = pack_w4a16(
            source_weight.contiguous(),
            self.weight_scale.contiguous(),
            layout=profile.layout,
            packed=source_is_packed,
        )

        self._prepared_weight = prepared
        if isinstance(prepared, PreparedWeight):
            _replace_parameter(self, "weight", prepared.packed)
            _replace_parameter(self, "weight_scale", prepared.scale)
            self._weight_format = "kernel"
        return self

    def _apply(self, fn):
        # PreparedWeight is a dataclass rather than a registered module,
        # so refresh its tensor references when a framework moves this layer.
        result = super()._apply(fn)
        if isinstance(self._prepared_weight, PreparedWeight):
            self._prepared_weight = dataclasses.replace(
                self._prepared_weight,
                packed=self.weight,
                scale=self.weight_scale,
            )
            if self.weight.is_cuda:
                # The physical MMA/WGMMA layout is profile-specific.  A packed
                # H200 layer cannot be migrated to another architecture and
                # silently keep using the old schedule.
                self.indexed_profile.validate_device(self.weight.device)
        return result

    process_weights_after_loading = transform

    def forward(
        self,
        inputs: torch.Tensor,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        top_k: int | None = None,
        *,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        valid_shape_m: int = 0,
        # Framework routers (including vLLM) already produce aligned buffers;
        # skip GPU-to-CPU count reads and route/expert value scans by default.
        # Standalone callers can enable synchronized validation explicitly.
        validate_routing: bool = False,
        block_m: int | None = None,
        # These names are accepted by the upstream Humming layer delegation.
        # Indexed W4A16 has no grouped expert layout or runtime tuning object;
        # profile selection owns those choices.
        expert_layout: torch.Tensor | None = None,
        m_indices: torch.Tensor | None = None,
        compute_config: object | None = None,
        tuning_config: object | None = None,
        sublayer_name: str = "",
        **kwargs,
    ) -> torch.Tensor:
        del sublayer_name, kwargs
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        if expert_layout is not None or m_indices is not None:
            raise ValueError("indexed W4A16 uses sorted_ids/expert_ids routing only")
        if compute_config is not None:
            _validate_indexed_compute_config(compute_config)
        if (
            sorted_ids is None
            or expert_ids is None
            or num_tokens_padded is None
            or top_k is None
        ):
            raise TypeError(
                "forward requires sorted_ids, expert_ids, num_tokens_padded, and top_k"
            )
        if self._prepared_weight is None:
            self.transform()
        if self._prepared_weight is None:
            raise RuntimeError(
                "W4A16 weights are not packed for CUDA; move the layer to CUDA and "
                "call process_weights_after_loading()"
            )
        if (
            isinstance(self._prepared_weight, PreparedWeight)
            and not self.weight.is_cuda
        ):
            raise RuntimeError(
                "prepared indexed W4A16 weights must remain on CUDA; move the layer "
                "to CUDA and repack it before forward"
            )
        profile = self.indexed_profile
        selected_shape_m = valid_shape_m
        if selected_shape_m <= 0:
            selected_shape_m = inputs.size(0) * top_k
        profile_meta = self.humming_metas.get("")
        if not isinstance(profile_meta, IndexedLayerMeta):
            profile_meta = self._set_humming_meta("")
        kernel_config = profile_meta.kernel_config(selected_shape_m)
        kernel_config = _resolve_tuning_config(
            profile_meta,
            selected_shape_m,
            tuning_config,
            default=kernel_config,
        )
        if block_m is not None and block_m != kernel_config.block_m:
            kernel_config = _config_with_block_m(
                kernel_config, block_m, layout=profile.layout
            )
        return w4a16_indexed(
            inputs,
            self._prepared_weight,
            sorted_ids,
            expert_ids,
            num_tokens_padded,
            top_k,
            outputs=outputs,
            config=kernel_config,
            layout=profile.layout,
            swap_ab=kernel_config.swap_ab,
            valid_shape_m=valid_shape_m,
            validate_routing=validate_routing,
        )

    def forward_indexed(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_layer(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _parameter_tensor(layer: object, name: str) -> torch.Tensor:
    try:
        tensor = getattr(layer, name)
    except AttributeError as exc:
        raise AttributeError(f"layer is missing required tensor {name!r}") from exc
    if isinstance(tensor, torch.nn.Parameter):
        tensor = tensor.data
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"layer.{name} must be a torch.Tensor")
    return tensor


def _validate_indexed_schema(
    weight_schema: object | None, input_schema: object | None
) -> None:
    """Reject schemas the extracted kernel cannot represent."""

    def parse_dtype(value: object) -> object:
        if isinstance(value, str):
            try:
                return dtypes.DataType.from_str(value)
            except (TypeError, ValueError):
                return value
        return value

    def is_bfloat16(value: object) -> bool:
        value = parse_dtype(value)
        return (
            getattr(value, "num_bits", None) == 16
            and getattr(value, "is_floating_point_type", False)
            and getattr(value, "exponent_bits", None) == 8
            and getattr(value, "mantissa_bits", None) == 7
        )

    if weight_schema is not None:
        b_dtype = parse_dtype(getattr(weight_schema, "b_dtype", None))
        if b_dtype is None or (
            getattr(b_dtype, "num_bits", None) != 4
            or not getattr(b_dtype, "is_integer_type", False)
            or getattr(b_dtype, "is_signed", True)
        ):
            raise ValueError("indexed W4A16 requires a 4-bit unsigned weight schema")
        group_size = getattr(weight_schema, "weight_scale_group_size", 32)
        if group_size not in (None, 32):
            raise ValueError("indexed W4A16 requires weight scale group size 32")
        group_size_n = getattr(weight_schema, "weight_scale_group_size_n", 1)
        if group_size_n not in (None, 0, 1):
            raise ValueError(
                "indexed W4A16 requires one scale per output channel "
                "(weight_scale_group_size_n must be 0 or 1)"
            )
        scale_type = getattr(weight_schema, "weight_scale_type", None)
        scale_type = getattr(scale_type, "value", scale_type)
        if scale_type is not None and "group" not in str(scale_type).lower():
            raise ValueError("indexed W4A16 requires group weight scales")
        bs_dtype = getattr(weight_schema, "bs_dtype", dtypes.bfloat16)
        if bs_dtype is not None and not is_bfloat16(bs_dtype):
            raise ValueError("indexed W4A16 requires BF16 group scales")
        if getattr(weight_schema, "has_zero_point", False):
            raise ValueError("indexed W4A16 does not support an explicit zero point")
        if getattr(weight_schema, "has_bias", False):
            raise ValueError("indexed W4A16 does not support a weight bias")
        if getattr(weight_schema, "is_fp_zero_point", False):
            raise ValueError("indexed W4A16 does not support an explicit zero point")
        if getattr(weight_schema, "hadamard_block_size", 0) not in (None, 0, 1):
            raise ValueError("indexed W4A16 does not support Hadamard-rotated weights")
        if getattr(weight_schema, "use_fused_e8m0_scale", False):
            raise ValueError("indexed W4A16 requires BF16, not fused E8M0 scales")
    if input_schema is not None:
        a_dtype = parse_dtype(getattr(input_schema, "a_dtype", None))
        if a_dtype is not None and not is_bfloat16(a_dtype):
            raise ValueError("indexed W4A16 requires BF16 activations")
        if getattr(input_schema, "input_scale_group_size", 0) not in (None, 0):
            raise ValueError("indexed W4A16 does not support input scales")


def _host_profile(layer: object, kwargs: Mapping[str, Any]) -> IndexedLayerProfile:
    explicit = kwargs.get("profile")
    if explicit is None:
        # A live `indexed_profile` property re-resolves auto layers after they
        # move to CUDA, so it must win over any previously cached value; the
        # `_w4a16_profile` cache only serves foreign host layers that carry no
        # profile attribute of their own.
        explicit = getattr(layer, "indexed_profile", None)
    if explicit is None:
        explicit = getattr(layer, "w4a16_profile", None)
    if explicit is None:
        explicit = getattr(layer, "_w4a16_profile", None)
    if explicit is None:
        candidate = getattr(layer, "profile", None)
        if isinstance(candidate, (IndexedLayerProfile, str)):
            explicit = candidate
    if explicit is None:
        explicit = "auto"
    mode = kwargs.get("mode")
    if mode is None:
        mode = getattr(layer, "w4a16_mode", getattr(layer, "_mode", None))
    if mode is None and isinstance(explicit, IndexedLayerProfile):
        mode = explicit.mode
    device = None
    for name in ("w13_weight", "w2_weight", "weight"):
        candidate = getattr(layer, name, None)
        if isinstance(candidate, torch.Tensor) and candidate.is_cuda:
            device = candidate.device
            break
    if explicit == "auto" and device is None and not torch.cuda.is_available():
        # CPU/meta construction cannot inspect an architecture.  Keep the
        # Hopper metadata for the SM90 role (explicit mode, else
        # CHORD_SM90_DECODE, else the prefill default); callers targeting
        # Blackwell should pass profile="blackwell_decode_ep8" before loading.
        role = mode or indexed_mode_from_env() or "prefill"
        explicit = f"h200_{role}_ep8"
    profile = select_indexed_profile(explicit, mode=mode, device=device)
    try:
        setattr(layer, "_w4a16_profile", profile)
    except Exception:
        pass
    return profile


def _host_prepared_weight(
    layer: object, meta: IndexedLayerMeta
) -> PreparedWeight:
    """Read a transformed sublayer straight off a host framework's module."""
    packed = _parameter_tensor(layer, meta.weight_name)
    scale = _parameter_tensor(layer, meta.weight_scale_name)
    expected_packed = (meta.num_experts, meta.shape_k // 16, meta.shape_n * 2)
    expected_scale = (meta.num_experts, meta.shape_k // 32, meta.shape_n)
    if tuple(packed.shape) != expected_packed or tuple(scale.shape) != expected_scale:
        raise RuntimeError(
            f"{meta.sublayer_name or 'weight'} is not transformed to the indexed layout; "
            "call IndexedW4A16Method.transform_humming_layer first"
        )
    return PreparedWeight(
        packed=packed,
        scale=scale,
        layout=meta.layout,
        n=meta.shape_n,
        k=meta.shape_k,
        num_experts=meta.num_experts,
    )


def _config_with_block_m(
    config: IndexedKernelConfig,
    block_m: int,
    *,
    layout: WeightLayout,
) -> IndexedKernelConfig:
    """Keep a projection's tile choice while matching the shared route blocks."""
    if block_m == config.block_m:
        return config
    if (
        isinstance(block_m, bool)
        or not isinstance(block_m, int)
        or block_m <= 0
        or block_m % 8
    ):
        raise ValueError(
            f"routing block_m must be a positive multiple of 8, got {block_m!r}"
        )
    swap_ab = config.swap_ab
    if layout == "mma" and block_m % 16 == 8:
        # MMA's non-swap path cannot issue an m8n8k8 BF16 instruction.  The
        # swap layout accepts these small token tiles and is compatible with
        # the same packed weight buffer.
        swap_ab = True
    return dataclasses.replace(
        config,
        block_shape=(block_m, config.block_n, config.block_k),
        warp_shape=(block_m, config.warp_shape[1], config.warp_shape[2]),
        swap_ab=swap_ab,
    )


def _transform_host_weight(layer: object, meta: IndexedLayerMeta) -> None:
    weight = _parameter_tensor(layer, meta.weight_name)
    scale = _parameter_tensor(layer, meta.weight_scale_name)
    if not weight.is_cuda:
        raise RuntimeError(
            "indexed W4A16 weight transformation requires CUDA; move the layer "
            "to its serving device before process_weights_after_loading"
        )
    if tuple(weight.shape) == (meta.num_experts, meta.shape_k // 16, meta.shape_n * 2):
        # Already transformed by a previous loading hook.
        return
    if tuple(weight.shape) == (meta.num_experts, meta.shape_n, meta.shape_k // 8):
        source_is_packed = True
    elif tuple(weight.shape) != (meta.num_experts, meta.shape_n, meta.shape_k):
        raise ValueError(
            f"{meta.weight_name} must be [E,N,K/8] packed or [E,N,K] unpacked, "
            f"got {tuple(weight.shape)}"
        )
    else:
        source_is_packed = False
    if scale.dtype != torch.bfloat16:
        raise TypeError(f"{meta.weight_scale_name} must be torch.bfloat16")
    prepared = pack_w4a16(
        weight.contiguous(),
        scale.contiguous(),
        layout=meta.layout,
        packed=source_is_packed,
    )
    _replace_parameter(layer, meta.weight_name, prepared.packed)
    _replace_parameter(layer, meta.weight_scale_name, prepared.scale)


class IndexedW4A16Method:
    """Tiny method object matching the common framework delegation pattern.

    Tuning rows come from the indexed profile resolver, and unsupported schema
    features are rejected before a weight is transformed.
    """

    @classmethod
    def may_set_param(
        cls, layer: torch.nn.Module, name: str, tensor: torch.Tensor | None
    ) -> None:
        """Install a non-trainable tensor using the upstream helper convention."""
        del cls
        if tensor is None:
            return
        _replace_parameter(layer, name, tensor)

    @staticmethod
    def may_quant_input(
        layer: object | torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        **kwargs,
    ):
        """Pass BF16 inputs through using both Humming call conventions.

        Upstream Humming calls this as ``(layer, inputs, input_scale=...)``;
        the small adapter also accepts ``(inputs)`` or ``inputs=...``.  There is
        no input quantization in this BF16-only extraction.  An explicitly
        supplied scale is rejected instead of being silently ignored.
        """
        del quanted_input, kwargs
        if inputs is None and isinstance(layer, torch.Tensor):
            inputs = layer
        if inputs is None:
            raise TypeError("inputs must be provided to may_quant_input")
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        return inputs, None

    @staticmethod
    def may_hadamard_quant_input(
        layer: object | torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        hadamard_block_size: int | None = None,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        m_major_scale: bool = False,
        sublayer_name: str = "",
        **kwargs,
    ):
        """BF16 passthrough for the companion upstream helper.

        The indexed extraction has no input quantization or Hadamard rotation;
        accepting this call shape lets a framework share its Humming dispatch
        code without silently allocating an unused temporary.
        """
        del layer, quanted_input, m_major_scale, sublayer_name, kwargs
        if hadamard_block_size not in (None, 0, 1):
            raise ValueError(
                "indexed W4A16 accepts BF16 input only; Hadamard quantization is unsupported"
            )
        if inputs is None:
            raise TypeError("inputs must be provided to may_hadamard_quant_input")
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        return inputs, None

    @classmethod
    def get_default_tuning_configs(
        cls,
        layer,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        use_m_major_input_scale: bool = False,
        gemm_type: object = "indexed",
        sublayer_name: str = "",
        **kwargs,
    ):
        del use_m_major_input_scale
        if use_f16_accum or use_batch_invariant:
            raise ValueError(
                "indexed W4A16 tuning rows require use_f16_accum=False and "
                "use_batch_invariant=False"
            )
        gemm_type_value = getattr(gemm_type, "value", gemm_type)
        if str(gemm_type_value).lower() != "indexed":
            raise ValueError(
                "IndexedW4A16Method only provides tuning rows for gemm_type='indexed'"
            )
        if not hasattr(layer, "humming_metas"):
            layer.humming_metas = {}
        meta = layer.humming_metas.get(sublayer_name)
        if not isinstance(meta, IndexedLayerMeta):
            meta_kwargs = dict(kwargs)
            meta = cls.prepare_layer_meta(
                layer, sublayer_name=sublayer_name, **meta_kwargs
            )
        return _indexed_tuning_rows(meta)

    @classmethod
    def prepare_layer_meta(
        cls,
        layer,
        shape_n: int | None = None,
        shape_k: int | None = None,
        weight_schema: object | None = None,
        input_schema: object | None = None,
        num_experts: int | None = None,
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        has_bias: bool = False,
        torch_dtype: torch.dtype | None = None,
        sublayer_name: str = "",
        **kwargs,
    ):
        if not isinstance(sublayer_name, str):
            raise TypeError(
                f"sublayer_name must be a string, got {type(sublayer_name).__name__}"
            )
        kwargs = dict(kwargs)
        for name, requested in (
            ("shape_n", shape_n),
            ("shape_k", shape_k),
            ("num_experts", num_experts),
        ):
            current = getattr(layer, name, None)
            if requested is not None and current is not None and requested != current:
                raise ValueError(
                    f"{name}={requested} does not match layer.{name}={current}"
                )
        pad_n_multiple = pad_n_to_multiple
        pad_k_multiple = pad_k_to_multiple
        if (
            isinstance(pad_n_multiple, bool)
            or not isinstance(pad_n_multiple, int)
            or pad_n_multiple <= 0
            or isinstance(pad_k_multiple, bool)
            or not isinstance(pad_k_multiple, int)
            or pad_k_multiple <= 0
        ):
            raise ValueError("padding multiples must be positive integers")
        shape_n = shape_n if shape_n is not None else getattr(layer, "shape_n", None)
        shape_k = shape_k if shape_k is not None else getattr(layer, "shape_k", None)
        if shape_n is not None and pad_n_multiple and shape_n % pad_n_multiple:
            raise ValueError("indexed W4A16 metadata requires shape_n without padding")
        if shape_k is not None and pad_k_multiple and shape_k % pad_k_multiple:
            raise ValueError("indexed W4A16 metadata requires shape_k without padding")
        if not hasattr(layer, "humming_metas"):
            layer.humming_metas = {}
        if has_bias or kwargs.get("has_zero_point", False):
            raise ValueError(
                "indexed W4A16 supports neither bias nor explicit zero point"
            )
        if torch_dtype is None:
            torch_dtype = getattr(layer, "param_dtype", None)
        if torch_dtype is not None and torch_dtype != torch.bfloat16:
            raise ValueError(
                f"indexed W4A16 requires torch.bfloat16 parameters, got {torch_dtype}"
            )
        _validate_indexed_schema(weight_schema, input_schema)
        profile = _host_profile(layer, kwargs)
        if hasattr(layer, "_set_humming_meta"):
            layer_profile = getattr(layer, "indexed_profile")
            if layer_profile.name != profile.name:
                raise ValueError(
                    f"profile {profile.name!r} does not match the layer's "
                    f"profile {layer_profile.name!r}"
                )
            return layer._set_humming_meta(sublayer_name)
        num_experts = (
            num_experts
            if num_experts is not None
            else getattr(layer, "num_experts", None)
        )
        if shape_n is None or shape_k is None or num_experts is None:
            raise TypeError(
                "shape_n, shape_k, and num_experts are required for layer metadata"
            )
        meta = IndexedLayerMeta(
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=num_experts,
            profile=profile,
            sublayer_name=sublayer_name,
            pad_shape_n=0,
            pad_shape_k=0,
        )
        layer.humming_metas[sublayer_name] = meta
        return meta

    @classmethod
    def transform_humming_layer(
        cls,
        layer,
        sublayer_name: str = "",
        already_padded: bool = False,
        **kwargs,
    ):
        del already_padded
        if "sublayer_name" in kwargs:
            requested = kwargs.pop("sublayer_name")
            if sublayer_name and requested != sublayer_name:
                raise ValueError("duplicate sublayer_name arguments disagree")
            sublayer_name = requested
        if not isinstance(sublayer_name, str):
            raise TypeError(
                f"sublayer_name must be a string, got {type(sublayer_name).__name__}"
            )
        if isinstance(layer, IndexedW4A16Layer):
            return layer.transform()
        metas = getattr(layer, "humming_metas", None)
        if not isinstance(metas, dict) or not isinstance(
            metas.get(sublayer_name), IndexedLayerMeta
        ):
            cls.prepare_layer_meta(layer, sublayer_name=sublayer_name, **kwargs)
        meta = layer.humming_metas[sublayer_name]
        _transform_host_weight(layer, meta)
        return layer

    @classmethod
    def forward_layer(
        cls,
        layer: IndexedW4A16Layer,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        m_indices: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
        compute_config: object | None = None,
        tuning_config: object | None = None,
        sublayer_name: str = "",
        hadamard_block_size: int | None = None,
        validate_routing: bool = False,
        block_m: int | None = None,
        **kwargs,
    ):
        del kwargs
        if hadamard_block_size not in (None, 0, 1):
            raise ValueError(
                "indexed W4A16 accepts BF16 input only; Hadamard quantization is unsupported"
            )
        if compute_config is not None:
            _validate_indexed_compute_config(compute_config)
        # Preserve the small delegation contract for framework shims that only
        # implement ``forward``.  A real vLLM RoutedExperts object takes the
        # named-parameter path below after prepare_layer_meta has installed its
        # indexed metadata.
        if (
            not isinstance(layer, IndexedW4A16Layer)
            and not isinstance(getattr(layer, "humming_metas", None), dict)
            and hasattr(layer, "forward")
        ):
            return layer.forward(
                inputs,
                sorted_ids,
                expert_ids,
                num_tokens_padded,
                top_k,
                outputs=outputs,
                input_scale=input_scale,
                expert_layout=expert_layout,
                m_indices=m_indices,
                valid_shape_m=valid_shape_m,
                validate_routing=validate_routing,
                block_m=block_m,
            )
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        if expert_layout is not None or m_indices is not None:
            raise ValueError("indexed W4A16 uses sorted_ids/expert_ids routing only")
        if isinstance(layer, IndexedW4A16Layer):
            return layer.forward(
                inputs,
                sorted_ids,
                expert_ids,
                num_tokens_padded,
                top_k,
                outputs=outputs,
                compute_config=compute_config,
                tuning_config=tuning_config,
                sublayer_name=sublayer_name,
                valid_shape_m=valid_shape_m,
                validate_routing=validate_routing,
                block_m=block_m,
            )

        metas = getattr(layer, "humming_metas", None)
        if not isinstance(metas, dict) or not isinstance(
            metas.get(sublayer_name), IndexedLayerMeta
        ):
            raise TypeError(
                f"layer has no indexed metadata for sublayer {sublayer_name!r}; "
                "call prepare_layer_meta first"
            )
        meta = metas[sublayer_name]
        prepared = _host_prepared_weight(layer, meta)
        selected_m = valid_shape_m if valid_shape_m > 0 else inputs.size(0) * top_k
        kernel_config = _resolve_tuning_config(
            meta,
            selected_m,
            tuning_config,
            default=meta.kernel_config(selected_m),
        )

        if block_m is not None and block_m != kernel_config.block_m:
            # Same contract as IndexedW4A16Layer.forward: an explicit routing
            # block-M (e.g. the shared w13 table's) overrides the row's M tile
            # while keeping the projection's N/K tile.
            kernel_config = _config_with_block_m(
                kernel_config, block_m, layout=meta.layout
            )
        return w4a16_indexed(
            inputs,
            prepared,
            sorted_ids,
            expert_ids,
            num_tokens_padded,
            top_k,
            outputs=outputs,
            config=kernel_config,
            layout=meta.layout,
            swap_ab=kernel_config.swap_ab,
            valid_shape_m=valid_shape_m,
            validate_routing=validate_routing,
        )


__all__ = [
    "BLACKWELL_DECODE_EP8",
    "H200_DECODE_EP8",
    "H200_PREFILL_EP8",
    "INDEXED_PROFILES",
    "IndexedKernelConfig",
    "IndexedLayerMeta",
    "IndexedLayerProfile",
    "IndexedMode",
    "IndexedW4A16Layer",
    "IndexedW4A16Method",
    "indexed_mode_from_env",
    "select_indexed_profile",
    "unpack_packed_uint4",
]
