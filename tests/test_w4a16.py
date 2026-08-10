#!/usr/bin/env python3
"""Performance and accuracy checks for the indexed W4A16 operator.

Shapes are declared in ``tests/generators.py`` as ``PERFORMANCE_CASES``.  This
module only iterates over whichever cases the current device supports and prints
one table row each, so adding or removing a shape is an edit to the case table
rather than to this file.

Run the file directly and it does everything in one pass::

    python tests/test_w4a16.py

Every case is checked against a plain-PyTorch W4A16 reference and then timed, so
each row carries both its accuracy (``cos_diff``) and its throughput.  Rows are
grouped into a block per profile and projection.

The same cases are also collected by pytest.  The GPU cases need ``-s`` for the
table to reach the terminal, and the production-sized ones additionally need
``--run-perf`` because they are marked ``perf`` and skipped by default::

    python -m pytest -m "not gpu" tests/test_w4a16.py
    python -m pytest -m gpu -s tests/test_w4a16.py
    python -m pytest --run-perf -m "gpu and perf" -s tests/test_w4a16.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from bench import BENCH_METHODS, DEFAULT_BENCH_METHOD, benchmark
from generators import (
    DECODE_TOKENS_PER_GPU,
    PERFORMANCE_CASES,
    PREFILL_SYSTEM_TOKENS,
    IndexedCase,
    align_routing_blocks,
    generate_indexed_case,
    generate_indexed_routing,
    generate_topk_ids,
    pack_checkpoint_uint4,
    reference_indexed,
    select_cases,
)

_SUPPORTED_COMPUTE_CAPABILITIES = {(9, 0), (10, 0), (10, 3)}
_COSINE_LIMIT = 5e-3

# Per-architecture roofline peaks.  The INT4 weight is dequantized to BF16
# before the MMA, so the compute ceiling is the dense BF16 tensor rate rather
# than an INT4 rate.  H200 SXM: 989 TFLOPS BF16, 4.8 TB/s HBM3e.  B200/B300:
# 2250 TFLOPS dense BF16; bandwidth follows the shipping memory clock
# (3996 MHz x 7680-bit = 7.67 TB/s) rather than the 8 TB/s spec-sheet figure.
_ROOFLINE_PEAKS: dict[tuple[int, int], tuple[float, float]] = {
    (9, 0): (989.0, 4800.0),
    (10, 0): (2250.0, 7700.0),
    (10, 3): (2250.0, 7700.0),
}


def _device_peaks(device: torch.device | None = None) -> tuple[float, float]:
    capability = torch.cuda.get_device_capability(device)
    peaks = _ROOFLINE_PEAKS.get(capability)
    if peaks is None:
        # An unlisted device still benchmarks; fall back to the H200 numbers
        # rather than refusing, since the ceilings only scale the util column.
        peaks = _ROOFLINE_PEAKS[(9, 0)]
    return peaks


def _roofline(
    flops: int, total_bytes: int, device: torch.device | None = None
) -> tuple[float, float]:
    """Return the (TFLOPS, GB/s) ceilings for this arithmetic intensity.

    The workload sits on one ray of slope ``AI = flops / bytes`` in the roofline
    plane, clipped by the two hardware peaks.  Both ceilings describe the same
    point, so the utilization percentage is identical in either unit; reporting
    both shows which wall is the binding one.
    """

    peak_tflops, peak_gbps = _device_peaks(device)
    if total_bytes <= 0:
        return peak_tflops, peak_gbps
    intensity = flops / total_bytes
    roof_tflops = min(peak_tflops, peak_gbps * intensity / 1e3)
    roof_gbps = min(peak_gbps, peak_tflops * 1e3 / intensity)
    return roof_tflops, roof_gbps


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured case, ready to be printed as a table row."""

    case: IndexedCase
    layout: str
    block_m: int
    num_active_experts: int
    seconds: float
    tflops: float
    gbps: float
    roof_tflops: float
    roof_gbps: float
    cosine_diff: float | None = None
    max_abs_diff: float | None = None

    @property
    def microseconds(self) -> float:
        return self.seconds * 1e6

    @property
    def compute_utilization(self) -> float:
        return 100.0 * self.tflops / self.roof_tflops if self.roof_tflops else 0.0


def _case_flops(case: IndexedCase) -> int:
    """Multiply-accumulate FLOPs for the routed rows this case computes."""

    return 2 * case.routed_m * case.n * case.k


def _case_traffic_bytes(case: IndexedCase, num_active_experts: int) -> int:
    """Estimate DRAM traffic for the activation, weight, scale and output.

    Only experts actually reached by routing contribute weight and scale bytes;
    with a random router and a small token count most experts are never read,
    so counting all of them would overstate achieved bandwidth.
    """

    weight_bytes = num_active_experts * case.n * (case.k // 2)
    scale_bytes = num_active_experts * case.n * (case.k // 32) * 2
    activation_bytes = case.input_rows * case.k * 2
    output_bytes = case.routed_m * case.n * 2
    return weight_bytes + scale_bytes + activation_bytes + output_bytes


def _error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    actual = actual.float()
    reference = reference.float()
    denominator = (actual.square() + reference.square()).sum()
    cosine_diff = 0.0
    if denominator.item() != 0:
        cosine_diff = float(1 - (2 * (actual * reference).sum() / denominator).item())
    max_abs_diff = float((actual - reference).abs().max().item())
    return cosine_diff, max_abs_diff


def _check_output(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    cosine_limit: float = _COSINE_LIMIT,
) -> tuple[float, float]:
    cosine_diff, max_abs_diff = _error_metrics(actual, reference)
    if not cosine_diff == cosine_diff or cosine_diff > cosine_limit:
        raise AssertionError(
            "indexed correctness check failed: relative cosine diff "
            f"{cosine_diff:.6g} exceeds {cosine_limit:.6g}"
        )
    return cosine_diff, max_abs_diff


def run_indexed_case(
    case: IndexedCase,
    device: torch.device,
    *,
    iterations: int,
    method: str = DEFAULT_BENCH_METHOD,
    check: bool = True,
    check_rows: int | None = None,
) -> BenchmarkResult:
    """Run one case, check a route prefix, and return measured throughput."""

    from chord_kernels import indexed

    tensors = generate_indexed_case(case, device)
    layout = tensors.profile.layout
    swap_ab = tensors.kernel_config.swap_ab

    def invoke() -> torch.Tensor:
        return indexed(
            tensors.activation,
            tensors.prepared_weight,
            tensors.sorted_ids,
            tensors.expert_ids,
            tensors.num_tokens_padded,
            # A down projection passes top_k=1 so the kernel's sorted-id
            # division becomes an identity map over already routed rows.
            case.kernel_top_k,
            outputs=tensors.output,
            config=tensors.kernel_config,
            layout=layout,
            swap_ab=swap_ab,
            valid_shape_m=case.routed_m,
            validate_routing=False,
        )

    output, seconds = benchmark(
        invoke, device=device, method=method, iterations=iterations
    )
    cosine_diff = max_abs_diff = None
    if check:
        reference, checked_rows = reference_indexed(
            tensors.activation,
            tensors.logical_weight,
            tensors.scale,
            tensors.topk_ids,
            case.top_k,
            projection=case.projection,
            max_rows=check_rows,
        )
        cosine_diff, max_abs_diff = _check_output(output[:checked_rows], reference)

    num_active_experts = tensors.num_active_experts
    flops = _case_flops(case)
    total_bytes = _case_traffic_bytes(case, num_active_experts)
    roof_tflops, roof_gbps = _roofline(flops, total_bytes, device)
    return BenchmarkResult(
        case=case,
        layout=layout,
        block_m=tensors.kernel_config.block_m,
        num_active_experts=num_active_experts,
        seconds=seconds,
        tflops=flops / seconds / 1e12,
        gbps=total_bytes / seconds / 1e9,
        roof_tflops=roof_tflops,
        roof_gbps=roof_gbps,
        cosine_diff=cosine_diff,
        max_abs_diff=max_abs_diff,
    )


# Columns whose value is constant inside a group are dropped from that group's
# table, because the group heading already states them.
_SHAPE_COLUMNS: tuple[tuple[str, int], ...] = (
    ("profile", 20),
    ("proj", 7),
    ("n", 5),
    ("k", 5),
)

_COLUMNS: tuple[tuple[str, int], ...] = (
    ("routing", 8),
    ("tokens", 6),
    ("routed_m", 8),
    ("act.E", 6),
    ("layout", 6),
    ("blk_m", 5),
    ("us", 9),
    ("TFLOPS", 11),
    ("GB/s", 13),
    ("util", 6),
    ("cos_diff", 9),
)


def _columns(shape_columns: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    prefix = tuple(
        column for column in _SHAPE_COLUMNS if column[0] in shape_columns
    )
    return (*prefix, *_COLUMNS)


def _print_group(title: str, shape_columns: tuple[str, ...]) -> None:
    """Start a block of rows with its own heading and column header."""

    columns = _columns(shape_columns)
    width = sum(size for _, size in columns) + len(columns) - 1
    header = " ".join(name.rjust(size) for name, size in columns)
    print(f"\n{title}", flush=True)
    print("=" * width, flush=True)
    print(header, flush=True)
    print("-" * width, flush=True)


def _print_row(result: BenchmarkResult, shape_columns: tuple[str, ...] = ()) -> None:
    case = result.case
    cosine = "-" if result.cosine_diff is None else f"{result.cosine_diff:.2e}"
    available = {
        "profile": case.profile,
        "proj": case.projection,
        "n": str(case.n),
        "k": str(case.k),
    }
    values = (
        *(available[name] for name, _ in _SHAPE_COLUMNS if name in shape_columns),
        case.distribution,
        str(case.token_count),
        str(case.routed_m),
        f"{result.num_active_experts}/{case.num_experts}",
        result.layout,
        str(result.block_m),
        f"{result.microseconds:.1f}",
        f"{result.tflops:.0f}/{result.roof_tflops:.0f}",
        f"{result.gbps:.0f}/{result.roof_gbps:.0f}",
        f"{result.compute_utilization:.1f}%",
        cosine,
    )
    print(
        " ".join(
            value.rjust(size)
            for value, (_, size) in zip(values, _columns(shape_columns))
        ),
        flush=True,
    )


def _group_title(case: IndexedCase) -> str:
    """Heading for the block of rows a case belongs to.

    One block per profile, holding both projections; ``PERFORMANCE_CASES`` orders
    gate/up before down inside each profile.
    """

    if case.is_prefill:
        tokens = (
            "tokens = num_tokens_total per chunk, across the whole 8-GPU system"
        )
    else:
        tokens = (
            "tokens = num_tokens per GPU per step = bs_per_gpu * (mtp + 1), DP8"
        )
    if case.is_tensor_parallel:
        # TP8 keeps every expert on every rank and slices moe_intermediate, so
        # the same token count produces 8x the local routed rows EP8 does.  It is
        # a single-instance (mix) deployment, so it carries no P/D role.
        shard = "TP8 mix (384 experts on each of 8 GPUs, moe_intermediate/8)"
    else:
        shard = "EP8 (384 experts / 8 GPUs)"
    return f"{case.profile} -- Kimi K2.5 {shard}, {tokens}"


def _run_case_table(
    device: torch.device,
    *,
    iterations: int,
    method: str = DEFAULT_BENCH_METHOD,
    check: bool = True,
    check_rows: int | None = None,
) -> list[BenchmarkResult]:
    """Run every case this device supports, printing one block per group."""

    capability = torch.cuda.get_device_capability(device)
    cases = select_cases(capability)
    if not cases:
        raise RuntimeError(
            f"no indexed W4A16 cases are declared for SM{capability[0]}{capability[1]}"
        )
    # Keep a shape column only where it actually varies inside its group; the
    # heading already names whatever is constant.
    groups = [_group_title(case) for case in cases]
    varying: dict[str, tuple[str, ...]] = {}
    for group in groups:
        members = [case for case, name in zip(cases, groups) if name == group]
        varying[group] = tuple(
            name
            for name, getter in (
                ("profile", lambda case: case.profile),
                ("proj", lambda case: case.projection),
                ("n", lambda case: case.n),
                ("k", lambda case: case.k),
            )
            if len({getter(case) for case in members}) > 1
        )

    results = []
    current_group: str | None = None
    for case, group in zip(cases, groups):
        if group != current_group:
            _print_group(group, varying[group])
            current_group = group
        result = run_indexed_case(
            case,
            device,
            iterations=iterations,
            method=method,
            check=check,
            check_rows=check_rows,
        )
        _print_row(result, varying[group])
        results.append(result)
    return results


@pytest.fixture(scope="module")
def supported_gpu() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("indexed W4A16 tests require CUDA")
    capability = torch.cuda.get_device_capability()
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(
            "indexed W4A16 tests require an SM90, SM100, or SM103 GPU; "
            f"got SM{capability[0]}{capability[1]}"
        )
    return torch.device("cuda")


@pytest.mark.gpu
def test_indexed_layer_correctness(supported_gpu: torch.device) -> None:
    """Drive the vLLM-facing layer adapter end to end on a checkpoint tensor."""

    from chord_kernels.operator import IndexedW4A16Layer

    capability = torch.cuda.get_device_capability(supported_gpu)
    profile = "h200_decode_ep8" if capability == (9, 0) else "blackwell_decode_ep8"
    case = IndexedCase(
        profile, m=8, n=128, k=128, num_experts=4, top_k=2, seed=20260810
    )
    tensors = generate_indexed_case(case, supported_gpu)
    layer = IndexedW4A16Layer(
        shape_n=case.n,
        shape_k=case.k,
        num_experts=case.num_experts,
        profile=profile,
    )
    layer.load_weight(
        pack_checkpoint_uint4(tensors.logical_weight).cpu(),
        tensors.scale.cpu(),
        packed=True,
    )
    layer.to(supported_gpu)
    layer.process_weights_after_loading()
    actual = layer(
        tensors.activation,
        tensors.sorted_ids,
        tensors.expert_ids,
        tensors.num_tokens_padded,
        case.kernel_top_k,
    )
    reference, rows = reference_indexed(
        tensors.activation,
        tensors.logical_weight,
        tensors.scale,
        tensors.topk_ids,
        case.top_k,
        projection=case.projection,
    )
    cosine_diff, _ = _check_output(actual[:rows], reference)
    assert cosine_diff <= _COSINE_LIMIT


@pytest.mark.perf
@pytest.mark.gpu
def test_indexed_w4a16_performance(supported_gpu: torch.device) -> None:
    """Run production-sized cases and print achieved throughput."""

    results = _run_case_table(
        supported_gpu,
        iterations=20,
        # A bounded prefix keeps the optional benchmark practical while still
        # checking the routing and layout path touched by the launch.
        check_rows=8,
    )
    for result in results:
        assert result.cosine_diff is not None
        assert result.cosine_diff <= _COSINE_LIMIT


@pytest.mark.parametrize("distribution", ["random", "balanced"])
@pytest.mark.parametrize(
    "num_tokens,top_k,num_experts,block_m",
    [(512, 8, 48, 16), (16, 8, 48, 8), (3, 2, 4, 16)],
)
def test_indexed_routing_contract(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    block_m: int,
    distribution: str,
) -> None:
    """Both distributions must satisfy the kernel's routing preconditions."""

    topk_ids, sorted_ids, expert_ids, count = generate_indexed_routing(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        block_m=block_m,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(20260811),
        distribution=distribution,
    )
    routed_m = num_tokens * top_k
    valid_ids = sorted_ids[sorted_ids < routed_m]
    assert topk_ids.shape == (num_tokens, top_k)
    # Every routed row must appear exactly once across all active blocks.
    assert torch.equal(
        torch.sort(valid_ids).values, torch.arange(routed_m, dtype=torch.int32)
    )
    assert sorted_ids.numel() % block_m == 0
    assert expert_ids.numel() == sorted_ids.numel() // block_m
    assert int(count.item()) == sorted_ids.numel()
    for block in sorted_ids.view(-1, block_m):
        padding = block >= routed_m
        if bool(padding.any()):
            first = int(torch.where(padding)[0][0])
            assert bool(padding[first:].all())


@pytest.mark.parametrize("distribution", ["random", "balanced"])
def test_topk_ids_select_distinct_experts(distribution: str) -> None:
    topk_ids = generate_topk_ids(
        num_tokens=64,
        top_k=8,
        num_experts=48,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(20260812),
        distribution=distribution,
    )
    assert topk_ids.shape == (64, 8)
    assert topk_ids.dtype == torch.int32
    assert int(topk_ids.min()) >= 0 and int(topk_ids.max()) < 48
    for row in topk_ids:
        assert torch.unique(row).numel() == row.numel()


def test_random_routing_is_more_skewed_than_balanced() -> None:
    """The default distribution must actually exercise uneven expert load."""

    counts = {}
    for distribution in ("random", "balanced"):
        topk_ids = generate_topk_ids(
            num_tokens=256,
            top_k=8,
            num_experts=48,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(20260813),
            distribution=distribution,
        )
        counts[distribution] = (
            topk_ids.reshape(-1).bincount(minlength=48).float().std().item()
        )
    assert counts["random"] > counts["balanced"]


def test_routing_rejects_top_k_above_num_experts() -> None:
    with pytest.raises(ValueError, match="top_k=8 cannot exceed num_experts=4"):
        generate_topk_ids(
            num_tokens=4,
            top_k=8,
            num_experts=4,
            device=torch.device("cpu"),
        )


def test_align_routing_blocks_skips_experts_without_routes() -> None:
    # Only experts 0 and 3 are routed, so exactly two blocks must be emitted.
    topk_ids = torch.tensor([[0, 3], [0, 3]], dtype=torch.int32)
    sorted_ids, expert_ids, count = align_routing_blocks(
        topk_ids, block_m=8, num_experts=4
    )
    assert torch.equal(expert_ids, torch.tensor([0, 3], dtype=torch.int32))
    assert sorted_ids.numel() == 16
    assert int(count.item()) == 16


def test_down_projection_case_shapes() -> None:
    """A down case consumes routed rows and keeps the kernel's top_k at 1."""

    gate_up = IndexedCase("h200_decode_ep8", m=16, n=4096, k=7168, num_experts=48, top_k=8, seed=1)
    down = IndexedCase(
        "h200_decode_ep8",
        m=16,
        n=7168,
        k=2048,
        num_experts=48,
        top_k=8,
        seed=1,
        projection="down",
    )
    assert (gate_up.input_rows, gate_up.routed_m, gate_up.kernel_top_k) == (16, 128, 8)
    assert (down.input_rows, down.routed_m, down.kernel_top_k) == (128, 128, 1)


def test_case_table_declares_both_projections() -> None:
    assert {case.projection for case in PERFORMANCE_CASES} == {"gate_up", "down"}


def test_prefill_routed_rows_match_system_tokens() -> None:
    """EP8 prefill: local routed rows equal the whole system's token count.

    A token's top_k=8 routes spread over all 384 experts and this GPU owns 48,
    so T * 8 * (48 / 384) = T rows land locally.
    """

    prefill = [
        case
        for case in PERFORMANCE_CASES
        if case.profile == "h200_prefill_ep8" and case.projection == "gate_up"
    ]
    assert [case.routed_m for case in prefill] == list(PREFILL_SYSTEM_TOKENS)


def test_tp8_routed_rows_are_eight_times_ep8() -> None:
    """TP8: every rank holds a slice of all 384 experts, so every route is local.

    At the same num_tokens_total that is 8x the EP8 row count, which is what pushes
    the table past EP8's largest bracket.
    """

    tp8 = [
        case
        for case in PERFORMANCE_CASES
        if case.profile == "h200_tp8" and case.projection == "gate_up"
    ]
    assert [case.token_count for case in tp8] == list(PREFILL_SYSTEM_TOKENS)
    assert [case.routed_m for case in tp8] == [
        tokens * 8 for tokens in PREFILL_SYSTEM_TOKENS
    ]
    # The largest case must exceed the EP8 maximum by the full shard factor.
    assert max(case.routed_m for case in tp8) == 131072


def test_tp8_shapes_slice_the_intermediate_dimension() -> None:
    """TP8 narrows gate/up's N and down's K by 8, keeping all 384 experts."""

    tp8 = [case for case in PERFORMANCE_CASES if case.profile == "h200_tp8"]
    assert tp8, "the TP8 profile declares no cases"
    gate_up = {(case.n, case.k) for case in tp8 if case.projection == "gate_up"}
    down = {(case.n, case.k) for case in tp8 if case.projection == "down"}
    # gate/up loses output width (2 * 2048/8), down loses K depth (2048/8);
    # hidden_size 7168 is untouched in both.
    assert gate_up == {(512, 7168)}
    assert down == {(7168, 256)}
    assert {case.num_experts for case in tp8} == {384}
    assert {case.top_k for case in tp8} == {8}
    assert all(case.is_tensor_parallel for case in tp8)


def test_tp8_and_ep8_sweep_the_same_token_counts() -> None:
    """Both 8-GPU shardings are quoted at the same serving load."""

    def tokens(profile: str) -> list[int]:
        return [
            case.token_count
            for case in PERFORMANCE_CASES
            if case.profile == profile and case.projection == "gate_up"
        ]

    assert tokens("h200_tp8") == tokens("h200_prefill_ep8")


def test_decode_tokens_are_already_per_gpu() -> None:
    """EP8 decode runs with DP8, so the declared token counts are per GPU."""

    decode = [
        case
        for case in PERFORMANCE_CASES
        if case.profile == "h200_decode_ep8" and case.projection == "gate_up"
    ]
    assert [case.m for case in decode] == list(DECODE_TOKENS_PER_GPU)
    assert [case.routed_m for case in decode] == [
        tokens * 8 for tokens in DECODE_TOKENS_PER_GPU
    ]


@pytest.mark.parametrize(
    "routed_m,shape_k,expected",
    [
        # tok_e < 80 minimizes block count instead of modelling padding; the
        # sampled routing reproduces upstream's, so these match Humming exactly.
        (1024, 7168, 40),
        (2048, 7168, 64),
        # tok_e >= 80: one or two blocks under the 176 register ceiling.
        (4096, 7168, 96),
        (8192, 7168, 96),
        # Three blocks: deep-K gate/up takes the extra 176 window, short-K down
        # does not.
        (16384, 7168, 176),
        (16384, 2048, 128),
    ],
)
def test_prefill_block_m_follows_tokens_per_expert(
    routed_m: int, shape_k: int, expected: int
) -> None:
    """Prefill block-M must track tok_e, matching the upstream Humming rule."""

    from chord_kernels.operator.layer import _h200_prefill_block_m

    assert _h200_prefill_block_m(routed_m, 48, shape_k) == expected


@pytest.mark.parametrize(
    "routed_m,expected",
    [
        # tok_e < 80 falls back to the block-count argmin, same as EP8.
        (8192, 40),
        (16384, 72),
        (30719, 120),
        # tok_e >= 80: one block per expert while tok_e <= 128 ...
        (30720, 88),
        (32768, 96),
        (49152, 144),
        # ... then a flat 96 window to tok_e 190 ...
        (49153, 96),
        (65536, 96),
        (72960, 96),
        # ... then 128 for good.
        (72961, 128),
        (131072, 128),
    ],
)
def test_tp8_block_m_follows_tokens_per_expert(
    routed_m: int, expected: int
) -> None:
    """TP8 block-M uses the flatter TP-scale windows, not EP8's padding model.

    Both TP8 projections are narrow in whichever dimension fills the grid, so
    block count and occupancy dominate rather than per-expert M-padding: the
    96..144 band beats the taller tiles EP8 grows into.
    """

    from chord_kernels.operator.layer import _h200_tp8_block_m

    assert _h200_tp8_block_m(routed_m, 384) == expected


@pytest.mark.parametrize(
    "shape_k,routed_m,expected",
    [
        # down (K=256): only 4 K-blocks to split, so one-pass until the workload
        # is imbalanced enough that load balancing repays the locks.
        (256, 8192, False),
        (256, 65536, False),
        (256, 65537, True),
        (256, 131072, True),
        # gate/up (K=7168): the mirror image -- split until the M*N tiles fill
        # the grid on their own at 65536.
        (7168, 8192, True),
        (7168, 65535, True),
        (7168, 65536, False),
        (7168, 131072, False),
    ],
)
def test_tp8_stream_k_gate(
    shape_k: int, routed_m: int, expected: bool
) -> None:
    """The two TP8 projections cross over in opposite directions at 65536."""

    from chord_kernels.operator.layer import _h200_tp8_use_stream_k

    assert _h200_tp8_use_stream_k(routed_m, shape_k) is expected


@pytest.mark.parametrize(
    "shape_n,shape_k,routed_m,block_shape,num_ctas_per_sm,use_stream_k",
    [
        # gate/up: block-K compensates for the narrow N while block-M is short,
        # then the wide 256 tile takes over past block-M 64.
        (512, 7168, 8192, (40, 128, 128), 1, True),
        (512, 7168, 16384, (72, 256, 64), 1, True),
        (512, 7168, 32768, (96, 256, 64), 1, True),
        (512, 7168, 65536, (96, 256, 64), 1, False),
        (512, 7168, 131072, (128, 256, 64), 1, False),
        # down: block-N halved to 128 unlocks 2 CTAs/SM at every size, and the
        # short K keeps block-K one notch shallower than gate/up's.
        (7168, 256, 8192, (40, 128, 64), 2, False),
        (7168, 256, 16384, (72, 128, 64), 2, False),
        (7168, 256, 32768, (96, 128, 64), 2, False),
        (7168, 256, 65536, (96, 128, 64), 2, False),
        (7168, 256, 131072, (128, 128, 64), 2, True),
    ],
)
def test_tp8_configs_at_benchmark_points(
    shape_n: int,
    shape_k: int,
    routed_m: int,
    block_shape: tuple[int, int, int],
    num_ctas_per_sm: int,
    use_stream_k: bool,
) -> None:
    """Pin the schedule behind the TP8 table in docs/performance.md.

    ``num_tokens_total`` 1024..16384 (routed_m 8192..131072) for both projections,
    so a change here is a change to published numbers.
    """

    from chord_kernels.operator import IndexedLayerMeta
    from chord_kernels.operator.layer import _PROFILES

    meta = IndexedLayerMeta(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=384,
        profile=_PROFILES["h200_tp8"],
    )
    config = meta.kernel_config(routed_m)
    assert config.block_shape == block_shape
    assert config.warp_shape == (block_shape[0], 32, 64)
    assert config.num_ctas_per_sm == num_ctas_per_sm
    assert config.use_stream_k is use_stream_k
    # TP8 is a WGMMA profile, so the swap-AB decode schedule never applies.
    assert config.swap_ab is False


def test_tp8_down_never_drops_to_one_cta() -> None:
    """Block-N 128 is what buys down its 2 CTAs/SM, so it must hold everywhere.

    At block-N 256 the accumulator plus B-smem pin occupancy at 1 CTA/SM, which is
    the regression this checks for across the sweep, not just the table's points.
    """

    from chord_kernels.operator import IndexedLayerMeta
    from chord_kernels.operator.layer import _PROFILES

    meta = IndexedLayerMeta(
        shape_n=7168, shape_k=256, num_experts=384,
        profile=_PROFILES["h200_tp8"],
    )
    for routed_m in (1, 512, 8192, 32768, 65536, 131072, 262144):
        config = meta.kernel_config(routed_m)
        assert config.block_n == 128, routed_m
        assert config.num_ctas_per_sm == 2, routed_m


def test_prefill_block_m_is_deterministic() -> None:
    """The block-count argmin samples a routing, so it must use a fixed seed."""

    from chord_kernels.operator.layer import _h200_prefill_block_m

    first = [_h200_prefill_block_m(m, 48, 7168) for m in (512, 1024, 2048)]
    second = [_h200_prefill_block_m(m, 48, 7168) for m in (512, 1024, 2048)]
    assert first == second


@pytest.mark.parametrize(
    "shape_n,shape_k,routed_m,expected",
    [
        # gate/up (deep K): stream-K helps at every prefill size.
        (4096, 7168, 1024, True),
        (4096, 7168, 16384, True),
        # down (mid K, 512 < K < 4096): stream-K only until routed_m hits the
        # crossover, then the K-split is pure overhead.
        (7168, 2048, 4096, True),
        (7168, 2048, 5120, False),
        (7168, 2048, 16384, False),
    ],
)
def test_prefill_stream_k_gate(
    shape_n: int, shape_k: int, routed_m: int, expected: bool
) -> None:
    """Prefill enables stream-K everywhere except large mid-K down shapes."""

    from chord_kernels.operator.layer import _h200_prefill_use_stream_k

    assert _h200_prefill_use_stream_k(routed_m, shape_n, shape_k) == expected


def test_decode_stream_k_policy() -> None:
    """H200 decode is one-pass; Blackwell decode splits K per projection.

    The deep-K gate/up projection keeps stream-K at every decode size (the K
    loop is 112 blocks, so the tail balance repays the locks), while the short-K
    down projection stays one-pass through the swap-AB window and only enables
    the split on the large non-swap tiles.
    """

    from chord_kernels.operator import IndexedLayerMeta
    from chord_kernels.operator.layer import _PROFILES

    for shape_n, shape_k in ((4096, 7168), (7168, 2048)):
        meta = IndexedLayerMeta(
            shape_n=shape_n, shape_k=shape_k, num_experts=48,
            profile=_PROFILES["h200_decode_ep8"],
        )
        for routed_m in (160, 400, 8192):
            assert not meta.kernel_config(routed_m).use_stream_k

    blackwell = _PROFILES["blackwell_decode_ep8"]
    gate_up = IndexedLayerMeta(
        shape_n=4096, shape_k=7168, num_experts=48, profile=blackwell
    )
    down = IndexedLayerMeta(
        shape_n=7168, shape_k=2048, num_experts=48, profile=blackwell
    )
    for routed_m in (160, 400, 1200, 8192):
        assert gate_up.kernel_config(routed_m).use_stream_k
    for routed_m in (160, 400, 848):
        assert not down.kernel_config(routed_m).use_stream_k
    for routed_m in (1200, 8192):
        assert down.kernel_config(routed_m).use_stream_k


def test_prefill_forces_two_ctas_in_block_m_window() -> None:
    """Prefill runs 2 CTAs/SM for the 40..80 block-M window at block_n=256.

    Doubling resident CTAs there hides the cp.async + dequant latency and is
    ~11% faster on the down 1024/2048 tiles. Outside the window (small block_m at
    block_n=128, or block_m > 80) it stays at 1 CTA/SM.
    """

    from chord_kernels.operator import IndexedLayerMeta
    from chord_kernels.operator.layer import _PROFILES

    profile = _PROFILES["h200_prefill_ep8"]
    for shape_n, shape_k in ((4096, 7168), (7168, 2048)):
        meta = IndexedLayerMeta(
            shape_n=shape_n, shape_k=shape_k, num_experts=48, profile=profile
        )
        for routed_m in (512, 1024, 2048, 4096, 8192, 16384, 65536):
            config = meta.kernel_config(routed_m)
            in_window = config.block_n == 256 and 40 <= config.block_m <= 80
            expected = 2 if in_window else 1
            assert config.num_ctas_per_sm == expected, (
                f"n={shape_n} k={shape_k} routed_m={routed_m} "
                f"block_m={config.block_m} block_n={config.block_n} -> "
                f"{config.num_ctas_per_sm} CTAs/SM, expected {expected}"
            )


def test_traffic_bytes_counts_only_active_experts() -> None:
    case = PERFORMANCE_CASES[0]
    assert _case_traffic_bytes(case, 48) > _case_traffic_bytes(case, 8)


def test_tuning_rows_cover_all_routed_m() -> None:
    """`get_default_tuning_configs` rows form contiguous brackets over routed-M.

    Frameworks index the table with `min_m < routed_m <= max_m`, so the rows
    must tile [0, 2^30] without gaps and carry the resolver's schedule.
    """

    from chord_kernels.operator import IndexedLayerMeta
    from chord_kernels.operator.layer import _PROFILES, _indexed_tuning_rows

    # Each profile is swept at one of its own tuned shapes: TP8 narrows gate/up's
    # N to 512 and holds all 384 experts, so the EP8 shape would only exercise
    # the generic fallback and never reach the TP8 rows.
    sweeps = (
        ("h200_prefill_ep8", 4096, 7168, 48),
        ("h200_tp8", 512, 7168, 384),
        ("h200_decode_ep8", 4096, 7168, 48),
        ("blackwell_decode_ep8", 4096, 7168, 48),
    )
    for profile_name, shape_n, shape_k, num_experts in sweeps:
        meta = IndexedLayerMeta(
            shape_n=shape_n, shape_k=shape_k, num_experts=num_experts,
            profile=_PROFILES[profile_name],
        )
        rows = _indexed_tuning_rows(meta)
        assert rows[0][0] == 0
        assert rows[-1][1] == 1 << 30
        for previous, current in zip(rows, rows[1:]):
            assert previous[1] == current[0]
        for _, _, config in rows:
            assert config["block_m"] == config["block_shape"][0]
            assert config["layout"] == meta.layout


def test_sm90_decode_env_selects_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHORD_SM90_DECODE off means prefill.

    The SM90 role bit only fills in when ``profile="auto"`` carries no explicit
    mode; explicit arguments always win, and any value other than 0/1 is
    rejected rather than silently mapped to a role.
    """

    from chord_kernels.operator import layer as layer_module

    monkeypatch.delenv("CHORD_SM90_DECODE", raising=False)
    assert layer_module.indexed_mode_from_env() is None
    layer = layer_module.IndexedW4A16Layer(shape_n=128, shape_k=64, num_experts=1)
    assert layer.indexed_profile.name == "h200_prefill_ep8"

    monkeypatch.setenv("CHORD_SM90_DECODE", "1")
    layer = layer_module.IndexedW4A16Layer(shape_n=128, shape_k=64, num_experts=1)
    assert layer.indexed_profile.name == "h200_decode_ep8"

    # Explicit arguments beat the environment.
    layer = layer_module.IndexedW4A16Layer(
        shape_n=128, shape_k=64, num_experts=1, mode="prefill"
    )
    assert layer.indexed_profile.name == "h200_prefill_ep8"

    monkeypatch.setenv("CHORD_SM90_DECODE", "yes")
    with pytest.raises(ValueError, match="CHORD_SM90_DECODE"):
        layer_module.indexed_mode_from_env()


def test_layer_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import chord_kernels.operator.layer as layer_module

    layer = layer_module.IndexedW4A16Layer(
        shape_n=128,
        shape_k=64,
        num_experts=1,
        profile="h200_decode_ep8",
    )
    checkpoint_weight = torch.zeros((1, 128, 8), dtype=torch.int32)
    checkpoint_scale = torch.ones((1, 128, 2), dtype=torch.bfloat16)
    layer.load_weight(checkpoint_weight, checkpoint_scale, packed=True)
    assert tuple(layer.weight.shape) == (1, 128, 8)
    assert layer._prepared_weight is None

    meta = layer_module.IndexedW4A16Method.prepare_layer_meta(
        layer,
        shape_n=128,
        shape_k=64,
        num_experts=1,
        sublayer_name="w13",
    )
    assert (meta.weight_name, meta.weight_scale_name) == (
        "w13_weight",
        "w13_weight_scale",
    )
    rows = layer_module.IndexedW4A16Method.get_default_tuning_configs(layer)
    assert rows[0][2]["block_shape"] == (8, 128, 64)

    expected = object()
    calls = {}

    def fake_indexed(*args, **kwargs):
        calls.update(args=args, kwargs=kwargs)
        return expected

    # The kernel call is dispatched flatly; the dispatch module owns the single
    # ``w4a16_indexed`` call site.
    from chord_kernels.operator import dispatch as dispatch_mod

    monkeypatch.setattr(dispatch_mod, "w4a16_indexed", fake_indexed)
    layer._prepared_weight = object()
    inputs = torch.zeros((1, 64), dtype=torch.bfloat16)
    sorted_ids = torch.arange(8, dtype=torch.int32)
    expert_ids = torch.zeros(1, dtype=torch.int32)
    count = torch.tensor(8, dtype=torch.int32)
    assert layer(inputs, sorted_ids, expert_ids, count, 1) is expected
    assert calls["args"][1] is layer._prepared_weight
    assert calls["kwargs"]["validate_routing"] is False


def test_checkpoint_uint4_layout() -> None:
    from chord_kernels.operator import unpack_packed_uint4

    logical = torch.arange(16, dtype=torch.int32).repeat(1, 128, 4)
    packed = pack_checkpoint_uint4(logical)
    unpacked = unpack_packed_uint4(packed, shape_k=64)
    assert torch.equal(unpacked, logical)


def test_sm103_jit_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from chord_kernels.operator.jit.runtime import KernelRuntime

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(major=10, minor=3),
    )
    runtime = object.__new__(KernelRuntime)
    runtime.init_sm_version()
    assert (runtime.sm_version, runtime.sm_version_str) == (103, "103a")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=BENCH_METHODS,
        default=DEFAULT_BENCH_METHOD,
        help=(
            "timing method (default: triton, matching the upstream Humming "
            "benchmark; kineto reads profiler kernel time; events is the "
            "CUDA-event fallback)"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="iterations per case for the kineto/events methods (default: 20)",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def test_cli_defaults_are_minimal() -> None:
    default = _parse_args([])
    assert default.iterations is None
    assert default.device == "cuda"
    assert default.method == "triton"
    assert _parse_args(["--iterations", "50"]).iterations == 50
    assert _parse_args(["--method", "kineto"]).method == "kineto"


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the indexed W4A16 benchmark requires a CUDA GPU")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"--device must select CUDA, got {device}")

    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    print(
        f"{name} (SM{major}{minor}) -- indexed W4A16, "
        f"BF16 activation x INT4 weight, group-32 scale -- timing: {args.method}"
    )

    # Every case is checked against the reference and then timed, so the
    # cos_diff column carries the accuracy result for the same launch that
    # produced the throughput numbers.
    _run_case_table(
        device,
        iterations=args.iterations if args.iterations is not None else 20,
        method=args.method,
        check=True,
        check_rows=8,
    )
    print(flush=True)


if __name__ == "__main__":
    main()
