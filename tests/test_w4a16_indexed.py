#!/usr/bin/env python3
"""Performance and accuracy checks for the indexed W4A16 operator.

Shapes are declared in ``tests/generators.py`` as ``PERFORMANCE_CASES``.  This
module only iterates over whichever cases the current device supports and prints
one table row each, so adding or removing a shape is an edit to the case table
rather than to this file.

Run the file directly and it does everything in one pass::

    python tests/test_w4a16_indexed.py

Every case is checked against a plain-PyTorch W4A16 reference and then timed, so
each row carries both its accuracy (``cos_diff``) and its throughput.  Rows are
grouped into a block per profile and projection.

The same cases are also collected by pytest.  The GPU cases need ``-s`` for the
table to reach the terminal, and the production-sized ones additionally need
``--run-perf`` because they are marked ``perf`` and skipped by default::

    python -m pytest -m "not gpu" tests/test_w4a16_indexed.py
    python -m pytest -m gpu -s tests/test_w4a16_indexed.py
    python -m pytest --run-perf -m "gpu and perf" -s tests/test_w4a16_indexed.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from bench import BENCH_METHODS, DEFAULT_BENCH_METHOD, benchmark
from roofline import (
    ROOFLINE_PEAKS as _ROOFLINE_PEAKS,
    device_peaks as _device_peaks,
    error_metrics as _error_metrics,
    format_throughput,
    print_group,
    print_row,
    roofline as _roofline,
    throughput_columns,
    traffic_bytes,
)
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
    """DRAM traffic for this case, counting only the experts routing reached."""
    return traffic_bytes(
        num_active_experts, case.n, case.k, case.input_rows, case.routed_m
    )


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
    """Run one case through the public ``indexed``, check a route prefix, and
    return measured throughput."""

    from chord_kernels import indexed

    tensors = generate_indexed_case(case, device)
    layout = "wgmma" if tensors.kernel_config.warp_shape[1] == 32 else "mma"

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
            locks=tensors.locks,
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
    # Base widths; _columns() widens whichever of the two carries the
    # utilization percentage for the block being printed.
    ("TFLOPS", 11),
    ("GB/s", 13),
    ("cos_diff", 9),
)


def _columns(
    shape_columns: tuple[str, ...], *, is_prefill: bool = True
) -> tuple[tuple[str, int], ...]:
    """Column table for one block.

    Shape columns are kept only where they vary inside the block; the
    utilization percentage rides in the column of the binding wall (prefill is
    compute bound, decode is bandwidth bound).
    """
    prefix = tuple(
        column for column in _SHAPE_COLUMNS if column[0] in shape_columns
    )
    return (*prefix, *throughput_columns(_COLUMNS, compute_bound=is_prefill))


def _print_row(result: BenchmarkResult, shape_columns: tuple[str, ...] = ()) -> None:
    case = result.case
    cosine = "-" if result.cosine_diff is None else f"{result.cosine_diff:.2e}"
    available = {
        "profile": case.scenario,
        "proj": case.projection,
        "n": str(case.n),
        "k": str(case.k),
    }
    tflops, gbps = format_throughput(
        result.tflops,
        result.roof_tflops,
        result.gbps,
        result.roof_gbps,
        compute_bound=case.is_prefill,
    )
    values = (
        *(available[name] for name, _ in _SHAPE_COLUMNS if name in shape_columns),
        case.distribution,
        str(case.token_count),
        str(case.routed_m),
        f"{result.num_active_experts}/{case.num_experts}",
        result.layout,
        str(result.block_m),
        f"{result.microseconds:.1f}",
        tflops,
        gbps,
        cosine,
    )
    print_row(values, _columns(shape_columns, is_prefill=case.is_prefill))


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
    return f"{case.scenario} -- Kimi K2.5 {shard}, {tokens}"


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
                ("profile", lambda case: case.scenario),
                ("proj", lambda case: case.projection),
                ("n", lambda case: case.n),
                ("k", lambda case: case.k),
            )
            if len({getter(case) for case in members}) > 1
        )
        # A block's header commits to one binding wall (TFLOPS for prefill,
        # GB/s for decode), so every case in it must share the P/D role.  Groups
        # are keyed by profile and a profile is one role, so this holds by
        # construction; the check keeps a future case table honest.
        assert len({case.is_prefill for case in members}) == 1, (
            f"group {group!r} mixes prefill and decode cases"
        )

    results = []
    current_group: str | None = None
    for case, group in zip(cases, groups):
        if group != current_group:
            print_group(
                group, _columns(varying[group], is_prefill=case.is_prefill)
            )
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
def test_indexed_checkpoint_weight_correctness(supported_gpu: torch.device) -> None:
    """Pack a checkpoint-form (eight nibbles per int32) weight and run the tuned
    decode config end to end against the BF16 dequant reference."""

    from chord_kernels.operator import w4a16_indexed as indexed
    from chord_kernels import pack_w4a16

    capability = torch.cuda.get_device_capability(supported_gpu)
    profile = "h200_decode_ep8" if capability == (9, 0) else "blackwell_decode_ep8"
    case = IndexedCase(
        profile, m=8, n=128, k=128, num_experts=4, top_k=2, seed=20260810
    )
    tensors = generate_indexed_case(case, supported_gpu)
    weight = pack_w4a16(
        pack_checkpoint_uint4(tensors.logical_weight), tensors.scale, packed=True
    )
    assert torch.equal(weight.packed, tensors.prepared_weight.packed)
    actual = indexed(
        tensors.activation,
        weight,
        tensors.sorted_ids,
        tensors.expert_ids,
        tensors.num_tokens_padded,
        case.kernel_top_k,
        config=tensors.kernel_config,
        locks=tensors.locks,
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
        if case.scenario == "h200_prefill_ep8" and case.projection == "gate_up"
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
        if case.scenario == "h200_tp8" and case.projection == "gate_up"
    ]
    assert [case.token_count for case in tp8] == list(PREFILL_SYSTEM_TOKENS)
    assert [case.routed_m for case in tp8] == [
        tokens * 8 for tokens in PREFILL_SYSTEM_TOKENS
    ]
    # The largest case must exceed the EP8 maximum by the full shard factor.
    assert max(case.routed_m for case in tp8) == 131072


def test_tp8_shapes_slice_the_intermediate_dimension() -> None:
    """TP8 narrows gate/up's N and down's K by 8, keeping all 384 experts."""

    tp8 = [case for case in PERFORMANCE_CASES if case.scenario == "h200_tp8"]
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
            if case.scenario == profile and case.projection == "gate_up"
        ]

    assert tokens("h200_tp8") == tokens("h200_prefill_ep8")


def test_decode_tokens_are_already_per_gpu() -> None:
    """EP8 decode runs with DP8, so the declared token counts are per GPU."""

    decode = [
        case
        for case in PERFORMANCE_CASES
        if case.scenario == "h200_decode_ep8" and case.projection == "gate_up"
    ]
    assert [case.m for case in decode] == list(DECODE_TOKENS_PER_GPU)
    assert [case.routed_m for case in decode] == [
        tokens * 8 for tokens in DECODE_TOKENS_PER_GPU
    ]


@pytest.mark.parametrize(
    "routed_m,shape_k,expected",
    [
        # Below the mid-size window: minimize block count; the sampled
        # routing reproduces upstream's draw.
        (512, 7168, 24),
        # Mid-size window (row-work block-M 32..80): the row-work argmin.
        (1024, 7168, 32),
        (2048, 7168, 56),
        (3370, 7168, 80),
        # Past it, tok_e < 80 is back on the block-count argmin.
        (3371, 7168, 104),
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
    """Prefill block-M must track tok_e."""

    from chord_kernels.operator.tuning import _h200_prefill_block_m

    assert _h200_prefill_block_m(routed_m, 48, shape_k) == expected


def test_mid_tile_window_bounds() -> None:
    """The mid-size window is where the row-work argmin asks for 32..80 rows.

    Both shard axes share the rule; the bounds below are the routed-M edges of
    the sampled routing for 48 (EP8) and 384 (TP8) experts.
    """

    from chord_kernels.operator.tuning import _uses_mid_tiles

    for experts, first, last in ((48, 980, 3370), (384, 7725, 26148)):
        assert not _uses_mid_tiles(first - 1, experts)
        assert _uses_mid_tiles(first, experts)
        assert _uses_mid_tiles(last, experts)
        assert not _uses_mid_tiles(last + 1, experts)


def test_sampled_expert_counts_match_the_upstream_draw() -> None:
    """The cached prefix-sum routing equals a fresh upstream draw at every m."""

    import numpy as np

    from chord_kernels.operator.tuning import (
        _SAMPLED_ROUTING_ROWS,
        _sampled_expert_counts,
    )

    for experts in (48, 384):
        for routed_m in (1, 7, 1023, 1024, 1025, 4097, 131072,
                         _SAMPLED_ROUTING_ROWS, _SAMPLED_ROUTING_ROWS + 1):
            draw = np.random.RandomState(seed=0).randint(0, experts, size=routed_m)
            expected = np.bincount(draw, minlength=experts)
            assert np.array_equal(
                _sampled_expert_counts(routed_m, experts), expected
            ), (experts, routed_m)


@pytest.mark.parametrize(
    "routed_m,shape_k,expected",
    [
        # Below the mid-size window: the block-count argmin, same as EP8.
        (4096, 7168, 24),
        # Mid-size window: the row-work argmin, both projections.
        (8192, 7168, 32),
        (16384, 7168, 56),
        (16384, 256, 56),
        (26148, 7168, 80),
        # Past it, tok_e < 80 is back on the block-count argmin.
        (26149, 256, 104),
        # tok_e 80..128: gate/up keeps the row-work argmin, down sizes one
        # block per expert to its padded rows.
        (30720, 7168, 96),
        (30720, 256, 88),
        (49152, 7168, 152),
        (49152, 256, 144),
        # Then a flat 96 window to tok_e 190 ...
        (49153, 7168, 96),
        (72960, 256, 96),
        # ... then 128 for good.
        (72961, 7168, 128),
        (131072, 256, 128),
    ],
)
def test_tp8_block_m_follows_tokens_per_expert(
    routed_m: int, shape_k: int, expected: int
) -> None:
    """TP8 block-M uses the flatter TP-scale windows, not EP8's padding model.

    Both TP8 projections are narrow in whichever dimension fills the grid, so
    block count and occupancy dominate rather than per-expert M-padding: the
    96..144 band beats the taller tiles EP8 grows into.
    """

    from chord_kernels.operator.tuning import _h200_tp8_block_m

    assert _h200_tp8_block_m(routed_m, 384, shape_k) == expected


@pytest.mark.parametrize(
    "shape_k,routed_m,expected",
    [
        # down (K=256): only 4 K-blocks to split, so one-pass everywhere ...
        (256, 8192, False),
        (256, 65536, False),
        (256, 131072, False),
        # ... except the mid-size window's 256-wide tiles from block-M 56.
        (256, 13090, False),
        (256, 16384, True),
        (256, 21235, True),
        (256, 24061, False),
        # gate/up (K=7168): split everywhere except the mid-size window's
        # 512-wide one-pass tile, which splits only on a partial last wave.
        (7168, 4096, True),
        (7168, 8192, False),
        (7168, 10240, True),
        (7168, 16384, False),
        (7168, 26149, True),
        (7168, 65536, True),
        (7168, 131072, True),
    ],
)
def test_tp8_stream_k_gate(
    shape_k: int, routed_m: int, expected: bool
) -> None:
    """down is one-pass outside the mid-size window; gate/up splits outside it."""

    from chord_kernels.operator.tuning import _h200_tp8_use_stream_k

    assert _h200_tp8_use_stream_k(routed_m, 384, shape_k) is expected


@pytest.mark.parametrize(
    "shape_n,shape_k,routed_m,block_shape,num_stages,num_ctas_per_sm,use_stream_k",
    [
        # gate/up: the mid-size window's one-pass 512-wide tile covers the
        # whole output per m-block; past it the 256 tile with stream-K.
        (512, 7168, 8192, (32, 512, 64), 6, 1, False),
        (512, 7168, 16384, (56, 512, 64), 6, 1, False),
        (512, 7168, 32768, (104, 256, 64), 6, 1, True),
        (512, 7168, 65536, (96, 256, 64), 6, 1, True),
        (512, 7168, 131072, (128, 256, 64), 6, 1, True),
        # down: 2 CTAs/SM at every size -- 256 wide in the mid-size window,
        # block-N 128 beyond it; one-pass except the window's 56..72 rows.
        (7168, 256, 8192, (32, 256, 64), 6, 2, False),
        (7168, 256, 16384, (56, 256, 64), 4, 2, True),
        (7168, 256, 32768, (96, 128, 64), 6, 2, False),
        (7168, 256, 65536, (96, 128, 64), 6, 2, False),
        # 6 stages would leave room for one CTA; 5 keeps two.
        (7168, 256, 131072, (128, 128, 64), 5, 2, False),
    ],
)
def test_tp8_configs_at_benchmark_points(
    shape_n: int,
    shape_k: int,
    routed_m: int,
    block_shape: tuple[int, int, int],
    num_stages: int,
    num_ctas_per_sm: int,
    use_stream_k: bool,
) -> None:
    """Pin the schedule behind the TP8 table in docs/performance.md.

    ``num_tokens_total`` 1024..16384 (routed_m 8192..131072) for both projections,
    so a change here is a change to published numbers.
    """

    from chord_kernels.operator.tuning import _select_indexed_kernel_config

    config = _select_indexed_kernel_config("h200_tp8", shape_n, shape_k, 384, routed_m)
    assert config.block_shape == block_shape
    assert config.warp_shape == (block_shape[0], 32, 64)
    assert config.num_stages == num_stages
    assert config.num_ctas_per_sm == num_ctas_per_sm
    assert config.use_stream_k is use_stream_k
    # TP8 is a WGMMA profile, so the swap-AB decode schedule never applies.
    assert config.swap_ab is False


def test_tp8_down_never_drops_to_one_cta() -> None:
    """down's short K needs 2 CTAs/SM at every size, so every row must fit two.

    Block-N 256 fits two CTAs only up to block-M 72 (the mid-size window);
    past it block-N 128 is what keeps the second CTA.  The shared-memory check
    covers the stage depth too, across the sweep and not just the table's
    points.
    """

    from chord_kernels.operator.tuning import _select_indexed_kernel_config
    from chord_kernels.operator.tuning import (
        _H200_SMEM_PER_SM,
        _H200_SMEM_RESERVED_PER_CTA,
        _indexed_smem_bytes,
    )

    for routed_m in (1, 512, 8192, 16384, 26148, 32768, 65536, 131072, 262144):
        config = _select_indexed_kernel_config("h200_tp8", 7168, 256, 384, routed_m)
        assert config.num_ctas_per_sm == 2, routed_m
        if config.block_n == 256:
            assert config.block_m <= 72, routed_m
        else:
            assert config.block_n == 128, routed_m
        per_cta = _indexed_smem_bytes(*config.block_shape, config.num_stages)
        assert 2 * (per_cta + _H200_SMEM_RESERVED_PER_CTA) <= _H200_SMEM_PER_SM


def test_stage_depth_never_costs_a_resident_cta() -> None:
    """Every published H200 row fits its CTAs/SM in shared memory."""

    from chord_kernels.operator.tuning import (
        _H200_SMEM_PER_SM,
        _H200_SMEM_RESERVED_PER_CTA,
        _indexed_smem_bytes,
        _select_indexed_kernel_config,
    )

    for name, shapes, experts in (
        ("h200_prefill_ep8", ((4096, 7168), (7168, 2048)), 48),
        ("h200_tp8", ((512, 7168), (7168, 256)), 384),
    ):
        for shape_n, shape_k in shapes:
            configs = {
                _select_indexed_kernel_config(name, shape_n, shape_k, experts, m)
                for m in range(1, 368 * experts + 1, 7)
            }
            for config in configs:
                per_cta = _indexed_smem_bytes(*config.block_shape, config.num_stages)
                resident = config.num_ctas_per_sm * (
                    per_cta + _H200_SMEM_RESERVED_PER_CTA
                )
                assert resident <= _H200_SMEM_PER_SM, (name, shape_n, config)


def test_prefill_block_m_is_deterministic() -> None:
    """The block-count argmin samples a routing, so it must use a fixed seed."""

    from chord_kernels.operator.tuning import _h200_prefill_block_m

    first = [_h200_prefill_block_m(m, 48, 7168) for m in (512, 1024, 2048)]
    second = [_h200_prefill_block_m(m, 48, 7168) for m in (512, 1024, 2048)]
    assert first == second


@pytest.mark.parametrize(
    "shape_n,shape_k,routed_m,expected",
    [
        # gate/up (deep K): stream-K helps at every prefill size outside the
        # mid-size window ...
        (4096, 7168, 512, True),
        (4096, 7168, 4096, True),
        (4096, 7168, 16384, True),
        # ... and inside it only when the 512-wide grid ends in a partial wave.
        (4096, 7168, 1024, False),
        (4096, 7168, 1280, True),
        (4096, 7168, 2048, False),
        # down (mid K, 512 < K < 4096): one-pass in the mid-size window, then
        # stream-K only until routed_m hits the crossover.
        (7168, 2048, 512, True),
        (7168, 2048, 1024, False),
        (7168, 2048, 2048, False),
        (7168, 2048, 4096, True),
        (7168, 2048, 5120, False),
        (7168, 2048, 16384, False),
    ],
)
def test_prefill_stream_k_gate(
    shape_n: int, shape_k: int, routed_m: int, expected: bool
) -> None:
    """Prefill enables stream-K except the mid-size window and large mid-K down."""

    from chord_kernels.operator.tuning import _h200_prefill_use_stream_k

    assert _h200_prefill_use_stream_k(routed_m, 48, shape_n, shape_k) == expected


@pytest.mark.parametrize(
    "shape_n,shape_k,routed_m,block_m,num_ctas_per_sm",
    [
        # gate/up (deep K): 3 CTAs/SM at every decode size.
        (4096, 7168, 160, 8, 3),
        (4096, 7168, 400, 16, 3),
        (4096, 7168, 800, 24, 3),
        # down: the block-M 8 tile keeps the fourth CTA, larger tiles take 3.
        (7168, 2048, 160, 8, 4),
        (7168, 2048, 240, 8, 4),
        (7168, 2048, 320, 16, 3),
        (7168, 2048, 800, 24, 3),
    ],
)
def test_h200_decode_ctas_per_sm(
    shape_n: int, shape_k: int, routed_m: int, block_m: int, num_ctas_per_sm: int
) -> None:
    """H200 decode runs 3 CTAs/SM except down's block-M 8 swap-AB tile."""

    from chord_kernels.operator.tuning import _select_indexed_kernel_config

    config = _select_indexed_kernel_config("h200_decode_ep8", shape_n, shape_k, 48, routed_m)
    assert config.block_shape == (block_m, 256, 64)
    assert config.swap_ab is True
    assert config.num_ctas_per_sm == num_ctas_per_sm


def test_decode_stream_k_policy() -> None:
    """H200 decode is one-pass; Blackwell decode splits K per projection.

    The deep-K gate/up projection keeps stream-K at every decode size (the K
    loop is 112 blocks, so the tail balance repays the locks), while the short-K
    down projection stays one-pass through the swap-AB window and only enables
    the split on the large non-swap tiles.
    """

    from chord_kernels.operator.tuning import _select_indexed_kernel_config

    for shape_n, shape_k in ((4096, 7168), (7168, 2048)):
        for routed_m in (160, 400, 8192):
            config = _select_indexed_kernel_config("h200_decode_ep8", shape_n, shape_k, 48, routed_m)
            assert not config.use_stream_k

    def blackwell(shape_n, shape_k, routed_m):
        return _select_indexed_kernel_config("blackwell_decode_ep8", shape_n, shape_k, 48, routed_m)

    for routed_m in (160, 400, 1200, 8192):
        assert blackwell(4096, 7168, routed_m).use_stream_k
    for routed_m in (160, 400, 848):
        assert not blackwell(7168, 2048, routed_m).use_stream_k
    for routed_m in (1200, 8192):
        assert blackwell(7168, 2048, routed_m).use_stream_k


def test_prefill_forces_two_ctas_in_block_m_window() -> None:
    """Prefill runs 2 CTAs/SM for the 40..80 block-M window at block_n=256.

    Doubling resident CTAs there hides the cp.async + dequant latency and is
    ~11% faster on the down 1024/2048 tiles.  The mid-size window's down tile
    runs 2 CTAs/SM from block-M 32 for the same reason, while its gate/up
    512-wide tile and everything else (small block_m at block_n=128, or
    block_m > 80) stays at 1 CTA/SM.
    """

    from chord_kernels.operator.tuning import _select_indexed_kernel_config
    from chord_kernels.operator.tuning import _uses_mid_tiles

    for shape_n, shape_k in ((4096, 7168), (7168, 2048)):
        for routed_m in (512, 900, 1024, 2048, 3500, 4096, 8192, 16384, 65536):
            config = _select_indexed_kernel_config(
                "h200_prefill_ep8", shape_n, shape_k, 48, routed_m
            )
            in_window = config.block_n == 256 and 40 <= config.block_m <= 80
            if _uses_mid_tiles(routed_m, 48):
                in_window = config.block_n == 256
            expected = 2 if in_window else 1
            assert config.num_ctas_per_sm == expected, (
                f"n={shape_n} k={shape_k} routed_m={routed_m} "
                f"block_m={config.block_m} block_n={config.block_n} -> "
                f"{config.num_ctas_per_sm} CTAs/SM, expected {expected}"
            )


def test_traffic_bytes_counts_only_active_experts() -> None:
    case = PERFORMANCE_CASES[0]
    assert _case_traffic_bytes(case, 48) > _case_traffic_bytes(case, 8)


def test_select_indexed_config_contract() -> None:
    """The public selector picks the table from the GPU, the shape and routed-M,
    keeps N/K under a shared routing block-M, and checks its inputs."""

    from chord_kernels.operator import IndexedKernelConfig
    from chord_kernels.operator.tuning import select_indexed_config
    from chord_kernels.operator.tuning import (
        _H200_DECODE_MAX_TOKENS_PER_EXPERT,
        _select_indexed_kernel_config,
    )

    h200, b300 = (9, 0), (10, 3)

    def pick(n, k, e, m, cap, **kw):
        return select_indexed_config(n, k, e, m, capability=cap, **kw)

    # Every published benchmark point resolves to the table it was tuned in.
    for scenario, cap, (n, k), e, routed in (
        ("h200_prefill_ep8", h200, (4096, 7168), 48, (1024, 2048, 4096, 8192, 16384)),
        ("h200_prefill_ep8", h200, (7168, 2048), 48, (1024, 2048, 4096, 8192, 16384)),
        ("h200_tp8", h200, (512, 7168), 384, (8192, 16384, 32768, 65536, 131072)),
        ("h200_tp8", h200, (7168, 256), 384, (8192, 16384, 32768, 65536, 131072)),
        ("h200_decode_ep8", h200, (4096, 7168), 48, (160, 240, 320, 400)),
        ("h200_decode_ep8", h200, (7168, 2048), 48, (160, 240, 320, 400)),
        ("blackwell_decode_ep8", b300, (4096, 7168), 48, (160, 240, 320, 400, 8192)),
        ("blackwell_decode_ep8", (10, 0), (7168, 2048), 48, (160, 1200)),
    ):
        for m in routed:
            assert pick(n, k, e, m, cap) == _select_indexed_kernel_config(scenario, n, k, e, m)

    # H200 EP8 switches from the swap-AB decode table to the WGMMA prefill
    # table at the measured crossover.
    edge = int(_H200_DECODE_MAX_TOKENS_PER_EXPERT * 48)
    assert pick(4096, 7168, 48, edge - 8, h200).swap_ab
    assert pick(4096, 7168, 48, edge + 8, h200).warp_shape[1] == 32

    # Unpublished shapes run the untuned fallback of the chosen family.
    assert pick(2048, 2048, 48, 4096, h200) == IndexedKernelConfig.default(
        layout="wgmma", block_m=16, block_n=256, swap_ab=False
    )
    assert pick(2048, 2048, 48, 160, b300) == IndexedKernelConfig.default(
        layout="mma", block_m=8, block_n=256, swap_ab=True
    )

    # A shared routing block-M replaces the M tile and keeps N/K.
    base = pick(4096, 7168, 48, 3072, h200)
    for block_m in (8, 16, 24, 48):
        shared = pick(4096, 7168, 48, 3072, h200, block_m=block_m)
        assert shared.block_m == shared.warp_shape[0] == block_m
        assert shared.block_shape[1:] == base.block_shape[1:]
    assert pick(4096, 7168, 48, 160, b300, block_m=24).swap_ab

    with pytest.raises(RuntimeError, match="supports SM90"):
        pick(4096, 7168, 48, 128, (8, 0))
    with pytest.raises(ValueError, match="routing block_m"):
        pick(4096, 7168, 48, 128, h200, block_m=12)
    with pytest.raises(ValueError, match="routed_m"):
        pick(4096, 7168, 48, -1, h200)
    with pytest.raises(ValueError, match="shape_n"):
        pick(0, 7168, 48, 128, h200)


def test_public_api_surface() -> None:
    """The package exports the operators and packers, and nothing tuning-specific."""

    import chord_kernels

    assert sorted(chord_kernels.__all__) == [
        "contiguous", "indexed", "indexed_block_m", "masked", "pack_w4a16",
    ]
    for name in chord_kernels.__all__:
        assert callable(getattr(chord_kernels, name))
    import inspect

    params = inspect.signature(chord_kernels.indexed).parameters
    assert not {"config", "layout", "swap_ab"} & set(params)


def test_indexed_block_m_matches_the_tuned_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """``indexed_block_m`` is the tuned config's block-M for routed_m = tokens * top_k."""

    from chord_kernels import indexed_block_m
    from chord_kernels.operator import PreparedWeight
    from chord_kernels.operator import tuning

    monkeypatch.setattr(tuning, "_device_capability", lambda index: (9, 0))
    for (n, k, e), tokens in (
        ((4096, 7168, 48), 20), ((7168, 2048, 48), 2048), ((512, 7168, 384), 1024),
    ):
        w = PreparedWeight(packed=torch.empty(0), scale=torch.empty(0), n=n, k=k,
                           num_experts=e)
        expected = tuning.select_indexed_config(n, k, e, tokens * 8, capability=(9, 0)).block_m
        assert indexed_block_m(w, tokens, 8, device=0) == expected


def test_indexed_kernels_covers_every_selected_config() -> None:
    """``indexed_kernels`` (the precompile list) holds every config the selector
    can return for a projection, each once."""

    from chord_kernels.operator.tuning import select_indexed_config
    from chord_kernels.operator.cpp_jit import indexed_kernels

    for cap, (n, k), e in (((10, 3), (4096, 7168), 48), ((9, 0), (7168, 2048), 48)):
        configs = [c for _, _, _, c in indexed_kernels(n, k, e, capability=cap)]
        assert len(configs) == len(set(configs))
        for routed_m in list(range(0, 3000, 7)) + [1 << 20]:
            assert select_indexed_config(n, k, e, routed_m, capability=cap) in configs


def test_checkpoint_uint4_layout() -> None:
    """Eight little-endian nibbles per word: code i of a row is word i // 8,
    shift 4 * (i % 8) (the ``packed=True`` contract)."""
    logical = torch.arange(16, dtype=torch.int32).repeat(1, 128, 4)
    words = pack_checkpoint_uint4(logical).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64) * 4
    codes = ((words.unsqueeze(-1) >> shifts) & 0xF).reshape(logical.shape)
    assert torch.equal(codes.to(torch.int32), logical)


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


def _stream_k_swap_ab_case(device: torch.device, seed: int):
    """A small INT4 swap-AB stream-K gate/up problem plus its reference.

    K = 7168 (112 K-blocks) at 4 CTAs/SM leaves the tail tiles split more than
    three ways, so the stream-K reduction is exercised on every launch.
    """
    from chord_kernels import pack_w4a16
    from chord_kernels.operator import IndexedKernelConfig

    num_experts, top_k, m, n, k = 8, 4, 24, 512, 7168
    gen = torch.Generator(device=device).manual_seed(seed)
    codes = torch.randint(0, 16, (num_experts, n, k), dtype=torch.int32, device=device,
                          generator=gen)
    scale = (torch.rand((num_experts, n, k // 32), device=device, generator=gen) * 0.04
             + 0.005).to(torch.bfloat16)
    weight = pack_w4a16(codes, scale)
    config = IndexedKernelConfig(
        block_shape=(8, 256, 64), warp_shape=(8, 64, 64), num_stages=4,
        num_ctas_per_sm=4, swap_ab=True, use_stream_k=True,
    )
    topk_ids = generate_topk_ids(num_tokens=m, top_k=top_k, num_experts=num_experts,
                                 device=device, generator=gen)
    sorted_ids, expert_ids, ntp = align_routing_blocks(
        topk_ids, block_m=config.block_m, num_experts=num_experts)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=gen)
    reference, rows = reference_indexed(activation, codes, scale, topk_ids, top_k,
                                        projection="gate_up")
    return SimpleNamespace(weight=weight, config=config, sorted_ids=sorted_ids,
                           expert_ids=expert_ids, ntp=ntp, activation=activation,
                           top_k=top_k, n=n, rows=rows, reference=reference)


@pytest.mark.gpu
@pytest.mark.parametrize("stream_k_fp32", [False, True])
def test_stream_k_first_launch_inside_cuda_graph_capture(
    supported_gpu: torch.device, stream_k_fp32: bool,
) -> None:
    """A stream's first stream-K launch may be inside a CUDA-graph capture.

    vLLM captures decode graphs on a stream that never ran the kernel eagerly.
    The caller's lock buffer is allocated beforehand and the FP32 workspace
    comes from the caching allocator, so nothing in the launch allocates
    outside PyTorch: the capture must succeed, and graph replays and an eager
    launch on the capture stream must match the reference (FP32 replays bit for
    bit).
    """
    import dataclasses

    from chord_kernels.operator import w4a16_indexed as indexed

    case = _stream_k_swap_ab_case(supported_gpu, seed=20261009)
    config = dataclasses.replace(case.config, stream_k_fp32=stream_k_fp32)
    locks = torch.zeros(1024, dtype=torch.int32, device=supported_gpu)

    def run(out):
        return indexed(case.activation, case.weight, case.sorted_ids, case.expert_ids,
                       case.ntp, case.top_k, outputs=out, config=config,
                       validate_routing=False, locks=locks)

    # Compile + load the kernel eagerly on the default stream only.
    run(torch.empty((case.rows, case.n), dtype=torch.bfloat16, device=supported_gpu))
    torch.cuda.synchronize(supported_gpu)

    capture_stream = torch.cuda.Stream(device=supported_gpu)
    graph_out = torch.zeros((case.rows, case.n), dtype=torch.bfloat16, device=supported_gpu)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run(graph_out)
    replays = []
    for _ in range(3):
        graph_out.zero_()
        graph.replay()
        torch.cuda.synchronize(supported_gpu)
        cosine, _ = _check_output(graph_out[: case.rows], case.reference)
        assert cosine <= _COSINE_LIMIT
        replays.append(graph_out.clone())

    with torch.cuda.stream(capture_stream):
        eager_out = run(torch.empty_like(graph_out))
    torch.cuda.synchronize(supported_gpu)
    cosine, _ = _check_output(eager_out[: case.rows], case.reference)
    assert cosine <= _COSINE_LIMIT
    if stream_k_fp32:
        assert all(torch.equal(r, eager_out) for r in replays)


@pytest.mark.gpu
def test_mma_stream_k_fp32_matches_one_pass(supported_gpu: torch.device) -> None:
    """MMA stream-K reduces in FP32: deterministic, and equal to the same tile run
    in one pass up to FP32 summation order.  The BF16 reduction it replaces
    flipped 10-55% of outputs."""
    import dataclasses

    from chord_kernels.operator import w4a16_indexed as indexed

    case = _stream_k_swap_ab_case(supported_gpu, seed=20261013)
    assert case.config.stream_k_fp32
    one_pass = dataclasses.replace(case.config, use_stream_k=False, stream_k_fp32=False)
    locks = torch.zeros(1024, dtype=torch.int32, device=supported_gpu)

    def run(config):
        return indexed(case.activation, case.weight, case.sorted_ids, case.expert_ids,
                       case.ntp, case.top_k, config=config, validate_routing=False,
                       locks=locks)

    reference = run(one_pass)
    outputs = [run(case.config) for _ in range(5)]
    torch.cuda.synchronize(supported_gpu)
    assert all(torch.equal(out, outputs[0]) for out in outputs[1:])
    assert (outputs[0] == reference).double().mean().item() >= 0.995


def test_stream_k_fp32_is_the_default_for_stream_k() -> None:
    """Stream-K configs (MMA and WGMMA) reduce in FP32 unless asked not to."""
    from chord_kernels.operator import IndexedKernelConfig
    from chord_kernels.operator.tuning import _select_indexed_kernel_config

    mma = dict(block_shape=(8, 256, 64), warp_shape=(8, 64, 64), num_ctas_per_sm=4,
               swap_ab=True)
    stream_k = IndexedKernelConfig(use_stream_k=True, **mma)
    assert stream_k.stream_k_fp32
    one_pass = IndexedKernelConfig(**mma)
    assert not one_pass.stream_k_fp32
    wgmma = IndexedKernelConfig(block_shape=(64, 128, 64), warp_shape=(64, 32, 64),
                                use_stream_k=True)
    assert wgmma.stream_k_fp32
    bf16 = IndexedKernelConfig(use_stream_k=True, stream_k_fp32=False, **mma)
    assert not bf16.stream_k_fp32

    # The default does not show up in the serialized rows; an override does and
    # survives a round trip.
    assert "stream_k_fp32" not in stream_k.to_dict()
    assert "stream_k_fp32" not in one_pass.to_dict()
    for config in (stream_k, one_pass, wgmma, bf16):
        assert IndexedKernelConfig.from_dict(config.to_dict()) == config
    assert bf16.to_dict()["stream_k_fp32"] is False
    with pytest.raises(TypeError, match="stream_k_fp32"):
        IndexedKernelConfig(use_stream_k=True, stream_k_fp32=1, **mma)

    # Every stream-K row of every published profile reduces in FP32.
    shapes = {
        "h200_prefill_ep8": [(4096, 7168, 48), (7168, 2048, 48)],
        "h200_tp8": [(512, 7168, 384), (7168, 256, 384)],
        "h200_decode_ep8": [(4096, 7168, 48), (7168, 2048, 48)],
        "blackwell_decode_ep8": [(4096, 7168, 48), (7168, 2048, 48)],
    }
    for name, profile_shapes in shapes.items():
        for n, k, e in profile_shapes:
            for routed_m in list(range(1, 4097, 31)) + [16384, 131072]:
                config = _select_indexed_kernel_config(name, n, k, e, routed_m)
                assert config.stream_k_fp32 == config.use_stream_k


@pytest.mark.gpu
@pytest.mark.parametrize(
    "block_shape,num_ctas_per_sm,num_tokens",
    [((40, 256, 64), 2, 16), ((96, 256, 64), 1, 128), ((176, 256, 64), 1, 256)],
    ids=["40x256x2", "96x256", "176x256"],
)
def test_wgmma_stream_k_fp32_matches_one_pass(
    block_shape: tuple[int, int, int], num_ctas_per_sm: int, num_tokens: int,
) -> None:
    """WGMMA stream-K reduces in FP32: deterministic, and equal to one pass up to
    FP32 summation order.  The 176-row tile is the one that needs the
    accumulator fence after ``wgmma.wait_group`` (without it the epilogue read
    in-flight accumulators and two launches disagreed)."""
    import dataclasses

    from chord_kernels.operator import w4a16_indexed as indexed
    from chord_kernels import pack_w4a16
    from chord_kernels.operator import IndexedKernelConfig

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the WGMMA indexed layout needs SM90")
    device = torch.device("cuda")
    num_experts, top_k, n, k = 48, 8, 4096, 7168
    gen = torch.Generator(device=device).manual_seed(20261013 + num_tokens)
    codes = torch.randint(0, 16, (num_experts, n, k), dtype=torch.int32, device=device,
                          generator=gen)
    scale = (torch.rand((num_experts, n, k // 32), device=device, generator=gen) * 0.04
             + 0.005).to(torch.bfloat16)
    weight = pack_w4a16(codes, scale)
    del codes, scale
    stream_k = IndexedKernelConfig(
        block_shape=block_shape, warp_shape=(block_shape[0], 32, 64),
        num_ctas_per_sm=num_ctas_per_sm, use_stream_k=True,
    )
    assert stream_k.stream_k_fp32
    one_pass = dataclasses.replace(stream_k, use_stream_k=False, stream_k_fp32=False)
    topk_ids = generate_topk_ids(num_tokens=num_tokens, top_k=top_k,
                                 num_experts=num_experts, device=device, generator=gen)
    sorted_ids, expert_ids, ntp = align_routing_blocks(
        topk_ids, block_m=stream_k.block_m, num_experts=num_experts)
    activation = torch.randn((num_tokens, k), dtype=torch.bfloat16, device=device,
                             generator=gen)
    locks = torch.zeros(1024, dtype=torch.int32, device=device)

    def run(config):
        return indexed(activation, weight, sorted_ids, expert_ids, ntp, top_k,
                       config=config, validate_routing=False, locks=locks)

    reference = run(one_pass)
    outputs = [run(stream_k) for _ in range(5)]
    torch.cuda.synchronize(device)
    assert all(torch.equal(out, outputs[0]) for out in outputs[1:])
    assert (outputs[0] == reference).double().mean().item() >= 0.995


@pytest.mark.gpu
def test_stream_k_caller_owned_locks_and_scratch_workspace(
    supported_gpu: torch.device,
) -> None:
    """Stream-K buffers: the caller's lock tensor, and FP32 partials in
    per-launch scratch from the PyTorch allocator.

    A stream-K launch without locks, or with a buffer smaller than the launch's
    tile bound, is rejected; a one-pass config ignores locks.  Eager launches and
    a captured graph on the caller's locks are bit-identical and leave every lock
    word at 0, and repeated launches do not grow allocated memory.
    """
    import dataclasses

    from chord_kernels.operator import w4a16_indexed as indexed

    case = _stream_k_swap_ab_case(supported_gpu, seed=20261014)
    assert case.config.stream_k_fp32
    locks = torch.zeros(1024, dtype=torch.int32, device=supported_gpu)

    def run(out=None, owned_locks=None, config=case.config):
        return indexed(case.activation, case.weight, case.sorted_ids, case.expert_ids,
                       case.ntp, case.top_k, outputs=out, config=config,
                       validate_routing=False, locks=owned_locks)

    with pytest.raises(ValueError, match="locks"):
        run()
    with pytest.raises(RuntimeError, match="lock words"):
        run(owned_locks=torch.zeros(8, dtype=torch.int32, device=supported_gpu))
    one_pass = dataclasses.replace(case.config, use_stream_k=False, stream_k_fp32=False)
    run(config=one_pass)

    eager = run(owned_locks=locks)
    torch.cuda.synchronize(supported_gpu)
    assert not locks.any()
    _check_output(eager[: case.rows], case.reference)

    before = torch.cuda.memory_allocated(supported_gpu)
    for _ in range(4):
        run(eager, locks)
    torch.cuda.synchronize(supported_gpu)
    assert torch.cuda.memory_allocated(supported_gpu) == before

    capture_stream = torch.cuda.Stream(device=supported_gpu)
    graph_out = torch.zeros_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run(graph_out, locks)
    for _ in range(3):
        graph_out.zero_()
        graph.replay()
        torch.cuda.synchronize(supported_gpu)
        assert torch.equal(graph_out, eager)
        assert not locks.any()


@pytest.mark.gpu
def test_one_packed_weight_serves_mma_and_wgmma() -> None:
    """One packed weight feeds the WGMMA prefill/TP8 tile, the MMA non-swap tile
    and the swap-AB decode tile, each matching the BF16 dequant reference."""
    from chord_kernels.operator import w4a16_indexed as indexed
    from chord_kernels import pack_w4a16
    from chord_kernels.operator import IndexedKernelConfig

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the WGMMA indexed kernel needs SM90")
    device = torch.device("cuda")
    num_experts, top_k, num_tokens, n, k = 8, 4, 64, 256, 1024
    gen = torch.Generator(device=device).manual_seed(20261014)
    codes = torch.randint(0, 16, (num_experts, n, k), dtype=torch.int32, device=device,
                          generator=gen)
    scale = (torch.rand((num_experts, n, k // 32), device=device, generator=gen) * 0.04
             + 0.005).to(torch.bfloat16)
    weight = pack_w4a16(codes, scale)

    activation = torch.randn((num_tokens, k), dtype=torch.bfloat16, device=device,
                             generator=gen)
    topk_ids = generate_topk_ids(num_tokens=num_tokens, top_k=top_k,
                                 num_experts=num_experts, device=device, generator=gen)
    reference, rows = reference_indexed(activation, codes, scale, topk_ids, top_k)
    configs = (
        IndexedKernelConfig(block_shape=(16, 128, 64), warp_shape=(16, 32, 64)),
        IndexedKernelConfig(block_shape=(16, 256, 64), warp_shape=(16, 64, 64)),
        IndexedKernelConfig(block_shape=(8, 256, 64), warp_shape=(8, 64, 64),
                            num_ctas_per_sm=4, swap_ab=True),
    )
    for config in configs:
        sorted_ids, expert_ids, ntp = align_routing_blocks(
            topk_ids, block_m=config.block_m, num_experts=num_experts)
        actual = indexed(activation, weight, sorted_ids, expert_ids, ntp, top_k,
                         config=config, validate_routing=False)
        _check_output(actual[:rows], reference)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "block_shape,num_stages",
    [((56, 256, 128), 6), ((56, 128, 256), 4), ((72, 256, 128), 6), ((88, 128, 128), 6)],
    ids=["56x256x128", "56x128x256", "72x256x128", "88x128x128"],
)
def test_wgmma_split_k_odd_warp_m(block_shape: tuple[int, int, int], num_stages: int) -> None:
    """block_k > 64 splits K across warps and sums them in shared memory.  A
    warp-M of 16q+8 holds a number of accumulator int4s its q reduction rounds
    do not divide; the last ones used to be dropped (cos diff 1e-4 to 2e-2)."""
    from chord_kernels import pack_w4a16
    from chord_kernels.operator import IndexedKernelConfig, w4a16_indexed as indexed

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the WGMMA indexed kernel needs SM90")
    device = torch.device("cuda")
    num_experts, top_k, num_tokens, n, k = 16, 8, 192, 1024, 2048
    gen = torch.Generator(device=device).manual_seed(20261020)
    codes = torch.randint(0, 16, (num_experts, n, k), dtype=torch.int32, device=device, generator=gen)
    scale = (torch.rand((num_experts, n, k // 32), device=device, generator=gen) * 0.04
             + 0.005).to(torch.bfloat16)
    weight = pack_w4a16(codes, scale)
    activation = torch.randn((num_tokens, k), dtype=torch.bfloat16, device=device, generator=gen)
    topk_ids = generate_topk_ids(num_tokens=num_tokens, top_k=top_k, num_experts=num_experts,
                                 device=device, generator=gen)
    reference, rows = reference_indexed(activation, codes, scale, topk_ids, top_k)
    config = IndexedKernelConfig(block_shape=block_shape, warp_shape=(block_shape[0], 32, 64),
                                 num_stages=num_stages)
    sorted_ids, expert_ids, ntp = align_routing_blocks(
        topk_ids, block_m=config.block_m, num_experts=num_experts)
    out = indexed(activation, weight, sorted_ids, expert_ids, ntp, top_k, config=config)
    cosine, _ = _check_output(out[:rows], reference, cosine_limit=1e-5)
    assert cosine < 1e-5


def test_wgmma_register_budget_guard() -> None:
    """A WGMMA config of 512 threads at 2 CTAs/SM leaves under 64 registers a
    thread; it spilled and computed wrong results, so it is rejected."""
    from chord_kernels.operator import IndexedKernelConfig
    from chord_kernels.operator.api import _get_indexed_kernel

    config = IndexedKernelConfig(block_shape=(32, 512, 64), warp_shape=(32, 32, 64),
                                 num_ctas_per_sm=2)
    with pytest.raises(ValueError, match="threads x CTAs/SM"):
        _get_indexed_kernel.__wrapped__(0, 4096, 7168, 48, config)
