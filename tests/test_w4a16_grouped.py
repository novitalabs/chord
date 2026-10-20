#!/usr/bin/env python3
"""Correctness and performance checks for the grouped W4A16 operators.

These tests exercise chord's ``contiguous`` (prefill, BLOCK_K=64) and ``masked``
(decode, BLOCK_K=128) grouped GEMM entry points end to end: checkpoint weight ->
pack -> grouped GEMM -> BF16 dequant reference.  The kernel behind them is
adapted from DeepGEMM, but nothing here tests DeepGEMM itself: the reference is
computed in plain PyTorch against chord's own packing code, and the test
deliberately never imports ``deep_gemm``.

Run the file directly and it prints the roofline table straight away::

    python tests/test_w4a16_grouped.py

The same cases are also collected by pytest.  The GPU cases need ``-s`` for the
table to reach the terminal, and the production-sized ones additionally need
``--run-perf`` because they are marked ``perf`` and skipped by default::

    python -m pytest -m "not gpu" tests/test_w4a16_grouped.py
    python -m pytest -m gpu -s tests/test_w4a16_grouped.py
    python -m pytest --run-perf -m "gpu and perf" -s tests/test_w4a16_grouped.py
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Running this file directly puts tests/ on sys.path[0], not the repo root, so an
# uninstalled source checkout would not find chord_kernels.  Add the root too.
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from roofline import (  # noqa: E402  (needs the sys.path setup above)
    cosine_diff as _cos_diff,
    format_throughput,
    print_group,
    print_row,
    roofline,
    throughput_columns,
    traffic_bytes,
)
from bench import (  # noqa: E402  (needs the sys.path setup above)
    BENCH_METHODS,
    DEFAULT_BENCH_METHOD,
    benchmark,
)

_COSINE_LIMIT = 1e-3
_W4A16_GROUP = 32


def _heuristics():
    """Import the pure-python heuristic module without pulling in torch users."""
    path = (
        Path(__file__).resolve().parents[1]
        / "chord_kernels/operator/grouped/heuristics.py"
    )
    spec = importlib.util.spec_from_file_location("chord_dg_heuristics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dequant_reference(
    codes: torch.Tensor, scale_n_major: torch.Tensor
) -> torch.Tensor:
    """codes [N, K] int in [0, 15] (weight = code - 8), scale [N, K/32] bf16
    -> BF16 [N, K]."""
    n, k = codes.shape
    signed = codes.float() - 8.0
    scale = scale_n_major.float().reshape(n, k // _W4A16_GROUP, 1)
    return signed.reshape(n, k // _W4A16_GROUP, _W4A16_GROUP) * scale


@dataclass(frozen=True)
class MaskedCase:
    num_groups: int
    max_m: int
    expected_m_per_group: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return (
            f"G={self.num_groups} em={self.expected_m_per_group} "
            f"N={self.n} K={self.k}"
        )


# Per-expert runs must begin on a BLOCK_M boundary: the contiguous scheduler
# reads the expert id at each tile's first row
# (``grouped_layout[m_block_idx * BLOCK_M]``) and applies it to the whole tile,
# so a tile may not straddle two experts.  W4A16 picks BLOCK_M 64 or 128, and
# 128 is the safe bound that covers both -- the same value DeepGEMM carries as
# ``kLegacyMKAlignmentForContiguousLayout``.  Rows past an expert's real count
# are masked out per row (``grouped_layout[...] >= 0``), so they never corrupt
# the output but they DO occupy tiles and cost time.
CONTIGUOUS_ROW_ALIGNMENT = 128


@dataclass(frozen=True)
class ContiguousCase:
    num_groups: int
    expected_m_per_group: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return (
            f"G={self.num_groups} mpg={self.expected_m_per_group} "
            f"N={self.n} K={self.k}"
        )


# The upstream tuning matrix restricted to a fast, representative subset:
# EP8/EP16/EP32 groups, both projections, small/mid/large decode buckets.
# EP8 (G=48) is the shape docs/performance.md (Shapes) derives, and it is also the width
# where the masked BN rule flips: BN=256 needs enough N-tiles to fill the
# machine, and the tile count scales with G.
MASKED_CASES = [
    MaskedCase(g, 256, em, n, k)
    for g in (48, 24, 12)
    for (n, k) in ((4096, 7168), (7168, 2048))
    for em in (8, 16, 32, 64)
]
# ``m/grp`` values are multiples of CONTIGUOUS_ROW_ALIGNMENT so every expert
# fills whole tiles: the kernel then processes exactly the useful rows, and the
# timing is directly comparable with a baseline fed the same per-expert counts.
CONTIGUOUS_CASES = [
    ContiguousCase(g, mpg, n, k)
    for g in (48, 24)
    for (n, k) in ((4096, 7168), (7168, 2048))
    for mpg in (128, 256, 512)
]

# Kimi K2.5 routing, the configuration these shapes are derived for; see
# docs/performance.md (Shapes).  Used only to restate ``m/grp`` as the serving-side token
# count each row models, so a tuning row can be matched to a deployment.
_ROUTED_EXPERTS = 384
_TOP_K = 8


def prefill_chunk_tokens(case: ContiguousCase) -> int:
    """System-wide chunk size a contiguous-prefill row models.

    A chunk of ``T`` tokens fans out to ``T * top_k`` routes across all 384
    experts, of which this rank owns ``G``, so ``T * top_k * G / 384`` rows
    land here and each local expert sees ``m/grp`` of them.  Solving for ``T``
    cancels ``G``: at a given ``m/grp`` every EP width models the same chunk,
    which is what makes the two ``G`` blocks comparable row for row.
    """
    return case.expected_m_per_group * _ROUTED_EXPERTS // _TOP_K


def _reset_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def make_masked_case(case: MaskedCase, device: torch.device):
    """Build (a3, codes, scale, masked_m, ref) for the masked convention."""
    _reset_seed()
    g, max_m, em, n, k = (
        case.num_groups, case.max_m, case.expected_m_per_group, case.n, case.k,
    )
    a3 = torch.randn((g, max_m, k), device=device, dtype=torch.bfloat16)
    codes = torch.randint(
        0, 16, (g, n, k), device=device, dtype=torch.int8
    ).to(torch.int32)
    scale = (
        torch.randn((g, n, k // _W4A16_GROUP), device=device) * 0.02
    ).to(torch.bfloat16)
    masked_m = torch.empty((g,), device=device, dtype=torch.int32)
    for j in range(g):
        masked_m[j] = int(em * random.uniform(0.7, 1.3))
    masked_m.clamp_(1, max_m)

    ref = torch.zeros((g, max_m, n), device=device, dtype=torch.bfloat16)
    for j in range(g):
        mm = int(masked_m[j].item())
        if mm == 0:
            continue
        b_deq = _dequant_reference(codes[j], scale[j]).reshape(n, k)
        ref[j, :mm] = (a3[j, :mm].float() @ b_deq.t()).to(torch.bfloat16)
    return a3, codes, scale, masked_m, ref


def make_contiguous_case(case: ContiguousCase, device: torch.device):
    """Build (a2, codes, scale, m_indices, ref, bounds) in the kernel's native
    contiguous layout: per-expert runs starting on a
    ``CONTIGUOUS_ROW_ALIGNMENT`` boundary, padding rows zeroed with
    m_indices == -1."""
    _reset_seed()
    g, mpg, n, k = case.num_groups, case.expected_m_per_group, case.n, case.k
    align = CONTIGUOUS_ROW_ALIGNMENT
    actual_ms = [mpg] * g
    aligned_ms = [((am + align - 1) // align) * align for am in actual_ms]
    m = sum(aligned_ms)

    a2 = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    codes = torch.randint(
        0, 16, (g, n, k), device=device, dtype=torch.int8
    ).to(torch.int32)
    scale = (
        torch.randn((g, n, k // _W4A16_GROUP), device=device) * 0.02
    ).to(torch.bfloat16)
    m_indices = torch.empty((m,), device=device, dtype=torch.int32)
    ref = torch.zeros((m, n), device=device, dtype=torch.bfloat16)

    start = 0
    bounds = []
    for j, (actual, aligned) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end, aligned_end = start + actual, start + aligned
        m_indices[start:actual_end] = j
        m_indices[actual_end:aligned_end] = -1
        a2[actual_end:aligned_end] = 0
        b_deq = _dequant_reference(codes[j], scale[j]).reshape(n, k)
        ref[start:actual_end] = (a2[start:actual_end].float() @ b_deq.t()).to(
            torch.bfloat16
        )
        bounds.append((j, start, actual_end))
        start = aligned_end
    return a2, codes, scale, m_indices, ref, bounds


# ---------------------------------------------------------------------------
# Non-GPU unit tests: the heuristic port.
# ---------------------------------------------------------------------------


def _expected_masked_config(num_sms: int = 132):
    """Hand-derived rows from DeepGEMM sm90.hpp (H200, num_sms=132)."""
    return {
        # (k, expected_m, n, G) -> (BM, BN, BK, stages)
        (7168, 16, 4096, 24): (24, 256, 128, 4),
        (2048, 16, 7168, 12): (24, 256, 128, 4),
        (7168, 64, 4096, 48): (88, 256, 128, 4),
        # down keeps BN=256 at bm>=72 too: the two-wave guard is the only BN
        # rule, on both projections.
        (2048, 64, 7168, 48): (80, 256, 128, 4),
        (2048, 64, 7168, 24): (80, 256, 128, 4),
        (2048, 32, 7168, 12): (40, 256, 128, 4),
        # EP8 (G=48), the width docs/performance.md (Shapes) derives.  Its N-tile count is
        # past the 2-wave mark at both projections, so BN=256 holds in every
        # bracket -- EP8 exercises the wave guard's saturated side, while
        # EP32 (G=12, gate_up) is the only width that falls under 2 waves.
        (7168, 8, 4096, 48): (16, 256, 128, 4),
        (7168, 32, 4096, 48): (48, 256, 128, 4),
        (2048, 8, 7168, 48): (16, 256, 128, 4),
        (2048, 32, 7168, 48): (40, 256, 128, 4),
        # EP32 gate_up: 192 N-tiles < 2 waves, the one bracket that drops to
        # BN=128 on wave quantization rather than on the register ceiling.
        (7168, 8, 4096, 12): (16, 128, 128, 4),
        (7168, 32, 4096, 12): (48, 128, 128, 4),
    }


def test_sm90_w4a16_masked_heuristic_table() -> None:
    h = _heuristics()
    for (k, em, n, g), (bm, bn, bk, stages) in _expected_masked_config().items():
        config = h.select_w4a16_config(
            h.W4A16GemmDesc("masked", 256, n, k, g, 132, expected_m=em)
        )
        assert (
            config.layout.block_m,
            config.layout.block_n,
            config.layout.block_k,
        ) == (bm, bn, bk), (k, em, n, g)
        assert config.pipeline.num_stages == stages, (k, em, n, g)
        assert config.pipeline.smem_size <= 232448
        assert config.launch.num_threads == 384


def test_w4a16_masked_cta_residency() -> None:
    """Two resident CTAs per SM for the small masked tiles, one elsewhere.

    Measured on H200 (docs/performance.md): the second CTA only pays at
    ``block_m`` 16/24 once the doubled grid runs two waves, so the heuristic
    keeps one CTA below that and for every larger tile.
    """
    h = _heuristics()

    def ctas(k: int, em: int, n: int, g: int) -> tuple[int, int]:
        config = h.select_w4a16_config(
            h.W4A16GemmDesc("masked", 256, n, k, g, 132, expected_m=em)
        )
        return config.launch.num_ctas_per_sm, config.launch.grid_dim

    # EP8: both projections are long enough at block_m 16/24.
    assert ctas(7168, 8, 4096, 48) == (2, 264)
    assert ctas(2048, 16, 7168, 48) == (2, 264)
    # EP32 gate_up: 384 tiles at G=24 is below the two-wave mark of the
    # doubled grid; G=12 is one wave either way.
    assert ctas(7168, 8, 4096, 24) == (1, 132)
    assert ctas(7168, 16, 4096, 12) == (1, 132)
    # block_m >= 40: the halved ring drops under three stages.
    assert ctas(7168, 32, 4096, 48) == (1, 132)
    assert ctas(2048, 64, 7168, 48) == (1, 132)
    # Contiguous never takes the second CTA.
    contiguous = h.select_w4a16_config(
        h.W4A16GemmDesc("contiguous", 12288, 4096, 7168, 48, 132)
    )
    assert contiguous.launch.num_ctas_per_sm == 1
    assert contiguous.launch.grid_dim == 132


def test_sm90_w4a16_contiguous_heuristic_table() -> None:
    h = _heuristics()

    def tile(m: int, n: int, k: int, g: int):
        c = h.select_w4a16_config(h.W4A16GemmDesc("contiguous", m, n, k, g, 132))
        assert c.pipeline.smem_size <= 232448
        return (c.layout.block_m, c.layout.block_n, c.layout.block_k,
                c.layout.cluster_size, c.pipeline.num_stages)

    # gate/up at every size and down from 256 rows/E: the wide 128x256x128
    # tile, no multicast.
    assert tile(48 * 128, 4096, 7168, 48) == (128, 256, 128, 1, 3)
    assert tile(48 * 512, 7168, 2048, 48) == (128, 256, 128, 1, 3)
    assert tile(24 * 256, 7168, 2048, 24) == (128, 256, 128, 1, 3)
    # Short-K down below 256 rows/E keeps 128x128x64.
    assert tile(48 * 128, 7168, 2048, 48) == (128, 128, 64, 2, 8)
    # A small single-group problem doubles the M/N tiles to fill the SMs.
    small = h.select_w4a16_config(
        h.W4A16GemmDesc("contiguous", 128, 4096, 7168, 1, 132)
    )
    assert (small.layout.block_m, small.layout.block_n) == (64, 64)


# ---------------------------------------------------------------------------
# GPU tests: packing and direct-API correctness.
# ---------------------------------------------------------------------------


@pytest.fixture
def hopper() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("the grouped W4A16 backend is SM90-only")
    return device


def _pack_unpacked(codes: torch.Tensor, scale: torch.Tensor):
    from chord_kernels import pack_w4a16

    return pack_w4a16(codes, scale)


@pytest.mark.gpu
@pytest.mark.parametrize("gemm_type", ["masked", "contiguous"])
def test_nvrtc_compiles_both_instantiations(
    gemm_type: str, hopper: torch.device
) -> None:
    """Compile-only guard on the NVRTC adaptation of the vendored kernel.

    The kernel reaches NVRTC without a host standard library, and CUTLASS routes
    ``<type_traits>`` through ``CUDA_STD_HEADER`` under ``__CUDACC_RTC__``, so the
    ``std::`` names it spells are supplied by
    ``GroupedNVRTCCompiler.device_prelude()`` instead. A CUTLASS bump that
    changes that routing, or a kernel change that spells a new ``std::`` symbol,
    breaks compilation for every shape at once. Failing here names the cause
    directly rather than surfacing as a wall of numerical failures.
    """
    from chord_kernels.operator.grouped.heuristics import W4A16GemmDesc
    from chord_kernels.operator.grouped.kernel import (
        GroupedNVRTCCompiler,
        GroupedW4A16Kernel,
    )
    from chord_kernels.operator.grouped.heuristics import select_w4a16_config

    prelude = GroupedNVRTCCompiler.device_prelude()
    assert "conditional_t" in prelude
    # PDL is a CUDA runtime builtin NVRTC does not declare; both call sites in
    # the kernel resolve to the prelude's inline asm.
    assert prelude.count("griddepcontrol") == 2

    desc = W4A16GemmDesc(
        gemm_type=gemm_type,
        m=128,
        n=4096,
        k=7168,
        num_groups=8,
        num_sms=torch.cuda.get_device_properties(hopper).multi_processor_count,
        expected_m=64 if gemm_type == "masked" else 0,
    )
    kernel = GroupedW4A16Kernel(desc=desc, config=select_w4a16_config(desc))
    cubin = Path(kernel.kernel_filename)
    assert cubin.is_file() and cubin.stat().st_size > 0
    # ``Lb1`` is kIsW4A16=true: confirms the W4A16 branch, not the FP8 parent.
    assert "ELb1E" in kernel.kernel_name


@pytest.mark.gpu
def test_one_packed_weight_serves_indexed_masked_and_contiguous(
    hopper: torch.device,
) -> None:
    """``pack_w4a16`` feeds the grouped kernels too: masked (BLOCK_K 128)
    and contiguous (BLOCK_K 64) read the same bytes as ``indexed`` and match the
    BF16 dequant reference, and a K the masked tile does not cover is rejected."""
    from chord_kernels import contiguous, indexed, masked, pack_w4a16
    from generators import align_routing_blocks, generate_topk_ids

    _reset_seed()
    g, n, k, rows = 8, 256, 1024, 128
    codes = torch.randint(0, 16, (g, n, k), device=hopper, dtype=torch.int8).to(torch.int32)
    scale = (torch.randn((g, n, k // _W4A16_GROUP), device=hopper) * 0.02).to(torch.bfloat16)
    weight = pack_w4a16(codes, scale)
    deq = [_dequant_reference(codes[j], scale[j]).reshape(n, k).float() for j in range(g)]

    a3 = torch.randn((g, rows, k), device=hopper, dtype=torch.bfloat16)
    masked_m = torch.tensor([rows - 7 * j for j in range(g)], dtype=torch.int32, device=hopper)
    out3 = masked(a3, weight, masked_m, 64).view(g, rows, n)
    out2 = contiguous(a3.view(g * rows, k), weight,
                      torch.arange(g, dtype=torch.int32, device=hopper).repeat_interleave(rows))
    for j in range(g):
        mm = int(masked_m[j])
        ref = (a3[j].float() @ deq[j].t()).to(torch.bfloat16)
        assert _cos_diff(out3[j, :mm], ref[:mm]) < _COSINE_LIMIT, j
        assert _cos_diff(out2.view(g, rows, n)[j], ref) < _COSINE_LIMIT, j

    topk_ids = generate_topk_ids(num_tokens=64, top_k=4, num_experts=g, device=hopper)
    sorted_ids, expert_ids, ntp = align_routing_blocks(topk_ids, block_m=16, num_experts=g)
    x = torch.randn((64, k), device=hopper, dtype=torch.bfloat16)
    out = indexed(x, weight, sorted_ids, expert_ids, ntp, 4, block_m=16)
    flat = topk_ids.reshape(-1)
    ref = torch.stack([x[i // 4].float() @ deq[int(e)].t() for i, e in enumerate(flat)])
    assert _cos_diff(out[: flat.numel()], ref.to(torch.bfloat16)) < _COSINE_LIMIT

    # The masked kernel's BLOCK_K=128 must still tile K whole.
    short_k = pack_w4a16(codes[..., :192].contiguous(), scale[..., :6].contiguous())
    with pytest.raises(ValueError, match="BLOCK_K=128"):
        masked(torch.randn((g, 16, 192), device=hopper, dtype=torch.bfloat16), short_k,
               torch.full((g,), 4, dtype=torch.int32, device=hopper), 4)


@pytest.mark.gpu
@pytest.mark.parametrize("case", MASKED_CASES, ids=lambda c: c.label)
def test_masked_w4a16_correctness(case: MaskedCase, hopper: torch.device) -> None:
    from chord_kernels.operator.grouped.api import w4a16_masked

    a3, codes, scale, masked_m, ref = make_masked_case(case, hopper)
    weight = _pack_unpacked(codes, scale)
    g, max_m, n = case.num_groups, case.max_m, case.n

    out = w4a16_masked(
        a3, weight, masked_m, case.expected_m_per_group
    )
    assert out.shape == (g * max_m, n)
    out3 = out.view(g, max_m, n)
    for j in range(g):
        mm = int(masked_m[j].item())
        if mm == 0:
            continue
        diff = _cos_diff(out3[j, :mm], ref[j, :mm])
        assert diff < _COSINE_LIMIT, (case.label, j, mm, diff)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "max_m,masked_counts",
    [(8, [8]), (24, [20]), (24, [20, 4]), (40, [36]), (40, [(i * 7) % 41 for i in range(48)])],
    ids=["max_m_8", "max_m_24", "max_m_24_mixed", "max_m_40", "max_m_40_ragged"],
)
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 2048)], ids=["gate_up", "down"])
def test_masked_partial_last_tile_stays_in_its_group(
    hopper: torch.device, n: int, k: int, max_m: int, masked_counts: list[int],
) -> None:
    """A ``max_m`` that is not a multiple of BLOCK_M (16 at ``expected_m`` 8)
    leaves each group a partial last M tile.  Its loads and stores are clipped
    at ``max_m``: every group matches the reference and repeated launches agree.
    The flat 2D maps used to let that tile read and write the next group's rows.
    """
    from chord_kernels.operator.grouped.api import w4a16_masked

    _reset_seed()
    g = 48
    counts = (masked_counts * g)[:g]
    a3 = torch.randn((g, max_m, k), device=hopper, dtype=torch.bfloat16)
    codes = torch.randint(0, 16, (g, n, k), device=hopper, dtype=torch.int8).to(torch.int32)
    scale = (torch.randn((g, n, k // _W4A16_GROUP), device=hopper) * 0.02).to(torch.bfloat16)
    weight = _pack_unpacked(codes, scale)
    masked_m = torch.tensor(counts, dtype=torch.int32, device=hopper)

    outs = []
    for _ in range(3):
        out = torch.full((g * max_m, n), float("nan"), dtype=torch.bfloat16, device=hopper)
        w4a16_masked(a3, weight, masked_m, 8, outputs=out)
        outs.append(out.view(g, max_m, n))
    torch.cuda.synchronize(hopper)
    for j, mm in enumerate(counts):
        if mm == 0:
            continue
        ref = (a3[j, :mm].float() @ _dequant_reference(codes[j], scale[j]).reshape(n, k).t().float())
        diff = _cos_diff(outs[0][j, :mm], ref.to(torch.bfloat16))
        assert diff < _COSINE_LIMIT, (j, mm, diff)
        assert all(torch.equal(o[j, :mm], outs[0][j, :mm]) for o in outs[1:]), (j, mm)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "n,k,em", [(4096, 7168, 8), (7168, 2048, 16)], ids=["gate_up", "down"]
)
def test_masked_two_cta_residency_matches_one(
    hopper: torch.device, n: int, k: int, em: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2-CTA masked path (EP8 small tiles) agrees with the 1-CTA path bit
    for bit and is repeatable.

    The residency is a launch-shape change only -- the scheduler walks the same
    block list, just over twice as many persistent CTAs -- so any difference
    would mean a race in the pipeline or the 3D masked store.  Forcing
    ``CHORD_W4A16_CTAS`` also covers the config plumbing through both JITs.
    """
    from chord_kernels.operator.grouped.api import (
        _resolve_kernel_cached,
        w4a16_masked,
    )

    _reset_seed()
    g, max_m, n_exp, k_exp = 48, 256, n, k
    weights = torch.randint(
        0, 16, (g, n_exp, k_exp), device=hopper, dtype=torch.int8
    ).to(torch.int32)
    scale = (torch.randn((g, n_exp, k_exp // _W4A16_GROUP), device=hopper) * 0.02).to(
        torch.bfloat16
    )
    weight = _pack_unpacked(weights, scale)
    masked_m = torch.tensor([em] * g, dtype=torch.int32, device=hopper)

    def run(ctas: int, a3: torch.Tensor) -> torch.Tensor:
        monkeypatch.setenv("CHORD_W4A16_CTAS", str(ctas))
        _resolve_kernel_cached.cache_clear()
        out = w4a16_masked(a3, weight, masked_m, em)
        torch.cuda.synchronize(hopper)
        return out

    a3 = torch.randn((g, max_m, k_exp), device=hopper, dtype=torch.bfloat16)
    one = run(1, a3).view(g, max_m, n_exp)
    two = run(2, a3).view(g, max_m, n_exp)
    two_again = run(2, a3).view(g, max_m, n_exp)
    monkeypatch.delenv("CHORD_W4A16_CTAS")
    _resolve_kernel_cached.cache_clear()

    # Only the first `em` rows of each group carry results; the rest of max_m
    # is padding the masked path leaves untouched.
    valid = slice(0, em)
    assert torch.equal(two[:, valid], two_again[:, valid]), (
        "the 2-CTA launch is not repeatable"
    )
    assert torch.equal(one[:, valid], two[:, valid]), (
        "the 2-CTA launch differs from the 1-CTA launch"
    )


# One case per decode bucket at a single shape: the flat/3D equivalence is a
# calling-convention property, so it needs the bucket spread rather than the
# full shape matrix.  Selected by value so extending MASKED_CASES cannot
# silently change which buckets this covers.
_FLAT_BUFFER_CASES = [
    case for case in MASKED_CASES if (case.num_groups, case.n) == (24, 4096)
]


@pytest.mark.gpu
@pytest.mark.parametrize("case", _FLAT_BUFFER_CASES, ids=lambda c: c.label)
def test_masked_w4a16_flat_input_and_output_buffer(
    case: MaskedCase, hopper: torch.device
) -> None:
    """Flat [G*max_m, K] input and a caller-provided flat output are the
    serving-facing forms; they must agree with the 3D path."""
    from chord_kernels.operator.grouped.api import w4a16_masked

    a3, codes, scale, masked_m, ref = make_masked_case(case, hopper)
    weight = _pack_unpacked(codes, scale)
    g, max_m, n = case.num_groups, case.max_m, case.n

    flat_out = torch.zeros((g * max_m, n), device=hopper, dtype=torch.bfloat16)
    returned = w4a16_masked(
        a3.view(g * max_m, case.k),
        weight,
        masked_m,
        case.expected_m_per_group,
        outputs=flat_out,
    )
    assert returned.data_ptr() == flat_out.data_ptr()
    for j in range(g):
        mm = int(masked_m[j].item())
        diff = _cos_diff(
            flat_out.view(g, max_m, n)[j, :mm], ref[j, :mm]
        )
        assert diff < _COSINE_LIMIT, (case.label, j, diff)


@pytest.mark.gpu
@pytest.mark.parametrize("case", CONTIGUOUS_CASES, ids=lambda c: c.label)
def test_contiguous_w4a16_correctness(
    case: ContiguousCase, hopper: torch.device
) -> None:
    from chord_kernels.operator.grouped.api import w4a16_contiguous

    a2, codes, scale, m_indices, ref, bounds = make_contiguous_case(case, hopper)
    weight = _pack_unpacked(codes, scale)
    out = w4a16_contiguous(a2, weight, m_indices)
    assert out.shape == (a2.size(0), case.n)
    for j, start, actual_end in bounds:
        if actual_end <= start:
            continue
        diff = _cos_diff(out[start:actual_end], ref[start:actual_end])
        assert diff < _COSINE_LIMIT, (case.label, j, diff)


# Column layout mirrors tests/test_w4a16_indexed.py: shape columns first, then
# the schedule, then timing / roofline / accuracy.  ``G`` and ``m/grp`` replace
# indexed's routing columns because the grouped convention carries its row counts
# per expert rather than through a routing table.
#
# The utilization percentage rides in the column of the wall that actually binds:
# masked decode is bandwidth bound, so it goes on GB/s; contiguous prefill is
# compute bound, so it goes on TFLOPS (matching the indexed tables).  Both
# ceilings describe the same operating point, so the number is the same either
# way -- only its placement says which resource is saturated.
#
# ``m/grp`` is the kernel-side knob (rows per expert, what picks the tile).
# Prefill adds one column restating it as the system-wide chunk size, since a
# chunked-prefill deployment is configured in tokens; decode omits it, as its
# per-GPU batch is already the ``m/grp``/top_k relation read off ``G``.
def _base_columns(token_column: str | None) -> tuple[tuple[str, int], ...]:
    return (
        ("proj", 7),
        ("n", 5),
        ("k", 5),
        ("G", 4),
        ("m/grp", 6),
        *(((token_column, 9),) if token_column else ()),
        ("routed_m", 8),
        ("blk_k", 5),
        ("us", 9),
        ("TFLOPS", 11),
        ("GB/s", 13),
        ("cos_diff", 9),
    )


def _projection(n: int, k: int) -> str:
    """Name the projection the way the indexed table and docs/performance.md (Shapes) do.

    The Kimi K2.5 shapes are gate/up ``N=4096 K=7168`` (fused w13, reads the
    model dim) and down ``N=7168 K=2048`` (reads the expert dim), so the wider
    K identifies gate/up.
    """
    return "gate_up" if k > n else "down"


def run_perf_table(
    device: torch.device,
    iterations: int = 20,
    method: str = DEFAULT_BENCH_METHOD,
) -> None:
    """Print the grouped roofline tables in the indexed table's column layout.

    Each case is checked against the same plain-PyTorch reference the
    correctness tests use and then timed, so ``cos_diff`` describes the very
    launch that produced the throughput numbers.
    """
    from chord_kernels.operator.grouped.api import (
        w4a16_contiguous,
        w4a16_masked,
    )

    def bench(fn) -> float:
        # Shared with the indexed table (tests/bench.py): the default ``triton``
        # method is ``do_bench(warmup=100, rep=1000)``, matching upstream
        # Humming's benchmarks/bench_humming.py so rows compare one-to-one.
        # ``kineto`` reports the kernel's own GPU time, excluding the launch
        # path, which is what to use when a small decode shape is dominated by
        # per-call overhead rather than by the GEMM.
        _, seconds = benchmark(
            fn, device=device, method=method, iterations=iterations
        )
        return seconds

    # Prefill first, then decode, matching the order of the indexed tables.
    contiguous_columns = throughput_columns(
        _base_columns("chunk_tok"), compute_bound=True
    )
    print_group(
        "grouped contiguous prefill (BLOCK_K=64) -- inputs [m, K], routing = "
        f"m_indices with per-expert runs on {CONTIGUOUS_ROW_ALIGNMENT}-row "
        "boundaries; m/grp = rows per expert; chunk_tok = system-wide "
        f"chunked-prefill tokens that produces it at {_ROUTED_EXPERTS} experts "
        f"/ top_k {_TOP_K} (EP-width independent)",
        contiguous_columns,
    )
    for case in CONTIGUOUS_CASES:
        a2, codes, scale, m_indices, ref, _ = make_contiguous_case(case, device)
        weight = _pack_unpacked(codes, scale)
        out = w4a16_contiguous(a2, weight, m_indices)
        # m_indices == -1 marks the padding rows the kernel does not write.
        valid = m_indices != -1
        cos = _cos_diff(out[valid], ref[valid])
        t = bench(lambda: w4a16_contiguous(a2, weight, m_indices))
        valid_m = int(valid.sum().item())
        flops = 2 * valid_m * case.n * case.k
        total_bytes = traffic_bytes(
            case.num_groups, case.n, case.k, valid_m, valid_m
        )
        roof_t, roof_g = roofline(flops, total_bytes, device)
        tflops_cell, gbps_cell = format_throughput(
            flops / t / 1e12,
            roof_t,
            total_bytes / t / 1e9,
            roof_g,
            compute_bound=True,
        )
        print_row((
            _projection(case.n, case.k),
            str(case.n),
            str(case.k),
            str(case.num_groups),
            str(case.expected_m_per_group),
            str(prefill_chunk_tokens(case)),
            str(valid_m),
            "64",
            f"{t * 1e6:.1f}",
            tflops_cell,
            gbps_cell,
            f"{cos:.2e}",
        ), contiguous_columns)

    masked_columns = throughput_columns(_base_columns(None), compute_bound=False)
    print_group(
        "grouped masked decode (BLOCK_K=128) -- inputs [G, max_m, K], routing = "
        "per-expert valid row counts; m/grp = expected tokens per expert",
        masked_columns,
    )
    for case in MASKED_CASES:
        a3, codes, scale, masked_m, ref = make_masked_case(case, device)
        weight = _pack_unpacked(codes, scale)
        out = w4a16_masked(a3, weight, masked_m, case.expected_m_per_group)
        # Only the per-expert valid prefixes carry results; the padding rows up
        # to max_m are untouched, so compare the same slices the correctness
        # test does and report the worst expert.
        out3 = out.view(case.num_groups, case.max_m, case.n)
        cos = max(
            _cos_diff(out3[j, :mm], ref[j, :mm])
            for j in range(case.num_groups)
            if (mm := int(masked_m[j].item())) > 0
        )
        t = bench(
            lambda: w4a16_masked(a3, weight, masked_m, case.expected_m_per_group)
        )
        valid_m = int(masked_m.sum().item())
        flops = 2 * valid_m * case.n * case.k
        total_bytes = traffic_bytes(
            case.num_groups, case.n, case.k, valid_m, valid_m
        )
        roof_t, roof_g = roofline(flops, total_bytes, device)
        tflops_cell, gbps_cell = format_throughput(
            flops / t / 1e12,
            roof_t,
            total_bytes / t / 1e9,
            roof_g,
            compute_bound=False,
        )
        print_row((
            _projection(case.n, case.k),
            str(case.n),
            str(case.k),
            str(case.num_groups),
            str(case.expected_m_per_group),
            str(valid_m),
            "128",
            f"{t * 1e6:.1f}",
            tflops_cell,
            gbps_cell,
            f"{cos:.2e}",
        ), masked_columns)


@pytest.mark.gpu
@pytest.mark.perf
def test_w4a16_grouped_perf(hopper: torch.device) -> None:
    run_perf_table(hopper)


def test_pure_python_heuristic_rejects_bad_desc() -> None:
    h = _heuristics()
    with pytest.raises(ValueError):
        h.W4A16GemmDesc("masked", 256, 4096, 7168, 24, 131, expected_m=16)  # odd sms
    with pytest.raises(ValueError):
        h.W4A16GemmDesc("masked", 256, 4104, 7168, 24, 132, expected_m=16)  # n%16


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="timed iterations per case (default: 20; unused by 'triton')",
    )
    parser.add_argument(
        "--bench-method",
        choices=BENCH_METHODS,
        default=DEFAULT_BENCH_METHOD,
        help=(
            "timing method (default: %(default)s, matching upstream Humming's "
            "do_bench; 'kineto' excludes launch overhead)"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the grouped W4A16 benchmark requires a CUDA GPU")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"--device must select CUDA, got {device}")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (9, 0):
        raise RuntimeError(
            f"the grouped W4A16 backend is SM90-only, got SM{major}{minor}"
        )

    print(
        f"{torch.cuda.get_device_name(device)} (SM{major}{minor}) -- grouped "
        "W4A16, BF16 activation x INT4 weight, group-32 scale -- timing: "
        f"{args.bench_method}"
    )
    run_perf_table(
        device, iterations=args.iterations, method=args.bench_method
    )
    print(flush=True)


if __name__ == "__main__":
    main()
