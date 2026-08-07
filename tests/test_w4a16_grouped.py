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
import json
import os
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
# EP16/EP32 groups, both projections, small/mid/large decode buckets.
MASKED_CASES = [
    MaskedCase(g, 256, em, n, k)
    for g in (24, 12)
    for (n, k) in ((4096, 7168), (7168, 2048))
    for em in (8, 16, 64)
]
CONTIGUOUS_CASES = [
    ContiguousCase(g, mpg, n, k)
    for g in (48, 24)
    for (n, k) in ((4096, 7168), (7168, 2048))
    for mpg in (37, 128, 1024)
]


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
    contiguous layout: per-expert runs padded to a 128-row boundary, padding
    rows zeroed with m_indices == -1."""
    _reset_seed()
    g, mpg, n, k = case.num_groups, case.expected_m_per_group, case.n, case.k
    actual_ms = [max(1, int(mpg * random.uniform(0.7, 1.3))) for _ in range(g)]
    aligned_ms = [((am + 127) // 128) * 128 for am in actual_ms]
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


def _pack_checkpoint_int32(codes: torch.Tensor) -> torch.Tensor:
    """[G, N, K] codes [0,15] -> INT32 [G, N, K/8], 8 little-endian nibbles
    per word (the framework checkpoint contract)."""
    g, n, k = codes.shape
    nib = codes.to(torch.int64).reshape(g, n, k // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=codes.device) * 4
    return (nib << shifts).sum(-1).to(torch.int32)


# ---------------------------------------------------------------------------
# Non-GPU unit tests: the heuristic port and the profile registry invariants.
# ---------------------------------------------------------------------------


def _expected_masked_config(num_sms: int = 132):
    """Hand-derived rows from DeepGEMM sm90.hpp (H200, num_sms=132)."""
    return {
        # (k, expected_m, n, G) -> (BM, BN, BK, stages)
        (7168, 16, 4096, 24): (24, 256, 128, 4),
        (2048, 16, 7168, 12): (24, 256, 128, 4),
        (7168, 64, 4096, 48): (88, 256, 128, 4),
        (2048, 64, 7168, 48): (80, 128, 128, 6),
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


def test_sm90_w4a16_contiguous_heuristic_table() -> None:
    h = _heuristics()
    config = h.select_w4a16_config(
        h.W4A16GemmDesc("contiguous", 12288, 4096, 7168, 48, 132)
    )
    assert (
        config.layout.block_m,
        config.layout.block_n,
        config.layout.block_k,
    ) == (128, 128, 64)
    assert config.pipeline.num_stages == 8
    assert config.pipeline.smem_size <= 232448
    # A small single-group problem doubles the M/N tiles to fill the SMs.
    small = h.select_w4a16_config(
        h.W4A16GemmDesc("contiguous", 128, 4096, 7168, 1, 132)
    )
    assert (small.layout.block_m, small.layout.block_n) == (64, 64)


def test_grouped_profiles_are_load_time_consistent() -> None:
    from chord_kernels.operator.layer import (
        H200_GROUPED_DECODE,
        H200_GROUPED_PREFILL,
        IndexedLayerProfile,
    )

    assert H200_GROUPED_DECODE.backend == "grouped_masked"
    assert H200_GROUPED_DECODE.mode == "decode"
    assert H200_GROUPED_PREFILL.backend == "grouped_contiguous"
    assert H200_GROUPED_PREFILL.mode == "prefill"
    for profile in (H200_GROUPED_DECODE, H200_GROUPED_PREFILL):
        assert profile.is_grouped
        assert profile.layout == "wgmma" and not profile.swap_ab
    with pytest.raises(ValueError):
        IndexedLayerProfile(
            name="bad_mode",
            device_major=9,
            mode="prefill",
            expert_parallel_size=8,
            layout="wgmma",
            swap_ab=False,
            block_m=8,
            compute_capabilities=((9, 0),),
            backend="grouped_masked",
        )


def test_select_indexed_profile_honors_grouped_switch(monkeypatch) -> None:
    from chord_kernels.operator.layer import select_indexed_profile

    monkeypatch.setenv("CHORD_USE_GROUPED", "1")
    masked = select_indexed_profile("h200_grouped_decode")
    contiguous = select_indexed_profile("h200_grouped_prefill")
    assert masked.backend == "grouped_masked"
    assert contiguous.backend == "grouped_contiguous"
    for ep in (8, 16, 32):
        select_indexed_profile("h200_grouped_decode", expert_parallel_size=ep)
    monkeypatch.delenv("CHORD_USE_GROUPED")
    with pytest.raises(ValueError):
        select_indexed_profile("h200_grouped_decode", mode="prefill")
    with pytest.raises(ValueError):
        # The indexed profiles remain EP8-only.
        select_indexed_profile("h200_decode_ep8", expert_parallel_size=16)


def test_grouped_env_validation(monkeypatch) -> None:
    from chord_kernels.operator.layer import use_grouped_from_env

    monkeypatch.delenv("CHORD_USE_GROUPED", raising=False)
    assert not use_grouped_from_env()
    monkeypatch.setenv("CHORD_USE_GROUPED", "1")
    assert use_grouped_from_env()
    monkeypatch.setenv("CHORD_USE_GROUPED", "0")
    assert not use_grouped_from_env()
    monkeypatch.setenv("CHORD_USE_GROUPED", "yes")
    with pytest.raises(ValueError):
        use_grouped_from_env()


# ---------------------------------------------------------------------------
# GPU tests: packing, direct-API correctness, layer dispatch.
# ---------------------------------------------------------------------------


@pytest.fixture
def hopper() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("the grouped W4A16 backend is SM90-only")
    return device


def _pack_unpacked(codes: torch.Tensor, scale: torch.Tensor, mode: str):
    from chord_kernels.operator.grouped.packing import pack_w4a16_grouped

    return pack_w4a16_grouped(codes, scale, mode, packed=False)


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
def test_pack_parity_packed_vs_unpacked(hopper: torch.device) -> None:
    """The checkpoint-packed [G,N,K/8] and unpacked [G,N,K] paths must produce
    bit-identical grouped buffers."""
    from chord_kernels.operator.grouped.packing import pack_w4a16_grouped

    _reset_seed()
    g, n, k = 4, 256, 512
    codes = torch.randint(0, 16, (g, n, k), device=hopper, dtype=torch.int8).to(
        torch.int32
    )
    scale = (torch.randn((g, n, k // 32), device=hopper) * 0.02).to(torch.bfloat16)
    for mode in ("masked", "contiguous"):
        unpacked = pack_w4a16_grouped(codes, scale, mode, packed=False)
        packed = pack_w4a16_grouped(
            _pack_checkpoint_int32(codes), scale, mode, packed=True
        )
        assert torch.equal(
            unpacked.packed.view(torch.uint8), packed.packed.view(torch.uint8)
        )
        assert torch.equal(unpacked.scale, packed.scale)


@pytest.mark.gpu
def test_pack_layout_shape_and_mode_guard(hopper: torch.device) -> None:
    from chord_kernels.operator.grouped.api import (
        w4a16_contiguous,
        w4a16_masked,
    )
    from chord_kernels.operator.grouped.packing import pack_w4a16_grouped

    _reset_seed()
    g, n, k = 2, 128, 256
    codes = torch.randint(0, 16, (g, n, k), device=hopper, dtype=torch.int8).to(
        torch.int32
    )
    scale = (torch.randn((g, n, k // 32), device=hopper) * 0.02).to(torch.bfloat16)
    masked_weight = pack_w4a16_grouped(codes, scale, "masked", packed=False)
    assert masked_weight.packed.shape == (g, n, k // 2)
    assert masked_weight.packed.dtype == torch.float8_e4m3fn
    assert masked_weight.scale.shape == (g, k // 32, n)
    assert masked_weight.scale.is_contiguous()
    assert masked_weight.scale.dtype == torch.bfloat16

    # A contiguous-packed weight must never reach the masked kernel: that is
    # the classic silent-wrong-answer the mode guard exists for.
    contiguous_weight = pack_w4a16_grouped(codes, scale, "contiguous", packed=False)
    a3 = torch.randn((g, 8, k), device=hopper, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="repack"):
        w4a16_masked(a3, contiguous_weight, torch.full((g,), 4, dtype=torch.int32, device=hopper), 4)
    with pytest.raises(ValueError, match="repack"):
        w4a16_contiguous(
            torch.randn((128, k), device=hopper, dtype=torch.bfloat16),
            masked_weight,
            torch.zeros((128,), dtype=torch.int32, device=hopper),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("case", MASKED_CASES, ids=lambda c: c.label)
def test_masked_w4a16_correctness(case: MaskedCase, hopper: torch.device) -> None:
    from chord_kernels.operator.grouped.api import w4a16_masked

    a3, codes, scale, masked_m, ref = make_masked_case(case, hopper)
    weight = _pack_unpacked(codes, scale, "masked")
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
@pytest.mark.parametrize("case", MASKED_CASES[:3], ids=lambda c: c.label)
def test_masked_w4a16_flat_input_and_output_buffer(
    case: MaskedCase, hopper: torch.device
) -> None:
    """Flat [G*max_m, K] input and a caller-provided flat output are the
    serving-facing forms; they must agree with the 3D path."""
    from chord_kernels.operator.grouped.api import w4a16_masked

    a3, codes, scale, masked_m, ref = make_masked_case(case, hopper)
    weight = _pack_unpacked(codes, scale, "masked")
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
    weight = _pack_unpacked(codes, scale, "contiguous")
    out = w4a16_contiguous(a2, weight, m_indices)
    assert out.shape == (a2.size(0), case.n)
    for j, start, actual_end in bounds:
        if actual_end <= start:
            continue
        diff = _cos_diff(out[start:actual_end], ref[start:actual_end])
        assert diff < _COSINE_LIMIT, (case.label, j, diff)


@pytest.mark.gpu
def test_layer_masked_decode_end_to_end(hopper: torch.device) -> None:
    """IndexedW4A16Layer with the grouped decode profile: load packed
    checkpoint weights, transform to the BK128 masked layout, forward with
    the masked routing convention."""
    from chord_kernels.operator.layer import IndexedW4A16Layer

    g, n, k, max_m, em = 12, 3072, 2048, 64, 16
    _reset_seed()
    layer = IndexedW4A16Layer(
        n, k, num_experts=g, profile="h200_grouped_decode", device=hopper
    )
    codes = torch.randint(0, 16, (g, n, k), device=hopper, dtype=torch.int8).to(
        torch.int32
    )
    scale = (torch.randn((g, n, k // 32), device=hopper) * 0.02).to(torch.bfloat16)
    layer.load_weight(_pack_checkpoint_int32(codes), scale)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert tuple(layer.weight.shape) == (g, n, k // 2)
    assert tuple(layer.weight_scale.shape) == (g, k // 32, n)

    a3 = torch.randn((g, max_m, k), device=hopper, dtype=torch.bfloat16)
    masked_m = torch.empty((g,), device=hopper, dtype=torch.int32)
    for j in range(g):
        masked_m[j] = int(em * random.uniform(0.7, 1.3))
    masked_m.clamp_(1, max_m)

    out = layer.forward(
        a3.view(g * max_m, k),
        expert_layout=masked_m,
        valid_shape_m=int(masked_m.sum().item()),
        compute_config=json.dumps({"gemm_type": "grouped_masked"}),
    )
    assert out.shape == (g * max_m, n)
    out3 = out.view(g, max_m, n)
    for j in range(g):
        mm = int(masked_m[j].item())
        ref = (
            a3[j, :mm].float()
            @ _dequant_reference(codes[j], scale[j]).reshape(n, k).t()
        ).to(torch.bfloat16)
        diff = _cos_diff(out3[j, :mm], ref)
        assert diff < _COSINE_LIMIT, (j, mm, diff)

    # The mode guard: asking the masked-packed layer for a contiguous forward
    # must fail loudly instead of mis-computing.
    with pytest.raises(ValueError, match="gemm_type"):
        layer.forward(
            a3.view(g * max_m, k),
            expert_layout=masked_m,
            valid_shape_m=int(masked_m.sum().item()),
            compute_config=json.dumps({"gemm_type": "grouped_contiguous"}),
        )
    # Index-routed calls are rejected on the grouped backend.
    with pytest.raises(ValueError, match="expert_layout"):
        layer.forward(
            a3.view(g * max_m, k),
            torch.zeros((max_m,), dtype=torch.int32, device=hopper),
            torch.zeros((1,), dtype=torch.int32, device=hopper),
            torch.zeros((1,), dtype=torch.int32, device=hopper),
            1,
            valid_shape_m=int(masked_m.sum().item()),
        )


@pytest.mark.gpu
def test_layer_contiguous_prefill_end_to_end(hopper: torch.device) -> None:
    from chord_kernels.operator.layer import IndexedW4A16Layer

    g, n, k, mpg = 8, 2048, 1408, 100
    _reset_seed()
    case = ContiguousCase(g, mpg, n, k)
    a2, codes, scale, m_indices, ref, bounds = make_contiguous_case(case, hopper)

    layer = IndexedW4A16Layer(
        n, k, num_experts=g, profile="h200_grouped_prefill", device=hopper
    )
    layer.load_weight(codes, scale, packed=False)
    out = layer.forward(
        a2,
        m_indices=m_indices,
        compute_config={"gemm_type": "grouped_contiguous"},
    )
    assert out.shape == (a2.size(0), n)
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
_BASE_COLUMNS: tuple[tuple[str, int], ...] = (
    ("proj", 7),
    ("n", 5),
    ("k", 5),
    ("G", 4),
    ("m/grp", 6),
    ("routed_m", 8),
    ("blk_k", 5),
    ("us", 9),
    ("TFLOPS", 11),
    ("GB/s", 13),
    ("cos_diff", 9),
)


def _projection(n: int, k: int) -> str:
    """Name the projection the way the indexed table and docs/shapes.md do.

    The Kimi K2.5 shapes are gate/up ``N=4096 K=7168`` (fused w13, reads the
    model dim) and down ``N=7168 K=2048`` (reads the expert dim), so the wider
    K identifies gate/up.
    """
    return "gate_up" if k > n else "down"


def run_perf_table(device: torch.device, iterations: int = 20) -> None:
    """Print the grouped roofline tables in the indexed table's column layout.

    Each case is checked against the same plain-PyTorch reference the
    correctness tests use and then timed, so ``cos_diff`` describes the very
    launch that produced the throughput numbers.
    """
    from chord_kernels.operator.grouped.api import (
        w4a16_contiguous,
        w4a16_masked,
    )

    def bench(fn, iters: int = iterations) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters / 1e3  # seconds

    # Prefill first, then decode, matching the order of the indexed tables.
    contiguous_columns = throughput_columns(_BASE_COLUMNS, compute_bound=True)
    print_group(
        "grouped contiguous prefill (BLOCK_K=64) -- inputs [m, K], routing = "
        "m_indices with per-expert runs padded to 128 rows; m/grp = expected "
        "rows per expert",
        contiguous_columns,
    )
    for case in CONTIGUOUS_CASES:
        a2, codes, scale, m_indices, ref, _ = make_contiguous_case(case, device)
        weight = _pack_unpacked(codes, scale, "contiguous")
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
            str(valid_m),
            "64",
            f"{t * 1e6:.1f}",
            tflops_cell,
            gbps_cell,
            f"{cos:.2e}",
        ), contiguous_columns)

    masked_columns = throughput_columns(_BASE_COLUMNS, compute_bound=False)
    print_group(
        "grouped masked decode (BLOCK_K=128) -- inputs [G, max_m, K], routing = "
        "per-expert valid row counts; m/grp = expected tokens per expert",
        masked_columns,
    )
    for case in MASKED_CASES:
        a3, codes, scale, masked_m, ref = make_masked_case(case, device)
        weight = _pack_unpacked(codes, scale, "masked")
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
        help="timed iterations per case (default: 20)",
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
        "W4A16, BF16 activation x INT4 weight, group-32 scale"
    )
    run_perf_table(device, iterations=args.iterations)
    print(flush=True)


if __name__ == "__main__":
    main()
