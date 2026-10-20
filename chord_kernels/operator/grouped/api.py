# Derived from deepseek-ai/DeepGEMM; modified for chord_kernels.
"""Grouped W4A16 entry points: DeepGEMM's masked/contiguous paths on chord's JIT."""

from __future__ import annotations

import functools

import torch

from chord_kernels.operator import cpp_jit, ops
from chord_kernels.operator.grouped.heuristics import (
    W4A16GemmDesc,
    _env_int,
    select_w4a16_config,
)
from chord_kernels.operator.grouped.kernel import GroupedW4A16Kernel
from chord_kernels.operator.packing import PreparedWeight

# BLOCK_K the heuristic picks per kernel; K must cover whole tiles.
_MASKED_BLOCK_K = 128
_CONTIGUOUS_BLOCK_K = 64


def _check_weight(
    weight: PreparedWeight, mode: str, block_k: int, device: torch.device
) -> None:
    if not isinstance(weight, PreparedWeight):
        raise TypeError("weight must be a PreparedWeight from pack_w4a16()")
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(
            f"the {mode!r} W4A16 kernel requires an SM90 Hopper GPU, got compute "
            f"capability {capability[0]}.{capability[1]} on {device}"
        )
    if weight.k % block_k:
        raise ValueError(
            f"the {mode!r} kernel uses BLOCK_K={block_k}, which does not divide "
            f"the weight's K={weight.k}"
        )
    packed, scale = weight.packed, weight.scale
    if packed.dtype != torch.int32 or scale.dtype != torch.bfloat16:
        raise TypeError(
            f"packed weight must be int32 and scale bfloat16, got "
            f"{packed.dtype}/{scale.dtype}"
        )
    if (
        tuple(packed.shape) != (weight.num_experts, weight.k // 16, weight.n * 2)
        or tuple(scale.shape)
        != (weight.num_experts, weight.k // 32, weight.n)
    ):
        raise ValueError(
            "packed tensors disagree with the recorded metadata: "
            f"packed {tuple(packed.shape)} vs "
            f"{(weight.num_experts, weight.k // 16, weight.n * 2)}, scale "
            f"{tuple(scale.shape)} vs "
            f"{(weight.num_experts, weight.k // 32, weight.n)}"
        )
    for name, tensor in (("packed weight", packed), ("scale", scale)):
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on {device}")


def _check_activation(inputs: torch.Tensor, shape_k: int) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"inputs must be a torch.Tensor, got {type(inputs).__name__}")
    if inputs.dtype != torch.bfloat16:
        raise TypeError(f"inputs must be bfloat16, got {inputs.dtype}")
    if not inputs.is_cuda or not inputs.is_contiguous():
        raise ValueError("inputs must be a contiguous CUDA tensor")
    if inputs.size(-1) != shape_k:
        raise ValueError(f"inputs K={inputs.size(-1)} != weight K={shape_k}")
    if inputs.size(-1) == 0 or inputs.numel() == 0:
        raise ValueError("inputs must be non-empty")


@functools.lru_cache(maxsize=8)
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _env_tuning_key() -> tuple[int | None, ...]:
    """The ``CHORD_W4A16_*`` overrides, so a sweep changing them misses the memo."""
    return tuple(
        _env_int(name) for name in ("BM", "BN", "BK", "CM", "CN", "STAGES", "CTAS")
    )


@functools.lru_cache(maxsize=256)
def _resolve_kernel_cached(
    desc: W4A16GemmDesc, _env_key: tuple[int | None, ...]
) -> GroupedW4A16Kernel:
    config = select_w4a16_config(desc)
    kernel = GroupedW4A16Kernel(desc=desc, config=config)
    kernel.load_cubin()
    return kernel


def _resolve_kernel(desc: W4A16GemmDesc) -> GroupedW4A16Kernel:
    """Select the layout and load its cubin, memoized (~30us CPU uncached)."""
    return _resolve_kernel_cached(desc, _env_tuning_key())


@functools.lru_cache(maxsize=None)
def _cpp_kernel_id_cached(
    device_index: int, desc: W4A16GemmDesc, _env_key: tuple[int | None, ...]
) -> int:
    return cpp_jit.get_grouped_kernel(device_index, desc, select_w4a16_config(desc))


def _cpp_kernel_id(device: torch.device, desc: W4A16GemmDesc) -> int:
    """The C++ JIT kernel id for ``desc``, memoized like :func:`_resolve_kernel`."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _cpp_kernel_id_cached(index, desc, _env_tuning_key())


def w4a16_masked(
    inputs: torch.Tensor,
    weight: PreparedWeight,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    outputs: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Masked (decode) grouped GEMM: ``[G*max_m, K]`` or ``[G, max_m, K]`` in,
    flat ``[G*max_m, N]`` out.

    ``masked_m`` ``[G] int32`` is each group's valid rows; ``expected_m`` only
    sizes the tiles.  Rows past ``masked_m[g]`` are not written.  Any
    ``max_m`` works: a group's last M tile is clipped at ``max_m``.
    """
    _check_weight(weight, "masked", _MASKED_BLOCK_K, inputs.device)
    _check_activation(inputs, weight.k)
    if inputs.dim() == 2:
        if inputs.size(0) % weight.num_experts:
            raise ValueError(
                f"flat inputs rows {inputs.size(0)} are not divisible by "
                f"num_experts={weight.num_experts}"
            )
        grouped = inputs.view(
            weight.num_experts, inputs.size(0) // weight.num_experts, weight.k
        )
    elif inputs.dim() == 3:
        if inputs.size(0) != weight.num_experts:
            raise ValueError(
                f"inputs groups {inputs.size(0)} != num_experts={weight.num_experts}"
            )
        grouped = inputs
    else:
        raise ValueError(f"inputs must be 2D or 3D, got {inputs.dim()}D")

    if not isinstance(masked_m, torch.Tensor):
        raise TypeError(f"masked_m must be a torch.Tensor, got {type(masked_m).__name__}")
    if masked_m.dtype != torch.int32:
        masked_m = masked_m.to(torch.int32)
    if (
        masked_m.dim() != 1
        or masked_m.numel() != weight.num_experts
        or not masked_m.is_cuda
        or not masked_m.is_contiguous()
    ):
        raise ValueError(
            f"masked_m must be contiguous CUDA int32 [{weight.num_experts}]"
        )
    if isinstance(expected_m, bool) or not isinstance(expected_m, int):
        raise TypeError(f"expected_m must be an int, got {type(expected_m).__name__}")
    if expected_m <= 0:
        raise ValueError(f"expected_m must be positive, got {expected_m}")
    if not isinstance(enable_pdl, bool):
        raise TypeError(f"enable_pdl must be a bool, got {type(enable_pdl).__name__}")

    if outputs is not None:
        if outputs.dim() == 3:
            out3 = outputs
        else:
            out3 = outputs.view(
                weight.num_experts, grouped.size(1), weight.n
            )
    else:
        out3 = None
    desc = W4A16GemmDesc(
        gemm_type="masked",
        m=grouped.size(1),
        n=weight.n,
        k=weight.k,
        num_groups=weight.num_experts,
        num_sms=_num_sms(inputs.device),
        expected_m=expected_m,
    )
    with torch.cuda.device(inputs.device):
        if cpp_jit.enabled(inputs.device):
            # CHORD_CPP_JIT=1: the C++ JIT (cpp_jit.py).
            d = cpp_jit.grouped_masked(
                _cpp_kernel_id(inputs.device, desc), grouped, weight.packed,
                weight.scale, masked_m, out3, enable_pdl,
            )
            return d.view(weight.num_experts * grouped.size(1), weight.n)
        kernel = _resolve_kernel(desc)
        d = ops.launch_grouped_w4a16_masked(
            kernel_id=kernel.kernel_id,
            inputs=grouped,
            packed_weight=weight.packed,
            weight_scale=weight.scale,
            masked_m=masked_m,
            outputs=out3,
            enable_pdl=enable_pdl,
        )
    return d.view(weight.num_experts * grouped.size(1), weight.n)


def w4a16_contiguous(
    inputs: torch.Tensor,
    weight: PreparedWeight,
    m_indices: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Contiguous (prefill) grouped GEMM: ``[m, K]`` with each expert's rows
    padded to 128 (zeroed), ``m_indices`` ``[m] int32`` (-1 padding) -> ``[m, N]``."""
    _check_weight(weight, "contiguous", _CONTIGUOUS_BLOCK_K, inputs.device)
    _check_activation(inputs, weight.k)
    if inputs.dim() != 2:
        raise ValueError(f"inputs must be 2D [m, K], got {inputs.dim()}D")
    if not isinstance(m_indices, torch.Tensor):
        raise TypeError(
            f"m_indices must be a torch.Tensor, got {type(m_indices).__name__}"
        )
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    if (
        m_indices.dim() != 1
        or not m_indices.is_cuda
        or not m_indices.is_contiguous()
        or m_indices.numel() != inputs.size(0)
    ):
        raise ValueError(f"m_indices must be contiguous CUDA int32 [{inputs.size(0)}]")
    # Sum of per-expert 128-aligned blocks is itself a multiple of 128 (cheap
    # sanity check that the caller supplied the 128-row-padded layout).
    if inputs.size(0) % 128:
        raise ValueError(
            f"contiguous m={inputs.size(0)} must be 128-aligned (per-expert padded)"
        )
    if not isinstance(enable_pdl, bool):
        raise TypeError(f"enable_pdl must be a bool, got {type(enable_pdl).__name__}")

    desc = W4A16GemmDesc(
        gemm_type="contiguous",
        m=inputs.size(0),
        n=weight.n,
        k=weight.k,
        num_groups=weight.num_experts,
        num_sms=_num_sms(inputs.device),
    )
    with torch.cuda.device(inputs.device):
        if cpp_jit.enabled(inputs.device):
            # CHORD_CPP_JIT=1: the C++ JIT (cpp_jit.py).
            return cpp_jit.grouped_contiguous(
                _cpp_kernel_id(inputs.device, desc), inputs, weight.packed,
                weight.scale, m_indices, outputs, enable_pdl,
            )
        kernel = _resolve_kernel(desc)
        return ops.launch_grouped_w4a16_contiguous(
            kernel_id=kernel.kernel_id,
            inputs=inputs,
            packed_weight=weight.packed,
            weight_scale=weight.scale,
            m_indices=m_indices,
            outputs=outputs,
            enable_pdl=enable_pdl,
        )


__all__ = [
    "w4a16_contiguous",
    "w4a16_masked",
]
