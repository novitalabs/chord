# Derived from deepseek-ai/DeepGEMM; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.
"""Operator entry points for the grouped W4A16 backend.

These mirror the two DeepGEMM dispatch paths the chord layer uses
(``sm90_m_grouped_w4a16_gemm_nt_masked`` / ``..._contiguous``), routed through
chord's own NVRTC JIT and cubin launcher instead of the upstream
``deep_gemm`` package.  ``expected_m`` (masked) feeds the launch heuristics
only; ``masked_m`` carries the authoritative per-group row counts.
"""

from __future__ import annotations

import torch

from chord_kernels.operator import ops
from chord_kernels.operator.grouped.heuristics import (
    W4A16GemmDesc,
    select_w4a16_config,
)
from chord_kernels.operator.grouped.kernel import GroupedW4A16Kernel
from chord_kernels.operator.grouped.packing import GroupedPreparedWeight


def _check_weight(
    weight: GroupedPreparedWeight, mode: str, device: torch.device
) -> None:
    if not isinstance(weight, GroupedPreparedWeight):
        raise TypeError(
            "weight must be a GroupedPreparedWeight from pack_w4a16_grouped()"
        )
    if weight.mode != mode:
        # The reorder perm width (= BLOCK_K) is baked into the packed buffer;
        # running the other mode's kernel is a silent wrong answer.
        raise ValueError(
            f"weight was packed for mode={weight.mode!r} (BLOCK_K={weight.block_k}) "
            f"but the {mode!r} kernel requires BLOCK_K="
            f"{128 if mode == 'masked' else 64}; repack the weight"
        )
    packed, scale = weight.packed, weight.scale
    if packed.dtype != torch.float8_e4m3fn or scale.dtype != torch.bfloat16:
        raise TypeError(
            f"packed weight must be float8_e4m3fn and scale bfloat16, got "
            f"{packed.dtype}/{scale.dtype}"
        )
    if (
        tuple(packed.shape) != (weight.num_experts, weight.n, weight.k // 2)
        or tuple(scale.shape)
        != (weight.num_experts, weight.k // 32, weight.n)
    ):
        raise ValueError(
            "packed tensors disagree with the recorded metadata: "
            f"packed {tuple(packed.shape)} vs "
            f"{(weight.num_experts, weight.n, weight.k // 2)}, scale "
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


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _resolve_kernel(desc: W4A16GemmDesc) -> GroupedW4A16Kernel:
    config = select_w4a16_config(desc)
    kernel = GroupedW4A16Kernel(desc=desc, config=config)
    kernel.load_cubin()
    return kernel


def w4a16_masked(
    inputs: torch.Tensor,
    weight: GroupedPreparedWeight,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    outputs: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run the masked (decode) grouped W4A16 GEMM.

    ``inputs`` is the flat humming-convention activation ``[G*max_m, K]`` or
    the grouped view ``[G, max_m, K]``; the return is flat ``[G*max_m, N]``.
    ``masked_m`` is the per-expert valid token count ``[G] int32``.
    ``expected_m`` is the nominal per-group token count used by the launch
    heuristics (block-M/N sizing); it does not bound the kernel's reads.

    Rows beyond a group's ``masked_m[g]`` are not written by the kernel; when
    ``outputs`` is omitted they are left UNINITIALIZED (matching upstream —
    a consumer must gather valid rows by the counts).
    """
    _check_weight(weight, "masked", inputs.device)
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
    weight: GroupedPreparedWeight,
    m_indices: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run the contiguous (prefill) grouped W4A16 GEMM.

    ``inputs`` is the grouped-native contiguous layout ``[m, K]``: every
    expert's rows are packed consecutively and padded to a 128-row boundary
    (padding rows zeroed), and ``m_indices`` ``[m] int32`` maps each row to its
    expert id (-1 for padding rows).  Returns ``[m, N]``.
    """
    _check_weight(weight, "contiguous", inputs.device)
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
