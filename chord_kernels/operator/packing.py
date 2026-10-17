# Derived from inclusionAI/humming; modified for chord_kernels.

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import torch

from chord_kernels.operator import cpp_jit
from chord_kernels.operator.kernel.repack_weight import RepackWeightKernel

SUPPORTED_INDEXED_COMPUTE_CAPABILITIES = frozenset({(9, 0), (10, 0), (10, 3)})


def _validate_supported_architecture(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability not in SUPPORTED_INDEXED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "indexed W4A16 requires an SM90 Hopper or SM100/SM103 "
            "Blackwell GPU, got compute capability "
            f"{capability[0]}.{capability[1]} on {device}"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PreparedWeight:
    """Packed weight ``[E, K/16, 2N]`` and scale ``[E, K/32, N]``; every indexed
    kernel (MMA, swap-AB MMA, WGMMA) reads the same bytes."""

    packed: torch.Tensor
    scale: torch.Tensor
    n: int
    k: int
    num_experts: int

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.packed, self.scale

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.tensors)


def _pack_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    # The GEMM loader consumes [E, K/32, N] with this 64-value lane permutation.
    perm = [0, 8, 16, 24, 32, 40, 48, 56]
    perm = [value + offset for offset in range(8) for value in perm]
    perm_tensor = torch.tensor(perm, dtype=torch.int64, device=weight_scale.device)
    packed = weight_scale.transpose(-1, -2).contiguous()
    shape = packed.shape
    return packed.view(-1, 64)[:, perm_tensor].contiguous().view(shape)


def pack_w4a16_indexed(
    weight_uint4: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    packed: bool = False,
) -> PreparedWeight:
    """Pack INT4 weights and their BF16 group-32 scales for ``indexed``.

    ``weight_uint4`` is int32 ``[E, N, K]`` holding unsigned codes in ``[0, 15]``
    (decoded as ``code - 8``), or with ``packed=True`` the checkpoint form
    ``[E, N, K/8]``, eight little-endian nibbles per word.  ``weight_scale`` is
    BF16 ``[E, N, K/32]``.
    """
    if not isinstance(packed, bool):
        raise TypeError(f"packed must be a bool, got {type(packed).__name__}")
    if not isinstance(weight_uint4, torch.Tensor):
        raise TypeError(
            f"weight_uint4 must be a torch.Tensor, got {type(weight_uint4).__name__}"
        )
    if not isinstance(weight_scale, torch.Tensor):
        raise TypeError(
            f"weight_scale must be a torch.Tensor, got {type(weight_scale).__name__}"
        )
    if weight_uint4.ndim != 3:
        expected = "[E, N, K/8]" if packed else "[E, N, K]"
        raise ValueError(
            f"weight_uint4 must have shape {expected}, got {weight_uint4.shape}"
        )
    if weight_uint4.dtype != torch.int32:
        raise TypeError(f"weight_uint4 must be torch.int32, got {weight_uint4.dtype}")
    if not weight_uint4.is_cuda or not weight_uint4.is_contiguous():
        raise ValueError("weight_uint4 must be a contiguous CUDA tensor")
    _validate_supported_architecture(weight_uint4.device)

    num_experts, shape_n, stored_shape_k = weight_uint4.shape
    if num_experts == 0 or shape_n == 0 or stored_shape_k == 0:
        raise ValueError(
            f"weight_uint4 dimensions must be non-zero, got {tuple(weight_uint4.shape)}"
        )
    shape_k = stored_shape_k * 8 if packed else stored_shape_k
    expected_scale_shape = (num_experts, shape_n, shape_k // 32)
    if shape_n % 128 != 0:
        raise ValueError(f"shape_n must be divisible by 128, got {shape_n}")
    if shape_k % 64 != 0:
        raise ValueError(f"shape_k must be divisible by 64, got {shape_k}")
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale must have shape {expected_scale_shape}, got {tuple(weight_scale.shape)}"
        )
    if weight_scale.dtype != torch.bfloat16:
        raise TypeError(
            f"weight_scale must be torch.bfloat16, got {weight_scale.dtype}"
        )
    if weight_scale.device != weight_uint4.device or not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous and on the weight device")

    if not packed:
        minimum, maximum = torch.aminmax(weight_uint4)
        if minimum.item() < 0 or maximum.item() > 15:
            raise ValueError(
                "weight_uint4 values must be in [0, 15], got "
                f"[{minimum.item()}, {maximum.item()}]"
            )

    packed_weight = torch.empty(
        (num_experts, shape_k // 16, shape_n * 2),
        dtype=torch.int32,
        device=weight_uint4.device,
    )
    with torch.cuda.device(weight_uint4.device):
        if cpp_jit.enabled(weight_uint4.device):
            # CHORD_CPP_JIT=1: the C++ JIT (cpp_jit.py).
            cpp_jit.repack_w4a16(weight_uint4, packed_weight, packed=packed)
        else:
            kernel = RepackWeightKernel(is_weight_packed=packed)
            kernel(inputs=weight_uint4, outputs=packed_weight)

    return PreparedWeight(
        packed=packed_weight,
        scale=_pack_weight_scale(weight_scale),
        n=shape_n,
        k=shape_k,
        num_experts=num_experts,
    )


__all__ = ["PreparedWeight", "pack_w4a16_indexed"]
