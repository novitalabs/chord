# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Literal

import torch

from chord_kernels.operator.kernel.repack_weight import RepackWeightKernel

# The operator has only two physical layouts.
WeightLayout = Literal["mma", "wgmma"]

SUPPORTED_INDEXED_COMPUTE_CAPABILITIES = frozenset(
    {
        (9, 0),
        (10, 0),
        (10, 3),
    }
)


def _validate_layout(layout: str) -> WeightLayout:
    if not isinstance(layout, str):
        raise TypeError(f"layout must be a string, got {type(layout).__name__}")
    if layout not in ("mma", "wgmma"):
        raise ValueError(f"layout must be 'mma' or 'wgmma', got {layout!r}")
    return layout  # type: ignore[return-value]


def _validate_supported_architecture(device: torch.device) -> None:
    """Validate architectures covered by the indexed W4A16 kernel.

    The indexed kernel is used on Hopper SM90 and Blackwell SM100/SM103.  Do
    not accept arbitrary 10.x or newer architectures: the generated code and
    layout contracts are covered only for the capabilities listed above.
    """
    capability = torch.cuda.get_device_capability(device)
    if capability not in SUPPORTED_INDEXED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "indexed W4A16 requires an SM90 Hopper or SM100/SM103 "
            "Blackwell GPU, got compute capability "
            f"{capability[0]}.{capability[1]} on {device}"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PreparedWeight:
    """Packed tensors plus the physical layout required by the W4A16 kernel."""

    packed: torch.Tensor
    scale: torch.Tensor
    layout: WeightLayout
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


def pack_w4a16(
    weight_uint4: torch.Tensor,
    weight_scale: torch.Tensor,
    layout: WeightLayout = "mma",
    *,
    packed: bool = False,
) -> PreparedWeight:
    """Pack quantized W4A16 weights for the indexed kernel.

    By default, ``weight_uint4`` is an unpacked int32 CUDA tensor shaped
    ``[E, N, K]`` whose values are INT4 weights held as unsigned codes in
    ``[0, 15]``. With
    ``packed=True``, it is the common checkpoint representation ``[E, N, K/8]``
    with eight little-endian nibbles in each int32 word. The packed path feeds
    those words directly to the retained Humming repack kernel and avoids
    materializing an
    eight-times-larger intermediate tensor. With no explicit zero point, the
    GEMM interprets each code as ``code - 8``.

    ``weight_scale`` is BF16 ``[E, N, K/32]``. The returned object carries an
    int32 kernel layout ``[E, K/16, 2*N]``, a BF16 scale layout
    ``[E, K/32, N]``, and the physical MMA layout metadata required to prevent
    incompatible launches.

    MMA and WGMMA weights use different mini-block layouts and are not
    interchangeable. The indexed swap-AB API uses ``layout="mma"``.
    """
    layout = _validate_layout(layout)
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
        kernel = RepackWeightKernel(
            is_weight_packed=packed,
            use_wgmma=layout == "wgmma",
        )
        kernel(inputs=weight_uint4, outputs=packed_weight)

    return PreparedWeight(
        packed=packed_weight,
        scale=_pack_weight_scale(weight_scale),
        layout=layout,
        n=shape_n,
        k=shape_k,
        num_experts=num_experts,
    )


__all__ = [
    "WeightLayout",
    "PreparedWeight",
    "pack_w4a16",
]
