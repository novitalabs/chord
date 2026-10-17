# Derived from deepseek-ai/DeepGEMM; modified for chord_kernels.
"""Grouped W4A16 weight packing, a bit-exact port of DeepGEMM's ``reorder_w4a16``.

Codes are bit-permuted within each ``64 x BLOCK_K`` atom for the RS-WGMMA
A-operand and pre-flipped to excess-8.  The BLOCK_K 128 order is two 64-wide
orders side by side, so one buffer serves masked (BLOCK_K 128) and contiguous
(64).  Scales become
``[G, K/32, N]``, the layout the SFA TMA descriptor reads.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterator

import torch

from chord_kernels.operator.grouped.heuristics import W4A16_N_ATOM, W4A16_SCALE_GROUP

# The reorder works in 64-wide K slabs; a BLOCK_K 128 kernel reads two side by side.
PACK_BLOCK_K = 64

# The vendored kernel is an SM90 (Hopper) WGMMA kernel; Blackwell deployments
# keep using the phase-1 indexed kernel.
SUPPORTED_GROUPED_COMPUTE_CAPABILITIES = frozenset({(9, 0)})


@dataclasses.dataclass(frozen=True, eq=False)
class GroupedPreparedWeight:
    """Packed tensors for ``masked`` and ``contiguous``."""

    packed: torch.Tensor  # [G, N, K/2] viewed as float8_e4m3fn
    scale: torch.Tensor   # [G, K/32, N] BF16, N contiguous (MN-major)
    n: int
    k: int
    num_experts: int

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.packed, self.scale

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.tensors)


@functools.lru_cache(maxsize=8)
def _w4a16_perm(block_k: int, device_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form (row, nib) <- (nl, kl) bit-permutation for the N64 x block_k atom.

    Bit map (from DeepGEMM ``_load_w4a16_perm``, verified bit-identical to its
    JSON tables):
        logical nl=(n0..n5), kl=(k0..k6)
        nib = (n3,k3,k0,k1,k2,k5,k6)   # 7 bits -> nibble position within block_k
        row = (n0,n1,n2,k4,n4,n5)      # 6 bits -> row within the 64-row atom
    Returns ``(rows[N_ATOM, block_k], nibs[N_ATOM, block_k])`` long tensors.
    """
    device = torch.device(device_str)
    nk = block_k.bit_length() - 1  # k-index bit count (block_k is a power of two)
    nl = torch.arange(W4A16_N_ATOM, dtype=torch.long, device=device).reshape(-1, 1)
    kl = torch.arange(block_k, dtype=torch.long, device=device).reshape(1, -1)
    n = [(nl >> b) & 1 for b in range(6)]

    def k(b: int) -> torch.Tensor:
        return ((kl >> b) & 1) if b < nk else torch.zeros_like(kl)

    nib = (n[3] << 0) | (k(3) << 1) | (k(0) << 2) | (k(1) << 3) | (k(2) << 4) | (k(5) << 5) | (k(6) << 6)
    row = (n[0] << 0) | (n[1] << 1) | (n[2] << 2) | (k(4) << 3) | (n[4] << 4) | (n[5] << 5)
    rows = row.expand(W4A16_N_ATOM, block_k).clone()
    nibs = nib.expand(W4A16_N_ATOM, block_k).clone()
    return rows, nibs


def reorder_w4a16(b_int8: torch.Tensor, block_k: int = PACK_BLOCK_K) -> torch.Tensor:
    """Reorder signed int4 ``[n, k]`` (int8 in [-8, 7]) to ``[n, k//2]``
    ``float8_e4m3fn``, excess-8 pre-flipped."""
    if b_int8.dim() != 2:
        raise ValueError(f"b_int8 must be 2D [n, k], got {tuple(b_int8.shape)}")
    n, k = b_int8.shape
    if n % W4A16_N_ATOM or k % block_k:
        raise ValueError(
            f"n must be a multiple of {W4A16_N_ATOM} and k of {block_k}, "
            f"got n={n}, k={k}"
        )
    rows, nibs = _w4a16_perm(block_k, str(b_int8.device))
    na, nb = n // W4A16_N_ATOM, k // block_k
    src = b_int8.reshape(na, W4A16_N_ATOM, nb, block_k)
    dst = torch.zeros_like(src)
    nl_idx = (
        torch.arange(W4A16_N_ATOM, device=b_int8.device)
        .reshape(-1, 1)
        .repeat(1, block_k)
        .reshape(-1)
    )
    kl_idx = (
        torch.arange(block_k, device=b_int8.device)
        .reshape(1, -1)
        .repeat(W4A16_N_ATOM, 1)
        .reshape(-1)
    )
    dst[:, rows.reshape(-1), :, nibs.reshape(-1)] = src[:, nl_idx, :, kl_idx]
    dst = dst.reshape(n, k)
    u = dst.view(torch.uint8)
    # Pre-flip nibbles to excess-8 (XOR 0x8): the kernel's LOP3 dequant expects it.
    lo = (u[:, 0::2] & 0x0F) ^ 0x08
    hi = (u[:, 1::2] & 0x0F) ^ 0x08
    return ((hi << 4) | lo).view(torch.float8_e4m3fn)


def _unpack_checkpoint_codes(weight: torch.Tensor) -> torch.Tensor:
    """Unpack [G, N, K/8] INT32 checkpoint words into [G, N, K] uint4 codes.

    Each INT32 word holds eight little-endian nibbles; the returned int8 tensor
    holds the unsigned codes in [0, 15].
    """
    words = weight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64, device=weight.device) * 4
    codes = (words.unsqueeze(-1) >> shifts) & 0xF
    return codes.reshape(*weight.shape[:-1], weight.shape[-1] * 8).to(torch.int8)


def pack_weight_grouped(codes_uint4: torch.Tensor, block_k: int = PACK_BLOCK_K) -> torch.Tensor:
    """Pack ``[G, N, K]`` uint4 codes into ``[G, N, K/2]`` ``float8_e4m3fn``."""
    signed = (codes_uint4.to(torch.int16) - 8).to(torch.int8)
    out = torch.empty(
        (*codes_uint4.shape[:-1], codes_uint4.shape[-1] // 2),
        dtype=torch.float8_e4m3fn,
        device=codes_uint4.device,
    )
    for g in range(codes_uint4.shape[0]):
        out[g] = reorder_w4a16(signed[g], block_k=block_k)
    return out


def pack_w4a16_grouped(
    weight_uint4: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    packed: bool = False,
) -> GroupedPreparedWeight:
    """Pack for ``masked``/``contiguous``; same inputs as ``pack_w4a16_indexed``.

    int32 codes ``[G, N, K]`` in ``[0, 15]`` (``packed=True``: ``[G, N, K/8]``),
    BF16 scale ``[G, N, K/32]``.
    """
    if not isinstance(weight_uint4, torch.Tensor):
        raise TypeError(
            f"weight_uint4 must be a torch.Tensor, got {type(weight_uint4).__name__}"
        )
    if not isinstance(weight_scale, torch.Tensor):
        raise TypeError(
            f"weight_scale must be a torch.Tensor, got {type(weight_scale).__name__}"
        )
    if weight_uint4.ndim != 3:
        raise ValueError(
            f"weight_uint4 must have shape [G, N, {'K/8' if packed else 'K'}], "
            f"got {tuple(weight_uint4.shape)}"
        )
    if weight_uint4.dtype != torch.int32:
        raise TypeError(f"weight_uint4 must be torch.int32, got {weight_uint4.dtype}")
    if not weight_uint4.is_cuda or not weight_uint4.is_contiguous():
        raise ValueError("weight_uint4 must be a contiguous CUDA tensor")
    capability = torch.cuda.get_device_capability(weight_uint4.device)
    if capability not in SUPPORTED_GROUPED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "the grouped W4A16 backend requires an SM90 Hopper GPU, got "
            f"compute capability {capability[0]}.{capability[1]} on "
            f"{weight_uint4.device}"
        )

    num_experts, shape_n, stored_k = weight_uint4.shape
    shape_k = stored_k * 8 if packed else stored_k
    for name, value in (
        ("num_experts", num_experts),
        ("shape_n", shape_n),
        ("shape_k", shape_k),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be non-zero, got {value}")
    if shape_n % W4A16_N_ATOM:
        raise ValueError(
            f"shape_n must be divisible by the {W4A16_N_ATOM}-row weight atom, "
            f"got {shape_n}"
        )
    if shape_k % PACK_BLOCK_K:
        raise ValueError(f"shape_k must be divisible by {PACK_BLOCK_K}, got {shape_k}")
    if shape_k % W4A16_SCALE_GROUP:
        raise ValueError(f"shape_k must be divisible by {W4A16_SCALE_GROUP}")
    if shape_n % 16:
        raise ValueError("shape_n must be a multiple of 16 for the scale TMA descriptor")
    if tuple(weight_scale.shape) != (num_experts, shape_n, shape_k // W4A16_SCALE_GROUP):
        raise ValueError(
            f"weight_scale must have shape "
            f"{(num_experts, shape_n, shape_k // W4A16_SCALE_GROUP)}, "
            f"got {tuple(weight_scale.shape)}"
        )
    if weight_scale.dtype != torch.bfloat16:
        raise TypeError(f"weight_scale must be torch.bfloat16, got {weight_scale.dtype}")
    if weight_scale.device != weight_uint4.device or not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous and on the weight device")

    if packed:
        codes = _unpack_checkpoint_codes(weight_uint4)
    else:
        minimum, maximum = torch.aminmax(weight_uint4)
        if minimum.item() < 0 or maximum.item() > 15:
            raise ValueError(
                "weight_uint4 values must be in [0, 15], got "
                f"[{minimum.item()}, {maximum.item()}]"
            )
        codes = weight_uint4.to(torch.int8)
    packed_weight = pack_weight_grouped(codes)

    # Humming checkpoints store scales N-major [G, N, K/32]; the kernel's SFA
    # TMA descriptor reads MN-major [G, K/32, N] (N contiguous).  Transposing
    # once here at pack time keeps the forward hot path copy-free.
    packed_scale = weight_scale.transpose(1, 2).contiguous()
    return GroupedPreparedWeight(
        packed=packed_weight,
        scale=packed_scale,
        n=shape_n,
        k=shape_k,
        num_experts=num_experts,
    )


__all__ = [
    "GroupedPreparedWeight",
    "PACK_BLOCK_K",
    "SUPPORTED_GROUPED_COMPUTE_CAPABILITIES",
    "pack_w4a16_grouped",
    "reorder_w4a16",
]
