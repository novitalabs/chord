# Derived from deepseek-ai/DeepGEMM; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.
"""Weight packing for the grouped W4A16 backend.

The packing itself was bit-exactly ported from DeepGEMM's
``tests/generators.py::reorder_w4a16``: within each ``N_ATOM(64) x BLOCK_K``
tile the
INT4 weight is reordered by a pure bit-permutation of the 13-bit ``(nl, kl)``
atom index so that after TMA + ldmatrix + the kernel's int4->bf16 dequant the
fragments line up with the BF16 RS-WGMMA A-operand; nibbles are pre-flipped
to excess-8 (``^0x08``) because the kernel's LOP3 dequant expects that.

The reorder perm width MUST equal the kernel's ``BLOCK_K``, and the vendored
heuristic fixes ``BLOCK_K`` per mode (128 for masked decode, 64 for contiguous
prefill), so the two modes pack DIFFERENT weight buffers.  The mode recorded
on :class:`GroupedPreparedWeight` is the guard that keeps a masked-packed
weight from ever being fed to the contiguous kernel (a silent wrong answer).
The BF16 scale layout ``[G, K/32, N]`` (N contiguous) is shared by both modes —
it is exactly what the kernel's SFA TMA descriptor reads, so no per-forward
transpose happens on the hot path.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterator
from typing import Literal

import torch

from chord_kernels.operator.grouped.heuristics import W4A16_N_ATOM, W4A16_SCALE_GROUP

GroupedMode = Literal["masked", "contiguous"]

# Perm widths baked into the packed buffer; they must match the vendored
# heuristic's per-mode BLOCK_K selection (masked decode BK128, contiguous
# prefill BK64).
MASKED_BLOCK_K = 128
CONTIGUOUS_BLOCK_K = 64

_MODE_BLOCK_K: dict[str, int] = {
    "masked": MASKED_BLOCK_K,
    "contiguous": CONTIGUOUS_BLOCK_K,
}

# The vendored kernel is an SM90 (Hopper) WGMMA kernel; Blackwell deployments
# keep using the phase-1 indexed kernel.
SUPPORTED_GROUPED_COMPUTE_CAPABILITIES = frozenset({(9, 0)})


def block_k_for_mode(mode: GroupedMode) -> int:
    try:
        return _MODE_BLOCK_K[mode]
    except KeyError:
        raise ValueError(f"mode must be 'masked' or 'contiguous', got {mode!r}") from None


@dataclasses.dataclass(frozen=True, eq=False)
class GroupedPreparedWeight:
    """Packed tensors plus the mode the grouped weight layout was built for."""

    packed: torch.Tensor  # [G, N, K/2] viewed as float8_e4m3fn
    scale: torch.Tensor   # [G, K/32, N] BF16, N contiguous (MN-major)
    mode: GroupedMode
    n: int
    k: int
    num_experts: int

    @property
    def block_k(self) -> int:
        return block_k_for_mode(self.mode)

    @property
    def layout(self) -> str:
        # Present a WeightLayout-compatible spelling so framework code can key
        # off ``weight.layout`` uniformly across the two operator families.
        return f"grouped_{self.mode}"

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


def reorder_w4a16(b_int8: torch.Tensor, block_k: int = MASKED_BLOCK_K) -> torch.Tensor:
    """Reorder a [n, k] signed-int4 weight (values in [-8, 7], stored int8) into
    the grouped packed layout.  Returns ``[n, k//2]`` viewed as
    ``float8_e4m3fn`` (2 nibbles/byte), excess-8 pre-flipped.  Bit-exact port of
    DeepGEMM ``tests/generators.py::reorder_w4a16``.
    """
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
    holds the unsigned codes in [0, 15].  Mirrors
    ``chord_kernels.operator.layer.unpack_packed_uint4`` for the DeepGEMM
    packing path without importing the layer module.
    """
    words = weight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64, device=weight.device) * 4
    codes = (words.unsqueeze(-1) >> shifts) & 0xF
    return codes.reshape(*weight.shape[:-1], weight.shape[-1] * 8).to(torch.int8)


def pack_weight_grouped(codes_uint4: torch.Tensor, block_k: int) -> torch.Tensor:
    """Pack unpacked [G, N, K] uint4 codes ([0, 15]) into the DeepGEMM layout
    [G, N, K/2] float8_e4m3fn.  The DeepGEMM reorder works on signed [-8, 7],
    so map back (v - 8) first; the reorder re-applies the +8 excess-8 flip
    internally.
    """
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
    mode: GroupedMode,
    *,
    packed: bool = False,
) -> GroupedPreparedWeight:
    """Pack quantized W4A16 weights for the grouped W4A16 kernels.

    ``weight_uint4`` follows the chord checkpoint contract: unpacked INT32
    ``[G, N, K]`` unsigned codes in ``[0, 15]``, or with ``packed=True`` the
    compact ``[G, N, K/8]`` INT32 form (eight little-endian nibbles per word).
    ``weight_scale`` is BF16 ``[G, N, K/32]``.
    """
    block_k = block_k_for_mode(mode)
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
    if shape_k % block_k:
        raise ValueError(
            f"shape_k must be divisible by the {mode} BLOCK_K={block_k}, "
            f"got {shape_k}"
        )
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
    packed_weight = pack_weight_grouped(codes, block_k)

    # Humming checkpoints store scales N-major [G, N, K/32]; the kernel's SFA
    # TMA descriptor reads MN-major [G, K/32, N] (N contiguous).  Transposing
    # once here at pack time keeps the forward hot path copy-free.
    packed_scale = weight_scale.transpose(1, 2).contiguous()
    return GroupedPreparedWeight(
        packed=packed_weight,
        scale=packed_scale,
        mode=mode,
        n=shape_n,
        k=shape_k,
        num_experts=num_experts,
    )


__all__ = [
    "CONTIGUOUS_BLOCK_K",
    "GroupedMode",
    "GroupedPreparedWeight",
    "MASKED_BLOCK_K",
    "SUPPORTED_GROUPED_COMPUTE_CAPABILITIES",
    "block_k_for_mode",
    "pack_w4a16_grouped",
    "reorder_w4a16",
]
