"""Humming-name aliases for the operator entry points.

Mirrors the ``humming.ops`` surface that MoE W4A16 serving actually calls.
Ops outside the extracted scope (quantization, Hadamard, MXFP4 processing)
are intentionally absent — an adapter that reaches them on the chord backend
should fail at import, not silently diverge at runtime.
"""

from __future__ import annotations

import torch

from chord_kernels.operator.ops import (
    init_launcher,
    launch_kernel,
    register_kernel,
)

# Upstream spelling for the launcher bootstrap.
init_humming_launcher = init_launcher


def unpack_weight(inputs: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Unpack INT32 checkpoint words into 4-bit codes (W4A16 scope only)."""
    if num_bits != 4:
        raise ValueError(
            f"chord ships the W4A16 operator; unpack_weight supports num_bits=4, "
            f"got {num_bits}"
        )
    from chord_kernels.operator.layer import unpack_packed_uint4

    return unpack_packed_uint4(inputs)


__all__ = [
    "init_humming_launcher",
    "init_launcher",
    "launch_kernel",
    "register_kernel",
    "unpack_weight",
]
