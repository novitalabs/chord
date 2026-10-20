"""Chord W4A16 MoE CUDA operators."""

from importlib import import_module
from typing import Any

__all__ = [
    "contiguous",
    "indexed",
    "indexed_block_m",
    "masked",
    "pack_w4a16",
]

# Resolved on first access, so importing the package does not import torch.
_EXPORTS = {
    "indexed": "chord_kernels.ops",
    "masked": "chord_kernels.ops",
    "contiguous": "chord_kernels.ops",
    "indexed_block_m": "chord_kernels.operator.api",
    "pack_w4a16": "chord_kernels.operator.packing",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    if name == "operator":
        return import_module("chord_kernels.operator")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
