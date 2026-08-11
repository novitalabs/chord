"""Humming-shaped config enums for framework integration.

``chord.config`` carries chord's own deployment policy surface; this module
instead mirrors the leaf enum names a Humming-shaped framework adapter
imports.  Chord implements only the indexed GEMM family — the grouped members
exist so class-level references resolve, and selecting them raises during
config/schedule validation rather than silently falling back.
"""

from enum import Enum


class GemmType(Enum):
    DENSE = "dense"
    INDEXED = "indexed"
    GROUPED_CONTIGUOUS = "grouped_contiguous"
    GROUPED_MASKED = "grouped_masked"


class WeightScaleType(Enum):
    GROUP = "group"
    BLOCK = "block"
    CHANNEL = "channel"
    TENSOR = "tensor"
    # Present in vLLM's QuantKey mapping (second-level tensor scale); the
    # indexed W4A16 schema never selects it.
    GROUP_TENSOR = "group_tensor"

    def __str__(self) -> str:
        return self.value


class WeightScale2Type(Enum):
    NONE = "none"
    CHANNEL = "channel"
    TENSOR = "tensor"

    def __str__(self) -> str:
        return self.value


__all__ = ["GemmType", "WeightScale2Type", "WeightScaleType"]
