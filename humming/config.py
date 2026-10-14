"""Humming-shaped config enums for framework integration.

``chord.config`` carries chord's own deployment policy surface; this module
instead mirrors the leaf enum names a Humming-shaped framework adapter
imports, plus the ``LayerConfig`` metadata object of the functional
interface (``humming.transform`` / ``humming.forward``).  Chord implements
only the indexed GEMM family — the grouped members exist so class-level
references resolve, and selecting them raises during config/schedule
validation rather than silently falling back.
"""

import dataclasses
from enum import Enum
from typing import Any


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


@dataclasses.dataclass(eq=False)
class LayerConfig:
    """The layer metadata vLLM's functional Humming interface carries.

    Built by ``humming.transform.prepare_layer_config`` and handed back to
    ``humming_forward`` / ``get_heuristics_config``.  It exposes the upstream
    attribute names vLLM reads to size buffers and pick activation paths
    (shapes, padding, dtypes, scale group sizes); ``meta`` is chord's resolved
    indexed layer metadata, which owns the profile and its tuning table.
    Upstream fields chord has no use for keep their neutral defaults.
    """

    shape_n: int
    shape_k: int
    num_experts: int
    a_dtype: Any
    b_dtype: Any
    c_dtype: Any
    bs_dtype: Any = None
    as_dtype: Any = None
    pad_shape_n: int = 0
    pad_shape_k: int = 0
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | None = None
    weight_scale_2_type: WeightScale2Type = WeightScale2Type.NONE
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    has_bias: bool = False
    use_packed_k_layout: bool = False
    mma_type: Any = None
    meta: Any = dataclasses.field(default=None, repr=False)

    def to_str(self) -> str:
        return (
            f"chord-indexed-w4a16:{self.meta.profile.name}:"
            f"E{self.num_experts}:N{self.shape_n}:K{self.shape_k}"
        )


__all__ = ["GemmType", "LayerConfig", "WeightScale2Type", "WeightScaleType"]
