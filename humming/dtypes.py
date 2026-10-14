"""Humming-compatible dtype table, re-exporting chord's shared dtype objects.

vLLM builds ``{humming.dtype: torch.dtype}`` lookup maps with every common
dtype at call time, so the full attribute set must exist even though the
indexed W4A16 kernel only consumes ``bfloat16`` activations/scales and
``uint4`` weights.  These are the same ``DataType`` instances that
``chord.dtypes`` and ``chord_kernels.operator.dtypes`` export, so layer
metadata compares equal whichever import root produced it.
"""

import torch

from chord_kernels.operator.dtypes import (
    DataType,
    bfloat16,
    float16,
    float32,
    float4e2m1,
    float8e4m3,
    float8e5m2,
    float8e8m0,
    int4,
    int8,
    uint2,
    uint3,
    uint4,
    uint5,
    uint6,
    uint7,
    uint8,
)

__all__ = [
    "DataType",
    "bfloat16",
    "float16",
    "float32",
    "float4e2m1",
    "float8e4m3",
    "float8e5m2",
    "float8e8m0",
    "int4",
    "int8",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "uint8",
]

# vLLM's humming experts size their workspace buffers through this map.
torch_dtype_map = {
    float16: torch.float16,
    bfloat16: torch.bfloat16,
    float32: torch.float32,
    float8e4m3: torch.float8_e4m3fn,
    float8e5m2: torch.float8_e5m2,
    int8: torch.int8,
    int4: torch.uint8,
}
__all__.append("torch_dtype_map")
