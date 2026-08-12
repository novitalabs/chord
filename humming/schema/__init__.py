"""Humming-compatible schema package for the chord indexed W4A16 operator.

Mirrors the upstream ``humming.schema`` layout: ``BaseWeightSchema`` /
``BaseInputSchema`` factory bases with the schema maps populated here, a real
``HummingWeightSchema`` (uint4 + group-32 + BF16) and BF16-passthrough
``HummingInputSchema``, and a working compressed-tensors weight schema for
pack-quantized INT4 group-32 checkpoints (the Kimi-K2.x MoE format).  Every
other schema name vLLM may import resolves but raises ``NotImplementedError``
when constructed, so unsupported quantizations fail closed at load.
"""

from humming.schema.base import BaseInputSchema, BaseWeightSchema
from humming.schema.compressed_tensors import (
    CompressedTensorsInputSchema,
    CompressedTensorsWeightSchema,
)
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
from humming.schema.stubs import (
    AWQWeightSchema,
    AutoRoundWeightSchema,
    BitnetWeightSchema,
    Fp8InputSchema,
    GPTQWeightSchema,
    GptOssMxfp4WeightSchema,
    Mxfp4WeightSchema,
)

WEIGHT_SCHEMA_MAP: dict[str, type[BaseWeightSchema]] = {
    "chord": HummingWeightSchema,
    "humming": HummingWeightSchema,
    "compressed-tensors": CompressedTensorsWeightSchema,
}

INPUT_SCHEMA_MAP: dict[str, type[BaseInputSchema]] = {
    "chord": HummingInputSchema,
    "humming": HummingInputSchema,
    "compressed-tensors": CompressedTensorsInputSchema,
}

BaseWeightSchema.WEIGHT_SCHEMA_MAP = WEIGHT_SCHEMA_MAP
BaseInputSchema.INPUT_SCHEMA_MAP = INPUT_SCHEMA_MAP


__all__ = [
    "AWQWeightSchema",
    "AutoRoundWeightSchema",
    "BaseInputSchema",
    "BaseWeightSchema",
    "BitnetWeightSchema",
    "CompressedTensorsInputSchema",
    "CompressedTensorsWeightSchema",
    "Fp8InputSchema",
    "GPTQWeightSchema",
    "GptOssMxfp4WeightSchema",
    "HummingInputSchema",
    "HummingWeightSchema",
    "INPUT_SCHEMA_MAP",
    "Mxfp4WeightSchema",
    "WEIGHT_SCHEMA_MAP",
]
