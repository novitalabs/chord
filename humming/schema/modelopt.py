"""Modelopt schema placeholders for the chord humming shim.

These names are imported unconditionally inside vLLM helpers, so they must
resolve; constructing or using them raises ``NotImplementedError``.
"""

from humming.schema.stubs import _UnsupportedSchema


class ModeloptMxfp8WeightSchema(_UnsupportedSchema):
    pass


class ModeloptNvfp4InputSchema(_UnsupportedSchema):
    pass


class ModeloptNvfp4WeightSchema(_UnsupportedSchema):
    pass


__all__ = [
    "ModeloptMxfp8WeightSchema",
    "ModeloptNvfp4InputSchema",
    "ModeloptNvfp4WeightSchema",
]
