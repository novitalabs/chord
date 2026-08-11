"""Out-of-scope schema placeholders for the chord humming shim.

The names below are imported unconditionally inside vLLM helpers, so they must
resolve; constructing or using them raises ``NotImplementedError`` — the
shim's working surface is indexed W4A16 (uint4 + group-32 + BF16) only.
"""

from typing import Any


class _UnsupportedSchema:
    quant_method = "unsupported"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"the chord humming shim implements indexed W4A16 only; "
            f"{type(self).__name__} is not supported"
        )


class AWQWeightSchema(_UnsupportedSchema):
    pass


class AutoRoundWeightSchema(_UnsupportedSchema):
    pass


class BitnetWeightSchema(_UnsupportedSchema):
    pass


class GPTQWeightSchema(_UnsupportedSchema):
    pass


class Mxfp4WeightSchema(_UnsupportedSchema):
    pass


class GptOssMxfp4WeightSchema(_UnsupportedSchema):
    pass


class Fp8InputSchema(_UnsupportedSchema):
    pass


__all__ = [
    "AWQWeightSchema",
    "AutoRoundWeightSchema",
    "BitnetWeightSchema",
    "Fp8InputSchema",
    "GPTQWeightSchema",
    "GptOssMxfp4WeightSchema",
    "Mxfp4WeightSchema",
]
