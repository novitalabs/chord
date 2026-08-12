"""Online quantization placeholder for the chord humming shim."""

from typing import Any


def quantize_weight(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "the chord humming shim only serves pre-quantized W4A16 checkpoints; "
        "online quantize_weight is not implemented"
    )


__all__ = ["quantize_weight"]
