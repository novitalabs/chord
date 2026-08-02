# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import dataclasses


@dataclasses.dataclass(frozen=True)
class DataType:
    """Minimal dtype descriptor used by the layer integration schema."""

    name: str
    num_bits: int
    is_signed: bool
    is_integer_type: bool = False
    is_floating_point_type: bool = False
    exponent_bits: int | None = None
    mantissa_bits: int | None = None

    @classmethod
    def from_str(cls, value: object) -> "DataType":
        if isinstance(value, DataType):
            return value
        if not isinstance(value, str):
            raise TypeError(f"dtype must be a string, got {type(value).__name__}")

        normalized = value.lower()
        if normalized == "uint4":
            return uint4
        if normalized in ("bfloat16", "bf16"):
            return bfloat16
        raise ValueError(f"unsupported indexed W4A16 dtype: {value!r}")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


uint4 = DataType(
    name="uint4",
    num_bits=4,
    is_signed=False,
    is_integer_type=True,
)
bfloat16 = DataType(
    name="bfloat16",
    num_bits=16,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=8,
    mantissa_bits=7,
)
