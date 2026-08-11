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
        if normalized == "bf16":
            normalized = "bfloat16"
        if normalized in _BY_NAME:
            return _BY_NAME[normalized]
        raise ValueError(f"unsupported dtype: {value!r}")

    @classmethod
    def from_torch_dtype(cls, value: object) -> "DataType":
        """Map a ``torch.dtype`` to a :class:`DataType` without importing torch."""
        name = str(value).rsplit(".", 1)[-1].lower()
        if name in _BY_NAME:
            return _BY_NAME[name]
        raise ValueError(f"unsupported torch dtype: {value!r}")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


uint2 = DataType(name="uint2", num_bits=2, is_signed=False, is_integer_type=True)
uint3 = DataType(name="uint3", num_bits=3, is_signed=False, is_integer_type=True)
uint4 = DataType(
    name="uint4",
    num_bits=4,
    is_signed=False,
    is_integer_type=True,
)
uint8 = DataType(name="uint8", num_bits=8, is_signed=False, is_integer_type=True)
int4 = DataType(name="int4", num_bits=4, is_signed=True, is_integer_type=True)
int8 = DataType(name="int8", num_bits=8, is_signed=True, is_integer_type=True)
float16 = DataType(
    name="float16",
    num_bits=16,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=5,
    mantissa_bits=10,
)
bfloat16 = DataType(
    name="bfloat16",
    num_bits=16,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=8,
    mantissa_bits=7,
)
float32 = DataType(
    name="float32",
    num_bits=32,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=8,
    mantissa_bits=23,
)
float8e4m3 = DataType(
    name="float8e4m3",
    num_bits=8,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=4,
    mantissa_bits=3,
)
float8e5m2 = DataType(
    name="float8e5m2",
    num_bits=8,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=5,
    mantissa_bits=2,
)
float8e8m0 = DataType(
    name="float8e8m0",
    num_bits=8,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=8,
    mantissa_bits=0,
)
float4e2m1 = DataType(
    name="float4e2m1",
    num_bits=4,
    is_signed=True,
    is_floating_point_type=True,
    exponent_bits=2,
    mantissa_bits=1,
)

_BY_NAME = {
    dtype.name: dtype
    for dtype in (
        uint2,
        uint3,
        uint4,
        uint8,
        int4,
        int8,
        float16,
        bfloat16,
        float32,
        float8e4m3,
        float8e5m2,
        float8e8m0,
        float4e2m1,
    )
}

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
    "uint8",
]
