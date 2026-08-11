"""Compressed-tensors weight schema, scoped to the shim's W4A16 contract.

Derived from upstream ``humming/schema/compressed_tensors.py``.  The
``pack-quantized`` + integer + group strategy layout (what Kimi-class INT4
MoE checkpoints ship) is fully wired through conversion; every other
compressed-tensors layout combination raises at construction or conversion
with an actionable message.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from chord_kernels.operator import dtypes

from humming.schema.base import BaseInputSchema, BaseWeightSchema
from humming.schema.humming import HummingInputSchema, HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class CompressedTensorsWeightSchema(BaseWeightSchema):
    quant_method: str = "compressed-tensors"

    format: str
    type: str
    num_bits: int
    strategy: str
    symmetric: bool = True
    block_structure: tuple[int, int] | None = None
    group_size: int | None = None
    actorder: str | None = None

    def __post_init__(self):
        if self.format != "pack-quantized":
            raise NotImplementedError(
                "the chord humming shim implements compressed-tensors "
                f"'pack-quantized' only, got format={self.format!r}"
            )
        if self.type != "int" or self.num_bits != 4:
            raise NotImplementedError(
                "the chord humming shim implements W4A16 only, got "
                f"{self.type}{self.num_bits}"
            )
        if self.actorder is not None:
            raise NotImplementedError("actorder is not supported by humming")
        if not self.symmetric:
            raise NotImplementedError(
                "the chord humming shim implements symmetric quantization only"
            )
        self.weight_key = "weight_packed"
        if isinstance(self.block_structure, list):
            self.block_structure = tuple(self.block_structure)

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        weight_shape = (shape_n, shape_k * self.num_bits // 32)

        if "group" in self.strategy:
            assert self.group_size is not None
            scale_shape = (shape_n, shape_k // self.group_size)
        else:
            raise NotImplementedError(
                "the chord humming shim implements group strategy only, "
                f"got strategy={self.strategy!r}"
            )

        tensor_meta: dict[str, Any] = {
            self.weight_key: {
                "shape": weight_shape,
                "dtype": torch.int32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            },
            "weight_scale": {
                "shape": scale_shape,
                "dtype": param_dtype,
                "extra_attrs": {"scale_type": "group"},
            },
        }

        tensor_meta["weight_scale"]["extra_attrs"]["packed_dim"] = 1
        packed_factor = 32 / self.num_bits
        tensor_meta["weight_scale"]["extra_attrs"]["packed_factor"] = packed_factor
        tensor_meta["weight_scale"]["extra_attrs"]["input_dim"] = 1
        tensor_meta["weight_scale"]["extra_attrs"]["output_dim"] = 0

        if self.format == "pack-quantized":
            tensor_meta["weight_shape"] = {
                "shape": (2,),
                "dtype": torch.int64,
            }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta

    def infer_shape(
        self, tensors: dict[str, torch.Tensor]
    ) -> tuple[int, int, int | None, bool]:
        weight = tensors[self.weight_key]
        shape_n = weight.size(-2)
        shape_k = weight.size(-1) * 32 // self.num_bits
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[HummingWeightSchema, dict[str, torch.Tensor]]:
        del shape_n_stacks, shape_k_stacks
        weight = tensors[self.weight_key].view(torch.int32)
        weight_scale = tensors["weight_scale"].to(param_dtype)

        assert self.group_size is not None
        # CT pack-quantized keeps the same little-endian nibble packing along K
        # that humming's checkpoint layout uses; only the scale needs casting.
        output_tensors = {"weight": weight, "weight_scale": weight_scale}
        schema = HummingWeightSchema(
            b_dtype=dtypes.DataType.from_str(f"uint{self.num_bits}"),
            weight_scale_group_size=self.group_size,
            weight_scale_type="group",
            has_zero_point=False,
        )
        return schema, output_tensors


@dataclasses.dataclass(kw_only=True)
class CompressedTensorsInputSchema(BaseInputSchema):
    quant_method: str = "compressed-tensors"

    format: str
    type: str
    num_bits: int
    dynamic: bool | str
    group_size: int
    symmetric: bool = True

    def __post_init__(self):
        self.input_scale_key = "input_scale"

    def get_activation_bits(self):
        return self.num_bits

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        sm_version: int | tuple[int, int] | None = None,
    ) -> tuple[HummingInputSchema, dict[str, torch.Tensor]]:
        a_dtype = self.get_fallback_input_dtype(
            dtypes.DataType.from_str(f"{self.type}{self.num_bits}"), sm_version
        )
        return HummingInputSchema(a_dtype=a_dtype), {}


__all__ = [
    "CompressedTensorsInputSchema",
    "CompressedTensorsWeightSchema",
]
