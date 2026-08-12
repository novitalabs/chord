"""Humming-native weight/input schemas, scoped to the indexed W4A16 contract.

Derived from upstream ``humming/schema/humming.py`` with the fields vLLM's
integration reads.  The class itself stays permissive about bit width/scale
variants so vLLM's own gates produce the right QuantKey; chord's layer
machinery (``_validate_indexed_schema``) rejects anything outside
uint4 + group-32 + BF16 before a kernel is packed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import torch

from chord_kernels.operator import dtypes

from humming.config import WeightScale2Type, WeightScaleType
from humming.schema.base import BaseInputSchema, BaseWeightSchema


@dataclasses.dataclass(kw_only=True)
class HummingWeightSchema(BaseWeightSchema):
    quant_method: str = "humming"
    b_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType | None = None
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | str | None = None
    weight_scale_2_type: WeightScale2Type | str | None = None
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    hadamard_block_size: int = 0

    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {
        "b_dtype": ["weight_dtype", "dtype"],
        "weight_scale_group_size": ["group_size"],
        "weight_scale_group_size_n": ["group_size_n"],
        "weight_scale_type": ["scale_type"],
        "weight_scale_2_type": ["scale_2_type"],
        "bs_dtype": ["weight_scale_dtype", "scale_dtype"],
    }

    def __post_init__(self):
        if isinstance(self.b_dtype, str):
            self.b_dtype = dtypes.DataType.from_str(str(self.b_dtype))
        if (
            isinstance(self.b_dtype, dtypes.DataType)
            and self.b_dtype.is_integer_type
            and self.b_dtype.is_signed
        ):
            self.b_dtype = dtypes.DataType.from_str("u" + str(self.b_dtype))
        if isinstance(self.bs_dtype, str):
            self.bs_dtype = dtypes.DataType.from_str(str(self.bs_dtype))

        if isinstance(self.weight_scale_type, str):
            self.weight_scale_type = WeightScaleType(self.weight_scale_type)
        elif self.weight_scale_type is None:
            if self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL
            elif self.weight_scale_group_size > 0:
                self.weight_scale_type = WeightScaleType.GROUP

        if isinstance(self.weight_scale_2_type, str):
            self.weight_scale_2_type = WeightScale2Type(self.weight_scale_2_type)
        if self.weight_scale_2_type is None:
            self.weight_scale_2_type = WeightScale2Type.NONE
        if self.weight_scale_2_type != WeightScale2Type.NONE:
            assert self.weight_scale_type == WeightScaleType.GROUP, (
                "weight_scale_2_type requires weight_scale_type='group'"
            )

        if self.weight_scale_type == WeightScaleType.BLOCK:
            self.bs_dtype = dtypes.float32
        if self.bs_dtype is None and self.weight_scale_type in (
            WeightScaleType.GROUP,
            WeightScaleType.CHANNEL,
        ):
            self.bs_dtype = dtypes.bfloat16

    @property
    def has_tensor_weight_scale(self) -> bool:
        return (
            self.weight_scale_type == WeightScaleType.TENSOR
            or self.weight_scale_2_type == WeightScale2Type.TENSOR
        )

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        num_bits = self.b_dtype.num_bits
        group_size = self.weight_scale_group_size or shape_k

        scale_torch_dtype = param_dtype
        if self.bs_dtype == dtypes.float8e8m0:
            scale_torch_dtype = torch.float8_e8m0fnu
        elif self.bs_dtype == dtypes.float8e4m3:
            scale_torch_dtype = torch.float8_e4m3fn
        elif self.bs_dtype == dtypes.float8e5m2:
            scale_torch_dtype = torch.float8_e5m2

        tensor_meta: dict[str, Any] = {
            "weight": {
                "shape": (shape_n, shape_k * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {
                    "input_dim": 1,
                    "output_dim": 0,
                    "packed_factor": 32 / num_bits,
                    "packed_dim": 1,
                },
            }
        }

        if self.weight_scale_type == WeightScaleType.GROUP:
            tensor_meta["weight_scale"] = {
                "shape": (shape_n, shape_k // group_size),
                "dtype": scale_torch_dtype,
                "extra_attrs": {
                    "input_dim": 1,
                    "output_dim": 0,
                    "scale_type": "group",
                },
            }
        elif self.weight_scale_type == WeightScaleType.CHANNEL:
            tensor_meta["weight_scale"] = {
                "shape": (shape_n, 1),
                "dtype": scale_torch_dtype,
                "extra_attrs": {"output_dim": 0, "scale_type": "channel"},
            }
        else:
            raise NotImplementedError(
                "the chord humming shim implements group/channel scales only"
            )

        if self.has_zero_point:
            raise NotImplementedError(
                "the chord humming shim implements symmetric W4A16 only"
            )
        if has_bias:
            tensor_meta["bias"] = {
                "shape": (shape_n,),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 0},
            }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta

    def infer_shape(
        self, tensors: dict[str, torch.Tensor]
    ) -> tuple[int, int, int | None, bool]:
        num_bits = self.b_dtype.num_bits
        shape_n = tensors["weight"].size(-2)
        shape_k = tensors["weight"].size(-1) * 32 // num_bits
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    def requant_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        target_weight_schema: "HummingWeightSchema",
        param_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "requantization is not supported by the chord humming shim"
        )

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple["HummingWeightSchema", dict[str, torch.Tensor]]:
        del shape_n_stacks, shape_k_stacks
        schema = dataclasses.replace(self)
        if schema.weight_scale_type in (
            WeightScaleType.GROUP,
            WeightScaleType.CHANNEL,
        ):
            if tensors["weight_scale"].dtype == torch.float32:
                tensors["weight_scale"] = tensors["weight_scale"].to(param_dtype)
        return schema, tensors


@dataclasses.dataclass(kw_only=True)
class HummingInputSchema(BaseInputSchema):
    quant_method: str = "humming"
    a_dtype: dtypes.DataType | None = None
    input_scale_group_size: int = 0
    input_scale_dtype: dtypes.DataType | None = None

    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {
        "a_dtype": ["input_dtype", "dtype"],
        "input_scale_group_size": ["group_size"],
        "input_scale_dtype": ["scale_dtype"],
    }

    def __post_init__(self):
        if isinstance(self.a_dtype, str):
            self.a_dtype = dtypes.DataType.from_str(str(self.a_dtype))
        if isinstance(self.input_scale_dtype, str):
            self.input_scale_dtype = dtypes.DataType.from_str(
                str(self.input_scale_dtype)
            )

    def get_activation_bits(self):
        if self.a_dtype is None:
            return 16
        return self.a_dtype.num_bits

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        sm_version: int | tuple[int, int] | None = None,
    ) -> tuple["HummingInputSchema", dict[str, torch.Tensor]]:
        return self, {}


__all__ = ["HummingInputSchema", "HummingWeightSchema"]
