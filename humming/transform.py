"""``humming.transform`` for vLLM's functional Humming interface.

vLLM 0.29 prepares a MoE sublayer with two calls through its lazy facade::

    config = prepare_layer_config(shape_n=..., shape_k=..., weight_schema=...,
                                  input_schema=..., num_experts=...,
                                  pad_n_to_multiple=256, pad_k_to_multiple=128,
                                  has_bias=..., torch_dtype=...)
    tensors = transform_humming_tensors(config, {"weight": ..., "weight_scale": ...})

and then replaces ``layer.{sublayer}_{name}`` with the returned tensors.  The
packed weight is resolved against chord's indexed profiles here: the shard
axis comes from the projection shapes, the SM90 role from
``CHORD_SM90_DECODE``, exactly as ``profile='auto'`` resolves on the layer
path.  Only the indexed BF16 x INT4 group-32 contract is backed by a kernel;
anything else fails here, at load, rather than at the first forward.
"""

from __future__ import annotations

import dataclasses

import torch

from chord_kernels.operator import dispatch, dtypes
from chord_kernels.operator.env import resolve_backend_name
from chord_kernels.operator.profiles import (
    IndexedLayerMeta,
    resolve_shard_axis,
    select_indexed_profile,
)

from humming.config import LayerConfig, WeightScale2Type, WeightScaleType
from humming.schema import HummingInputSchema, HummingWeightSchema


def _check_indexed_w4a16(
    weight_schema: HummingWeightSchema, input_schema: HummingInputSchema
) -> None:
    if weight_schema.b_dtype != dtypes.uint4:
        raise NotImplementedError(
            "the chord humming shim implements INT4 weights only, got "
            f"b_dtype={weight_schema.b_dtype}"
        )
    if (
        weight_schema.weight_scale_type != WeightScaleType.GROUP
        or weight_schema.weight_scale_group_size != 32
        or weight_schema.weight_scale_group_size_n not in (0, 1)
    ):
        raise NotImplementedError(
            "the chord humming shim implements group-32 weight scales only, got "
            f"{weight_schema.weight_scale_type} group "
            f"{weight_schema.weight_scale_group_size}"
        )
    if weight_schema.weight_scale_2_type not in (None, WeightScale2Type.NONE):
        raise NotImplementedError("a second-level weight scale is not supported")
    if weight_schema.has_zero_point:
        raise NotImplementedError(
            "the chord humming shim implements symmetric INT4 only"
        )
    if weight_schema.bs_dtype not in (None, dtypes.bfloat16):
        raise NotImplementedError(
            f"the chord humming shim implements BF16 weight scales only, got "
            f"{weight_schema.bs_dtype}"
        )
    a_dtype = input_schema.a_dtype
    if a_dtype is not None and a_dtype != dtypes.bfloat16:
        raise NotImplementedError(
            "the chord humming shim implements BF16 activations only, got "
            f"a_dtype={a_dtype}"
        )


def prepare_layer_config(
    shape_n: int,
    shape_k: int,
    weight_schema: HummingWeightSchema,
    input_schema: HummingInputSchema | None = None,
    num_experts: int | None = None,
    pad_n_to_multiple: int = 1,
    pad_k_to_multiple: int = 1,
    has_bias: bool = False,
    torch_dtype: torch.dtype | None = None,
    device: int | torch.device | None = None,
) -> LayerConfig:
    """Resolve the indexed profile and return the layer's metadata.

    ``pad_*_to_multiple`` are accepted and ignored: the indexed kernel needs
    no padding for the shapes it is published for (``N % 128 == 0``,
    ``K % 64 == 0``), so ``pad_shape_n``/``pad_shape_k`` are always 0 and the
    activation/output buffers vLLM sizes from the config stay unpadded.
    """
    del pad_n_to_multiple, pad_k_to_multiple
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    if torch_dtype != torch.bfloat16:
        raise NotImplementedError(
            f"the chord humming shim implements BF16 layers only, got {torch_dtype}"
        )
    if input_schema is None:
        input_schema = HummingInputSchema(a_dtype=dtypes.bfloat16)
    if not isinstance(weight_schema, HummingWeightSchema) or not isinstance(
        input_schema, HummingInputSchema
    ):
        raise TypeError(
            "prepare_layer_config takes converted HummingWeightSchema / "
            "HummingInputSchema objects"
        )
    _check_indexed_w4a16(weight_schema, input_schema)
    if has_bias:
        raise NotImplementedError("the indexed W4A16 kernel has no bias input")
    if not num_experts:
        raise NotImplementedError(
            "the chord humming shim implements the MoE (indexed) GEMM only; "
            "dense layers are not supported"
        )
    if shape_n % 128 or shape_k % 64:
        raise NotImplementedError(
            f"indexed W4A16 needs N % 128 == 0 and K % 64 == 0, got "
            f"N={shape_n} K={shape_k}"
        )

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    axis = resolve_shard_axis(None, shape_n, shape_k)
    role = "mix" if axis == "tp" else None
    profile = select_indexed_profile(
        "auto",
        mode=role,
        device=device,
        shape_n=shape_n,
        shape_k=shape_k,
    )
    if profile.is_grouped:
        # CHORD_USE_GROUPED reroutes 'auto' to the grouped backends, whose
        # routing contract vLLM's indexed experts do not produce.
        raise NotImplementedError(
            f"profile {profile.name!r} is a grouped backend; the vLLM humming "
            "integration is indexed-only (unset CHORD_USE_GROUPED)"
        )
    meta = IndexedLayerMeta(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        profile=profile,
    )
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=32,
        weight_scale_type=WeightScaleType.GROUP,
        meta=meta,
    )


def transform_humming_tensors(
    config: LayerConfig,
    tensors: dict[str, torch.Tensor],
    already_padded: bool = False,
) -> dict[str, torch.Tensor]:
    """Pack checkpoint tensors into the profile's kernel layout.

    ``weight`` is the humming checkpoint form ``[E, N, K/8]`` int32 (eight
    little-endian nibbles per word, code - 8 = value) and ``weight_scale`` is
    ``[E, N, K/32]``.  Returns ``{"weight", "weight_scale"}`` in the kernel
    layout; vLLM installs them as ``{sublayer}_weight`` /
    ``{sublayer}_weight_scale`` and passes them back to ``humming_forward``.
    """
    del already_padded
    meta = config.meta
    weight = tensors["weight"]
    scale = tensors["weight_scale"]
    if isinstance(weight, torch.nn.Parameter):
        weight = weight.data
    if isinstance(scale, torch.nn.Parameter):
        scale = scale.data
    expected = (meta.num_experts, meta.shape_n, meta.shape_k // 8)
    if weight.dtype != torch.int32 or tuple(weight.shape) != expected:
        raise ValueError(
            f"weight must be int32 {expected} (packed checkpoint), got "
            f"{tuple(weight.shape)}/{weight.dtype}"
        )
    expected_scale = (meta.num_experts, meta.shape_n, meta.shape_k // 32)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"weight_scale must be {expected_scale}, got {tuple(scale.shape)}"
        )
    if not weight.is_cuda:
        raise RuntimeError("transform_humming_tensors needs the weight on CUDA")
    meta.profile.validate_device(weight.device)
    scale = scale.to(device=weight.device, dtype=torch.bfloat16).contiguous()
    prepared = dispatch.pack_weight(
        weight.contiguous(), scale, meta=meta, packed=True
    )
    return {"weight": prepared.packed, "weight_scale": prepared.scale}


__all__ = [
    "LayerConfig",
    "prepare_layer_config",
    "transform_humming_tensors",
]
