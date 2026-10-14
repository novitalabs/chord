"""``humming.forward`` for vLLM's functional Humming interface.

vLLM 0.29's humming MoE experts call, per sublayer and step::

    inputs, input_scale = may_quant_input(config, inputs=..., input_scale=...,
                                          quanted_input=...)
    humming_forward(config, inputs=..., weight=..., weight_scale=...,
                    zero_point=None, bias=None, weight_scale_2=None,
                    input_scale=None, outputs=..., locks=...,
                    sorted_ids=..., expert_ids=..., num_tokens_padded=...,
                    compute_config=..., valid_shape_m=..., top_k=...,
                    tuning_config=...)

with ``weight``/``weight_scale`` being the tensors
``humming.transform.transform_humming_tensors`` returned.

The serialized ``tuning_config`` is consulted for one thing only: the block-M
of the row covering ``valid_shape_m``, which is the block size the framework
aligned ``sorted_ids`` to (``moe_align_block_size``).  The tile itself comes
from chord's own table for this layer, re-mapped to that block-M.  vLLM edits
the rows it receives before serializing them (it caps block-K at 128 and
widens warp-N for upstream's TMA descriptors), so launching the edited row
verbatim would silently change chord's tuned tiles.
"""

from __future__ import annotations

import json

import torch

from chord_kernels.operator import dispatch

from humming.config import LayerConfig


def may_quant_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    quanted_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """BF16 activations pass through; the indexed kernel takes no input scale."""
    del config, quanted_input
    if input_scale is not None:
        raise NotImplementedError(
            "the indexed W4A16 kernel takes BF16 activations; a pre-quantized "
            "input (input_scale) is not supported"
        )
    return inputs, None


def _routing_block_m(tuning_config, shape_m: int) -> int | None:
    """The block-M of the framework's row covering ``shape_m``, if any."""
    if tuning_config is None:
        return None
    value = json.loads(tuning_config) if isinstance(tuning_config, str) else tuning_config
    if isinstance(value, dict):
        return int(value["block_shape"][0])
    for lower, upper, row in value:
        if int(lower) < shape_m <= int(upper):
            return int(row["block_shape"][0])
    raise ValueError(f"no tuning row covers valid_shape_m={shape_m}")


def humming_forward(
    config: LayerConfig,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    weight_scale_2: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
    compute_config: dict | str | None = None,
    tuning_config: dict | list | str | None = None,
    hadamard_block_size: int | None = None,
    **kwargs,
) -> torch.Tensor:
    del kwargs
    for name, value in (
        ("zero_point", zero_point),
        ("bias", bias),
        ("weight_scale_2", weight_scale_2),
        ("input_scale", input_scale),
    ):
        if value is not None:
            raise NotImplementedError(f"the indexed W4A16 kernel takes no {name}")
    if hadamard_block_size not in (None, 0, 1):
        raise NotImplementedError("Hadamard-rotated inputs are not supported")
    if weight_scale is None:
        raise TypeError("humming_forward requires weight_scale")
    meta = config.meta
    if meta is None:
        raise TypeError(
            "config was not produced by humming.transform.prepare_layer_config"
        )
    shape_m = valid_shape_m if valid_shape_m > 0 else inputs.size(0) * top_k
    prepared = dispatch.build_prepared(weight, weight_scale, meta)
    return dispatch.forward_w4a16(
        prepared,
        meta,
        inputs,
        outputs=outputs,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
        compute_config=compute_config,
        block_m=_routing_block_m(tuning_config, shape_m),
        validate_routing=False,
        locks=locks,
    )


__all__ = ["humming_forward", "may_quant_input"]
