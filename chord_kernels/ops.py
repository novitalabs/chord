"""Public call site for the indexed W4A16 MoE operator."""

from __future__ import annotations

from typing import Any


def indexed(
    activation: Any,
    weight: Any,
    sorted_ids: Any,
    expert_ids: Any,
    num_tokens_padded: Any,
    top_k: int,
    *,
    outputs: Any | None = None,
    block_m: int | None = None,
    layout: str | None = None,
    swap_ab: bool | None = None,
    config: Any | None = None,
    valid_shape_m: int = 0,
    validate_routing: bool = True,
) -> Any:
    """Run the indexed BF16 x INT4 MoE operator."""
    from .operator import w4a16_indexed

    call_kwargs = dict(
        outputs=outputs,
        block_m=block_m,
        layout=layout,
        swap_ab=swap_ab,
        valid_shape_m=valid_shape_m,
        validate_routing=validate_routing,
    )
    if config is not None:
        call_kwargs["config"] = config
    return w4a16_indexed(
        activation,
        weight,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        top_k,
        **call_kwargs,
    )


__all__ = ["indexed"]
