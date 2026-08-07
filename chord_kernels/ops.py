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


def masked(
    activation: Any,
    weight: Any,
    masked_m: Any,
    expected_m: int,
    *,
    outputs: Any | None = None,
    enable_pdl: bool = False,
) -> Any:
    """Run the grouped masked (decode) grouped W4A16 operator.

    ``activation`` is ``[G*max_m, K]`` flat or ``[G, max_m, K]`` BF16, the
    weight comes from ``pack_w4a16_grouped(mode="masked")``, and ``masked_m``
    is the per-expert valid token count ``[G] int32``.  Returns flat
    ``[G*max_m, N]``.
    """
    from .operator.grouped.api import w4a16_masked

    return w4a16_masked(
        activation, weight, masked_m, expected_m, outputs=outputs,
        enable_pdl=enable_pdl,
    )


def contiguous(
    activation: Any,
    weight: Any,
    m_indices: Any,
    *,
    outputs: Any | None = None,
    enable_pdl: bool = False,
) -> Any:
    """Run the grouped contiguous (prefill) grouped W4A16 operator.

    ``activation`` is the grouped-native ``[m, K]`` layout (per-expert rows
    padded to 128, padding rows zeroed) with ``m_indices`` ``[m] int32``
    selecting the expert (-1 for padding).  The weight comes from
    ``pack_w4a16_grouped(mode="contiguous")``.  Returns ``[m, N]``.
    """
    from .operator.grouped.api import w4a16_contiguous

    return w4a16_contiguous(
        activation, weight, m_indices, outputs=outputs, enable_pdl=enable_pdl
    )


__all__ = ["contiguous", "indexed", "masked"]
