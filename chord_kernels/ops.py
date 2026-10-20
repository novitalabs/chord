"""Public W4A16 MoE operators."""

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
    locks: Any | None = None,
    outputs: Any | None = None,
    block_m: int | None = None,
    validate_routing: bool = False,
) -> Any:
    """Indexed (``moe_align_block_size``-routed) BF16 x INT4 MoE GEMM.

    ``activation`` is BF16 ``[M, K]`` and ``weight`` comes from
    ``pack_w4a16``.  The routing must be aligned to
    ``indexed_block_m(weight, num_tokens, top_k)``; a host that aligns one
    routing for both projections passes that block size as ``block_m``.
    ``locks`` is a zeroed int32 tensor of 1024 words (needed when the tuned
    schedule uses stream-K).  Every launch indexes it from word 0 and leaves it
    zeroed, so it is reused without a reset by launches that are ordered, e.g.
    on one stream or within one graph; launches that may run concurrently (on
    different streams, or graphs replayed on different streams) each need their
    own buffer.  Returns BF16 ``[M * top_k, N]``.  For the down projection pass
    the routed activations and ``top_k=1``.

    The routing is not checked: a routing that breaks the contract in
    docs/getting_started.md gives wrong results or out-of-bounds reads without
    an error.  ``validate_routing=True`` checks it, at the cost of a device
    synchronization, for debugging.
    """
    from .operator import w4a16_indexed

    return w4a16_indexed(
        activation, weight, sorted_ids, expert_ids, num_tokens_padded, top_k,
        locks=locks, outputs=outputs, block_m=block_m,
        validate_routing=validate_routing,
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
    """Grouped masked (decode) GEMM, SM90.

    ``activation`` is BF16 ``[G, max_m, K]`` (or flat ``[G*max_m, K]``),
    ``weight`` comes from ``pack_w4a16``, ``masked_m`` is the valid row
    count per expert ``[G] int32`` and ``expected_m`` its typical value.
    Returns flat ``[G*max_m, N]``.
    """
    from .operator.grouped.api import w4a16_masked

    return w4a16_masked(
        activation, weight, masked_m, expected_m, outputs=outputs, enable_pdl=enable_pdl
    )


def contiguous(
    activation: Any,
    weight: Any,
    m_indices: Any,
    *,
    outputs: Any | None = None,
    enable_pdl: bool = False,
) -> Any:
    """Grouped contiguous (prefill) GEMM, SM90.

    ``activation`` is BF16 ``[m, K]`` with each expert's rows padded to a
    128-row boundary (padding zeroed); ``m_indices`` ``[m] int32`` names each
    row's expert (-1 for padding).  Returns ``[m, N]``.
    """
    from .operator.grouped.api import w4a16_contiguous

    return w4a16_contiguous(
        activation, weight, m_indices, outputs=outputs, enable_pdl=enable_pdl
    )


__all__ = ["contiguous", "indexed", "masked"]
