"""``humming.tune`` for vLLM's functional Humming interface.

``get_heuristics_config`` returns the ``(min_m, max_m, row)`` table vLLM
serializes once per sublayer and replays every step.  vLLM reads the w13
table's ``block_shape[0]`` to build one ``moe_align_block_size`` routing and
routes both projections with it, so each w2 row's M tile is re-mapped onto
the block-M of the w13 row covering the same routed-M (keeping the row's own
N/K tile and stream-K choice) -- the same adjustment the layer-path
``HummingMethod.get_default_tuning_configs`` applies.  The w13 projection is
derived from the w2 one: same profile and experts, ``N = 2 * K_w2`` (gated
gate/up), ``K = N_w2``.
"""

from __future__ import annotations

import dataclasses

from chord_kernels.operator.tuning import _indexed_tuning_rows

from humming.config import GemmType, LayerConfig


def _check_compute_options(gemm_type, use_f16_accum, use_batch_invariant) -> None:
    gemm_type = GemmType(getattr(gemm_type, "value", gemm_type))
    if gemm_type != GemmType.INDEXED:
        raise NotImplementedError(
            "the chord humming shim implements the indexed MoE GEMM only, got "
            f"gemm_type={gemm_type.value!r} (set VLLM_HUMMING_MOE_GEMM_TYPE=indexed)"
        )
    if use_f16_accum:
        raise NotImplementedError("use_f16_accum is not supported (FP32 accumulation)")
    if use_batch_invariant:
        raise NotImplementedError("use_batch_invariant is not supported")


def _is_down_projection(meta) -> bool:
    # gate/up reads the model dim and writes 2x the expert dim; down reads the
    # expert dim and writes the model dim, so N > K identifies down.
    return meta.shape_n > meta.shape_k


def _rows(meta) -> list[tuple[int, int, dict]]:
    return [(lo, hi, dict(row)) for lo, hi, row in _indexed_tuning_rows(meta)]


def get_heuristics_config(
    layer_config: LayerConfig,
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    use_m_major_input_scale: bool = False,
    gemm_type="dense",
):
    """Return chord's tuning rows for this layer, or one row for ``shape_m``."""
    from humming.layer import _align_w2_block_m_to_w13

    del use_m_major_input_scale
    _check_compute_options(gemm_type, use_f16_accum, use_batch_invariant)
    meta = layer_config.meta
    if meta is None:
        raise TypeError(
            "layer_config was not produced by humming.transform.prepare_layer_config"
        )
    rows = _rows(meta)
    if _is_down_projection(meta):
        w13 = dataclasses.replace(meta, shape_n=2 * meta.shape_k, shape_k=meta.shape_n)
        rows = [
            (lo, hi, dict(row))
            for lo, hi, row in _align_w2_block_m_to_w13(rows, _rows(w13))
        ]
    if shape_m is None:
        return rows
    for lower, upper, row in rows:
        if lower < shape_m <= upper:
            return row
    raise ValueError(f"no indexed tuning row covers shape_m={shape_m}")


__all__ = ["get_heuristics_config"]
