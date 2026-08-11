"""``HummingMethod``/``HummingLayerMeta`` names, backed by chord's operator.

The re-exported classes are the same objects the ``chord`` facade exposes,
except :class:`HummingMethod`, which adds one vLLM-specific adaptation:

chord tunes the w13 and w2 projections independently, so their per-bracket
``block_m`` can diverge (H200 EP8 prefill and Blackwell EP8 decode both have
such brackets).  vLLM builds the indexed routing once with
``moe_align_block_size`` using the w13 table's block size and routes both
projections through it without passing ``block_m`` — so the w2 table returned
by :meth:`HummingMethod.get_default_tuning_configs` has each row's M tile
re-mapped onto the w13 table's ``block_m`` for that bracket while keeping the
w2 row's own N/K tile, stage count and stream-K choice.  That is exactly the
adjustment chord applies when a framework passes an explicit ``block_m`` to
``forward_layer``; doing it in the published table keeps the framework's
existing call shape correct with no new argument.
"""

from __future__ import annotations

from chord.layer import (
    HummingLayer,
    HummingModule,
    get_default_f16_torch_dtype,
)
from chord_kernels.operator.api import IndexedKernelConfig
from chord_kernels.operator.dispatch import _config_with_block_m
from chord_kernels.operator.layer import (
    IndexedW4A16Method,
    unpack_packed_uint4,
)
from chord_kernels.operator.profiles import (
    IndexedLayerMeta,
    IndexedLayerProfile,
    select_indexed_profile,
)

HummingLayerMeta = IndexedLayerMeta
HummingLayerProfile = IndexedLayerProfile


def _covering_row(rows: list, valid_shape_m: int) -> dict:
    for lower, upper, row in rows:
        if valid_shape_m > lower and valid_shape_m <= upper:
            return row
    raise ValueError(f"no tuning row covers valid_shape_m={valid_shape_m}")


def _align_w2_block_m_to_w13(
    w2_rows: list[tuple[int, int, dict]],
    w13_rows: list[tuple[int, int, dict]],
) -> list[tuple[int, int, dict]]:
    """Re-map each w2 row's block-M onto the w13 routing alignment.

    Both tables share bracket boundaries after this pass, and every w2 row's
    ``block_m`` equals the w13 row covering the same routed-M — the same
    result as an explicit ``block_m`` forward argument, without one.
    """
    boundaries = sorted(
        {upper for _, upper, _ in w2_rows} | {upper for _, upper, _ in w13_rows}
    )
    aligned_rows: list[tuple[int, int, dict]] = []
    lower = 0
    for upper in boundaries:
        w2_row = _covering_row(w2_rows, upper)
        w13_block_m = _covering_row(w13_rows, upper)["block_shape"][0]
        layout = w2_row.get("layout", "mma")
        config = IndexedKernelConfig.from_dict(w2_row)
        aligned = _config_with_block_m(config, w13_block_m, layout=layout)
        row = aligned.to_dict()
        row.update({"block_m": aligned.block_m, "layout": layout})
        if aligned_rows and aligned_rows[-1][2] == row:
            aligned_rows[-1] = (aligned_rows[-1][0], upper, aligned_rows[-1][2])
        else:
            aligned_rows.append((lower, upper, row))
        lower = upper
    return aligned_rows


class HummingMethod(IndexedW4A16Method):
    """Drop-in replacement for ``humming.layer.HummingMethod`` (vLLM contract)."""

    @classmethod
    def get_default_tuning_configs(
        cls,
        layer,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        use_m_major_input_scale: bool = False,
        gemm_type: object = "indexed",
        sublayer_name: str = "",
        **kwargs,
    ):
        rows = super().get_default_tuning_configs(
            layer,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
            gemm_type=gemm_type,
            sublayer_name=sublayer_name,
            **kwargs,
        )
        if sublayer_name != "w2":
            return rows
        metas = getattr(layer, "humming_metas", None)
        w13_meta = metas.get("w13") if isinstance(metas, dict) else None
        if not isinstance(w13_meta, IndexedLayerMeta):
            return rows
        w13_rows = super().get_default_tuning_configs(
            layer,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
            gemm_type=gemm_type,
            sublayer_name="w13",
            **kwargs,
        )
        return _align_w2_block_m_to_w13(rows, w13_rows)


__all__ = [
    "HummingLayer",
    "HummingLayerMeta",
    "HummingLayerProfile",
    "HummingMethod",
    "HummingModule",
    "get_default_f16_torch_dtype",
    "select_indexed_profile",
    "unpack_packed_uint4",
]
