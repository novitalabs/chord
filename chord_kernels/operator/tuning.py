# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

"""Launch-schedule tables for the indexed backend.

The published tuning results for the indexed profiles live here: a
routed-M in, an :class:`IndexedKernelConfig` out.  The grouped backends draw no
tiles from this module — their tile selection is owned by the grouped SM90
heuristic at dispatch time — so the only grouped code here is the
compute-config gate that keeps a framework from requesting one grouped mode
against a buffer packed for the other.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Mapping

from chord_kernels.operator.api import IndexedKernelConfig, _compatible_block_n
from chord_kernels.operator.profiles import IndexedLayerMeta


def _indexed_config(
    meta: IndexedLayerMeta,
    *,
    block_m: int,
    block_n: int,
    block_k: int = 64,
    num_stages: int = 4,
    num_ctas_per_sm: int = 1,
    swap_ab: bool | None = None,
    use_stream_k: bool = False,
) -> IndexedKernelConfig:
    if swap_ab is None:
        swap_ab = meta.profile.swap_ab
    warp_n = 32 if meta.layout == "wgmma" else 64
    # Stream-K (K-dimension split with cross-CTA reduction) is opt-in per config;
    # decode and the small prefill tiles stay one-pass.
    return IndexedKernelConfig(
        block_shape=(block_m, block_n, block_k),
        warp_shape=(block_m, warp_n, min(block_k, 64)),
        num_stages=num_stages,
        num_ctas_per_sm=num_ctas_per_sm,
        swap_ab=swap_ab,
        use_stream_k=use_stream_k,
    )


# WGMMA accumulator register ceiling: block_m=176 already costs ~255 regs per
# thread, so this is the largest tile that stays off the spill cliff.
_H200_PREFILL_MAX_BLOCK_M = 176


def _min_block_count_block_m(valid_shape_m: int, num_experts: int) -> int:
    """Pick the block-M that minimizes total blocks over a modelled routing.

    Each expert's rows are padded to a whole block on its own, so the total
    block count is ``sum(ceil(count_e * 1.1 / block_m))`` rather than
    ``ceil(routed_m / block_m)``.  A single fixed random routing is sampled so
    the estimate tracks a real router rather than an even split.  The seed and
    RNG match upstream Humming's
    ``np.random.RandomState(seed=0).randint(0, num_experts, size=shape_m)``, so
    the chosen block-M is identical for the same shape.
    """

    import numpy as np

    experts = max(num_experts, 1)
    rows = max(valid_shape_m, 1)
    samples = np.random.RandomState(seed=0).randint(0, experts, size=rows)
    counts = np.bincount(samples, minlength=experts)

    best_block_m = 8
    best_blocks = None
    for block_m in range(8, _H200_PREFILL_MAX_BLOCK_M + 1, 8):
        blocks = int(np.ceil(counts * 1.1 / block_m).sum())
        if best_blocks is None or blocks < best_blocks:
            best_blocks, best_block_m = blocks, block_m
    return best_block_m


def _h200_prefill_block_m(valid_shape_m: int, num_experts: int, shape_k: int) -> int:
    """Size prefill block-M from routed tokens per expert.

    The governing quantity is tokens-per-expert, ``tok_e = routed_m /
    num_experts``, not ``routed_m`` alone: each expert's rows are padded up to a
    whole block independently, so two layers with the same routed_m but
    different expert counts want different tiles.

    For the EP-scale projections (N >= 4096, K > 512) the output is wide enough
    that ``N / block_n >= 16`` tiles keep the grid full even at large block-M, so
    per-expert M-padding dominates: split each expert's padded rows into the
    fewest blocks that fit under the 176 register ceiling, then size block-M to
    just cover them.  Past two blocks, deep-K shapes get one more 176 window
    because the long K loop amortizes the register spill, while short-K shapes
    settle on 128.

    Below ``tok_e`` 80 the padding model above would undershoot, so block-M comes
    from minimizing the total block count instead -- the regime there is block
    count and occupancy, not per-expert padding.
    """

    tokens_per_expert = valid_shape_m / max(num_experts, 1)
    if tokens_per_expert < 80:
        return _min_block_count_block_m(valid_shape_m, num_experts)

    # 1.1x covers the per-expert overshoot a random router leaves in each block.
    padded = tokens_per_expert * 1.1
    num_blocks = math.ceil(padded / _H200_PREFILL_MAX_BLOCK_M)
    if num_blocks <= 2:
        return math.ceil(padded / num_blocks / 8) * 8
    # Only the deep-K gate/up projection wins from a third 176 window; the
    # short-K down projection regresses there, so it goes straight to 128.
    if tokens_per_expert <= 352 and shape_k >= 4096:
        return _H200_PREFILL_MAX_BLOCK_M
    return 128


def _h200_tp8_block_m(valid_shape_m: int, num_experts: int) -> int:
    """Size TP8 block-M from routed tokens per expert.

    TP8 slices ``moe_intermediate`` instead of the expert list, so both
    projections are narrow in the dimension that decides how many n-tiles a
    single m-block covers: gate/up is ``N=512`` and down is ``K=256``.  With
    ``N / block_n`` as low as 2 the grid is not kept full by output width the way
    the EP8 shapes are, so total block count and occupancy dominate instead of
    per-expert M-padding, and the EP8 padding model would over-grow block-M into
    the register cliff.

    The windows are therefore flatter than EP8's: below ``tok_e`` 128 one block
    per expert still pays (block-M sized to the padded rows), then a 96 window to
    ``tok_e`` 190, then 128.  The whole 96..144 band beats the block-count argmin
    the EP8 path falls back to, because register pressure and occupancy — not
    block count — set the limit here.

    Below ``tok_e`` 80 the model would undershoot for the same reason it does on
    EP8, so block-M comes from minimizing total block count instead.
    """

    tokens_per_expert = valid_shape_m / max(num_experts, 1)
    if tokens_per_expert < 80:
        return _min_block_count_block_m(valid_shape_m, num_experts)
    if tokens_per_expert <= 128:
        # 1.1x covers the per-expert overshoot a random router leaves behind.
        return round(tokens_per_expert * 1.1 / 8) * 8
    if tokens_per_expert <= 190:
        return 96
    return 128


def _h200_tp8_use_stream_k(valid_shape_m: int, shape_k: int) -> bool:
    """Decide whether TP8 splits the K dimension across CTAs.

    The two projections cross in opposite directions at ``routed_m`` 65536.  Down
    (``K=256``) has only 4 K-blocks to split, so it stays one-pass until the
    workload is imbalanced enough for the load balancing to repay the locks.
    Gate/up (``K=7168``) is the mirror image: the split pays until the M*N tiles
    fill the grid on their own, after which it is pure overhead.
    """

    if shape_k <= 512:
        return valid_shape_m > 65536
    return valid_shape_m < 65536


def _h200_prefill_use_stream_k(valid_shape_m: int, shape_n: int, shape_k: int) -> bool:
    """Decide whether prefill splits the K dimension across CTAs.

    Stream-K helps the deep-K gate/up projection at every prefill size, but the
    mid-K down projection (512 < K < 4096) regresses once routed_m is large: the
    M*N tiles already fill the grid, so the K-split only adds reduction and lock
    overhead.  Below that crossover stream-K still balances the tail, so it
    stays on.
    """

    if shape_n >= 4096 and 512 < shape_k < 4096 and valid_shape_m >= 5120:
        return False
    return True


def _select_indexed_kernel_config(
    meta: IndexedLayerMeta, valid_shape_m: int = 0
) -> IndexedKernelConfig:
    """Resolve the small, published schedule table for one routed M.

    The table deliberately covers only the published target profiles.
    Unrecognised projection shapes use the conservative profile fallback instead
    of silently claiming a tuning result for a different model.
    """
    if isinstance(valid_shape_m, bool) or not isinstance(valid_shape_m, int):
        raise TypeError(
            f"valid_shape_m must be an int, got {type(valid_shape_m).__name__}"
        )
    if valid_shape_m < 0:
        raise ValueError(f"valid_shape_m must be non-negative, got {valid_shape_m}")
    m = max(valid_shape_m, 1)
    profile = meta.profile.name
    n, k, experts = meta.shape_n, meta.shape_k, meta.num_experts

    if profile == "h200_prefill_ep8" and (n, k) in {
        (4096, 7168),
        (7168, 2048),
    }:
        block_m = _h200_prefill_block_m(m, experts, k)
        use_stream_k = _h200_prefill_use_stream_k(m, n, k)
        if block_m <= 32:
            # Small blocks split by output width: the narrow gate/up
            # (N <= 4096) trades tile width for K depth, while the wide down
            # projection keeps block-N 256 and only doubles K.  Swapping the
            # two costs the down projection 13-16% at routed_m 112..856.
            if n <= 4096:
                return _indexed_config(
                    meta, block_m=block_m, block_n=128, block_k=256,
                    use_stream_k=use_stream_k,
                )
            return _indexed_config(
                meta, block_m=block_m, block_n=256, block_k=128,
                use_stream_k=use_stream_k,
            )
        # Above block-M 32 both projections use the wide 256x64 tile.  For
        # gate/up block-M 40..64 a generic N<=4096 heuristic would pick
        # 128x128, but 256x64 with the 2-CTA window below measures 12-15%
        # faster on H200 for these EP8 shapes (routed_m 863..2073).
        #
        # Forcing 2 CTAs/SM for the 40..80 block-M window at block_n=256
        # doubles resident warps to hide the cp.async + dequant latency.
        num_ctas_per_sm = 2 if 40 <= block_m <= 80 else 1
        return _indexed_config(
            meta, block_m=block_m, block_n=256, block_k=64,
            num_ctas_per_sm=num_ctas_per_sm,
            use_stream_k=use_stream_k,
        )

    if profile == "h200_tp8" and (n, k) in {
        (512, 7168),
        (7168, 256),
    }:
        block_m = _h200_tp8_block_m(m, experts)
        use_stream_k = _h200_tp8_use_stream_k(m, k)
        # A short m-block leaves the mainloop too little work per tile, so block-K
        # compensates and relaxes as block-M grows.  down starts one notch
        # shallower throughout because K=256 has less depth to spend.
        if k <= 512:
            block_k = 128 if block_m <= 32 else 64
            # down (N=7168 K=256).  At the generic block_n=256 the WGMMA
            # accumulator plus B-smem are too large to fit two CTAs on an SM,
            # pinning occupancy at 1 CTA/SM, and the short K leaves the mainloop
            # latency-bound rather than compute-bound.  Halving block_n to 128
            # halves both and unlocks 2 CTAs/SM, hiding the cp.async + dequant
            # latency.
            return _indexed_config(
                meta, block_m=block_m, block_n=128, block_k=block_k,
                num_ctas_per_sm=2,
                use_stream_k=use_stream_k,
            )
        # gate/up (N=512 K=7168).  A 128-wide tile already covers this narrow
        # output in 4 n-blocks, so the wide 256 tile only pays once block-M
        # passes 64 and the deep-K mainloop has enough rows per tile to feed it.
        block_k = 256 if block_m <= 32 else (128 if block_m <= 64 else 64)
        block_n = 128 if block_m <= 64 else 256
        return _indexed_config(
            meta, block_m=block_m, block_n=block_n, block_k=block_k,
            use_stream_k=use_stream_k,
        )

    if profile == "h200_decode_ep8" and (n, k) in {
        (4096, 7168),
        (7168, 2048),
    }:
        tokens_per_expert = m / max(experts, 1) / 0.9
        block_m = (
            8 if tokens_per_expert <= 6 else (16 if tokens_per_expert <= 13 else 24)
        )
        return _indexed_config(
            meta,
            block_m=block_m,
            block_n=256,
            num_ctas_per_sm=4,
            swap_ab=True,
        )

    if profile == "blackwell_decode_ep8" and (n, k, experts) in {
        (4096, 7168, 48),
        (7168, 2048, 48),
    }:
        # SM100 and SM103 share this table.  The two projections are tuned
        # independently; a host that shares one routing across both passes the
        # routing's block-M through the ``block_m`` forward argument, which
        # overrides the row's M tile while keeping the projection's N/K tile
        # (see _config_with_block_m).
        if (n, k) == (7168, 2048):
            # down is short-K (32 blocks of 64): the swap-AB padding-skip
            # tiles stay one-pass, and the K-split only repays on the large
            # non-swap tiles.
            if m <= 272:
                return _indexed_config(
                    meta, block_m=8, block_n=256, num_ctas_per_sm=4,
                    swap_ab=True,
                )
            if m <= 848:
                return _indexed_config(
                    meta, block_m=16, block_n=256, num_ctas_per_sm=4,
                    swap_ab=True,
                )
            if m <= 1424:
                return _indexed_config(
                    meta, block_m=32, block_n=512, swap_ab=False,
                    use_stream_k=True,
                )
            return _indexed_config(
                meta, block_m=48, block_n=512, swap_ab=False,
                use_stream_k=True,
            )

        # gate/up is deep-K (112 blocks of 64), which leaves a long scheduling
        # tail, so stream-K repays its lock overhead at every decode size.
        if m <= 320:
            # A block holds one populated 8-token mma-N tile; the small
            # swap-AB tile at 4 CTAs/SM keeps the machine latency-bound
            # rather than padding-bound.
            return _indexed_config(
                meta, block_m=8, block_n=256, num_ctas_per_sm=4,
                swap_ab=True, use_stream_k=True,
            )
        if m <= 736:
            # The second token tile fills; the wide non-swap tile with a
            # deep K block wins over both the swap rows and (16,512,64).
            return _indexed_config(
                meta, block_m=16, block_n=512, block_k=128,
                swap_ab=False, use_stream_k=True,
            )
        if m <= 1424:
            return _indexed_config(
                meta, block_m=32, block_n=512, swap_ab=False,
                use_stream_k=True,
            )
        return _indexed_config(
            meta, block_m=48, block_n=512, swap_ab=False,
            use_stream_k=True,
        )

    return IndexedKernelConfig.default(
        layout=meta.layout,
        block_m=meta.profile.block_m,
        block_n=_compatible_block_n(meta.shape_n),
        swap_ab=meta.profile.swap_ab,
    )


def _indexed_tuning_rows(meta: IndexedLayerMeta) -> list[tuple[int, int, dict]]:
    """Return ``(min_m, max_m, config)`` rows exactly matching the resolver.

    Every routed-M is swept through the same resolver the forward path uses and
    runs of equal configs are merged into rows.  Boundaries are therefore
    discovered, never hand-maintained, and a framework that aligns routing to a
    row's block-M is guaranteed the forward path launches that same tile for
    every routed-M inside the row.
    """
    if meta.profile.is_grouped:
        raise ValueError(
            "grouped-backend layers dispatch with the SM90 heuristic at launch "
            "time and have no indexed tuning rows"
        )
    rows = [
        (lower, upper, dict(config))
        for lower, upper, config in _swept_tuning_rows(meta)
    ]
    return rows


@functools.lru_cache(maxsize=64)
def _swept_tuning_rows(
    meta: IndexedLayerMeta,
) -> tuple[tuple[int, int, dict], ...]:
    # The last schedule change across the published profiles is prefill's
    # 176 -> 128 block-M step at tok_e = 352, i.e. routed_m = 352 * experts;
    # decode settles by tok_e ~ 13 and Blackwell by routed_m 1424.  The
    # sentinels then assert the final row really is constant.
    sweep_bound = max(4096, 368 * max(meta.num_experts, 1))

    rows: list[list] = []
    last_config: IndexedKernelConfig | None = None
    for m in range(1, sweep_bound + 1):
        config = _select_indexed_kernel_config(meta, m)
        if config == last_config:
            rows[-1][1] = m
        else:
            rows.append([m - 1, m, config])
            last_config = config
    rows[-1][1] = 1 << 30

    for sentinel in (2 * sweep_bound, 1 << 20, 1 << 29):
        if _select_indexed_kernel_config(meta, sentinel) != last_config:
            raise RuntimeError(
                "indexed tuning sweep bound is too small: the schedule still "
                f"changes at routed_m={sentinel} for profile "
                f"{meta.profile.name!r} (N={meta.shape_n}, K={meta.shape_k})"
            )

    frozen: list[tuple[int, int, dict]] = []
    for lower, upper, config in rows:
        row = config.to_dict()
        row.update({"block_m": config.block_m, "layout": meta.layout})
        frozen.append((lower, upper, row))
    return tuple(frozen)


def _validate_indexed_compute_config(compute_config: object) -> None:
    """Reject a framework compute config that selects another GEMM family."""
    value = compute_config
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("compute_config must be valid JSON") from exc
    if isinstance(value, Mapping):
        gemm_type = value.get("gemm_type")
        unsupported = [
            name
            for name in ("use_f16_accum", "use_batch_invariant")
            if bool(value.get(name, False))
        ]
    else:
        gemm_type = getattr(value, "gemm_type", None)
        unsupported = [
            name
            for name in ("use_f16_accum", "use_batch_invariant")
            if bool(getattr(value, name, False))
        ]
    if unsupported:
        raise ValueError(
            "indexed W4A16 does not implement compute option(s): "
            + ", ".join(unsupported)
        )
    if gemm_type is None:
        return
    gemm_type = getattr(gemm_type, "value", gemm_type)
    if str(gemm_type).lower() != "indexed":
        raise ValueError(
            "the indexed W4A16 layer only accepts compute_config gemm_type='indexed'"
        )


def _validate_grouped_compute_config(
    compute_config: object | None, mode: str
) -> None:
    """Gate a grouped-backend forward on its packed mode.

    Accepts the framework spellings ``grouped_masked`` / ``grouped_contiguous``
    (matching upstream Humming's ``GemmType`` values) and requires them to
    agree with the mode that was packed into the weight.  The reorder perm
    width is baked into the buffer, so a mismatch is a silent wrong answer
    rather than a slow path.
    """
    if compute_config is None:
        return
    value = compute_config
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("compute_config must be valid JSON") from exc
    if isinstance(value, Mapping):
        gemm_type = value.get("gemm_type")
        unsupported = [
            name
            for name in ("use_f16_accum", "use_batch_invariant")
            if bool(value.get(name, False))
        ]
    else:
        gemm_type = getattr(value, "gemm_type", None)
        unsupported = [
            name
            for name in ("use_f16_accum", "use_batch_invariant")
            if bool(getattr(value, name, False))
        ]
    if unsupported:
        raise ValueError(
            "grouped W4A16 does not implement compute option(s): "
            + ", ".join(unsupported)
        )
    if gemm_type is None:
        return
    gemm_type = str(getattr(gemm_type, "value", gemm_type)).lower()
    required = "grouped_masked" if mode == "masked" else "grouped_contiguous"
    if gemm_type != required:
        raise ValueError(
            f"the packed weight is grouped-{mode} but compute_config requests "
            f"gemm_type={gemm_type!r}; repack or pass {required!r}"
        )


def _resolve_tuning_config(
    meta: IndexedLayerMeta,
    selected_shape_m: int,
    tuning_config: object | None,
    *,
    default: IndexedKernelConfig,
) -> IndexedKernelConfig:
    """Parse the small tuning format used by Humming/vLLM."""
    if tuning_config is None:
        return default
    value = tuning_config
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tuning_config must be valid JSON") from exc
    if isinstance(value, Mapping):
        row = dict(value)
        row.setdefault("swap_ab", meta.swap_ab)
        if row.get("layout", meta.layout) != meta.layout:
            raise ValueError("tuning_config layout does not match the packed weight")
        return IndexedKernelConfig.from_dict(row)
    if isinstance(value, list):
        for lower, upper, row in value:
            if selected_shape_m > int(lower) and selected_shape_m <= int(upper):
                row = dict(row)
                row.setdefault("swap_ab", meta.swap_ab)
                if row.get("layout", meta.layout) != meta.layout:
                    raise ValueError(
                        "tuning_config layout does not match the packed weight"
                    )
                return IndexedKernelConfig.from_dict(row)
        raise ValueError(
            f"no indexed tuning row covers valid_shape_m={selected_shape_m}"
        )
    raise TypeError(
        "tuning_config must be a JSON string, dict, list, or None; "
        f"got {type(tuning_config).__name__}"
    )


__all__ = [
    "_h200_tp8_block_m",
    "_h200_tp8_use_stream_k",
    "_indexed_tuning_rows",
    "_resolve_tuning_config",
    "_select_indexed_kernel_config",
    "_validate_grouped_compute_config",
    "_validate_indexed_compute_config",
]
