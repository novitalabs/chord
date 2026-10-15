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
    num_stages: int = 4,  # cp.async ring depth
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

    counts = _sampled_expert_counts(valid_shape_m, num_experts)
    best_block_m = 8
    best_blocks = None
    for block_m in range(8, _H200_PREFILL_MAX_BLOCK_M + 1, 8):
        blocks = int(np.ceil(counts * 1.1 / block_m).sum())
        if best_blocks is None or blocks < best_blocks:
            best_blocks, best_block_m = blocks, block_m
    return best_block_m


# Fixed per-block cost, in rows, of the row-work model below: one m-block pays
# for its rows plus this much routing load, prologue and epilogue.
_ROW_WORK_BLOCK_OVERHEAD = 32

# The block-M window of the row-work model served by the wide mid-size tiles.
# Below it the small-block argmin and tiles stay in charge; above it (the
# model's 88/96 picks just under tok_e 80) the 512-wide gate/up tile runs
# 30-50% slower than the 256x64 stream-K tiles, so those chunks keep them.
_MID_TILE_MIN_BLOCK_M = 32
_MID_TILE_MAX_BLOCK_M = 80


# The legacy ``RandomState.randint`` stream is prefix-stable: the first m
# draws of size=N equal the draws of size=m.  So one stream per expert count
# serves every routed-M up to this bound (the tuning-row sweep's reach).
_SAMPLED_ROUTING_ROWS = 1 << 18
_SAMPLED_ROUTING_STRIDE = 1024


@functools.lru_cache(maxsize=8)
def _sampled_routing(num_experts: int):
    """The seeded expert stream plus cumulative per-expert counts every
    ``_SAMPLED_ROUTING_STRIDE`` rows, for :func:`_sampled_expert_counts`."""

    import numpy as np

    samples = np.random.RandomState(seed=0).randint(
        0, num_experts, size=_SAMPLED_ROUTING_ROWS
    )
    blocks = samples.reshape(-1, _SAMPLED_ROUTING_STRIDE)
    offsets = blocks + num_experts * np.arange(blocks.shape[0])[:, None]
    per_block = np.bincount(
        offsets.ravel(), minlength=blocks.shape[0] * num_experts
    ).reshape(blocks.shape[0], num_experts)
    checkpoints = np.concatenate(
        [np.zeros((1, num_experts), dtype=per_block.dtype),
         np.cumsum(per_block, axis=0)]
    )
    return samples, checkpoints


@functools.lru_cache(maxsize=1024)
def _sampled_expert_counts(valid_shape_m: int, num_experts: int):
    """Per-expert row counts of the seeded routing the block-M models share.

    Identical to ``np.bincount(RandomState(0).randint(0, E, size=m))``, the
    upstream draw.  Cached: the mid-window gate, block-M and stream-K
    decisions each consult it for the same routed-M, and the tuning-row sweep
    walks every routed-M.  The returned array is read-only.
    """

    import numpy as np

    experts = max(num_experts, 1)
    rows = max(valid_shape_m, 1)
    if rows > _SAMPLED_ROUTING_ROWS:
        samples = np.random.RandomState(seed=0).randint(0, experts, size=rows)
        counts = np.bincount(samples, minlength=experts)
    else:
        samples, checkpoints = _sampled_routing(experts)
        whole = rows // _SAMPLED_ROUTING_STRIDE
        tail = samples[whole * _SAMPLED_ROUTING_STRIDE:rows]
        counts = checkpoints[whole] + np.bincount(tail, minlength=experts)
    counts.setflags(write=False)
    return counts


@functools.lru_cache(maxsize=1024)
def _min_row_work_block_m(valid_shape_m: int, num_experts: int) -> int:
    """Pick the block-M that minimizes modelled row work over a routing.

    Each block costs its padded rows plus a fixed per-block overhead, so the
    total is ``sum(ceil(count_e / block_m)) * (block_m + 32)``.  Unlike the
    block-count argmin this charges the padding a tall tile leaves inside
    every expert, which is what decides the mid-size chunks (``tok_e`` ~20..80)
    once the wide tiles below keep the grid full.  Sampled from the same
    seeded routing as :func:`_min_block_count_block_m`.
    """

    import numpy as np

    counts = _sampled_expert_counts(valid_shape_m, num_experts)
    block_ms = np.arange(8, _H200_PREFILL_MAX_BLOCK_M + 1, 8)
    # Integer ceil-division over every candidate at once; argmin keeps the
    # smallest block-M on ties.
    blocks = (-(-counts[None, :] // block_ms[:, None])).sum(axis=1)
    work = blocks * (block_ms + _ROW_WORK_BLOCK_OVERHEAD)
    return int(block_ms[int(np.argmin(work))])


def _uses_mid_tiles(valid_shape_m: int, num_experts: int) -> bool:
    """True in the mid-size window served by the wide one-pass tiles.

    That is ``tok_e < 80`` where the row-work model asks for block-M 32..80
    (``tok_e`` ~20..70).  Smaller chunks keep the block-count argmin with the
    small-block tiles; larger ones keep the tiles they had before this window
    existed.
    """

    tokens_per_expert = valid_shape_m / max(num_experts, 1)
    if tokens_per_expert >= 80:
        return False
    block_m = _min_row_work_block_m(valid_shape_m, num_experts)
    return _MID_TILE_MIN_BLOCK_M <= block_m <= _MID_TILE_MAX_BLOCK_M


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

    Below ``tok_e`` 80 the padding model above would undershoot.  The mid-size
    window (``tok_e`` ~20..70, see :func:`_uses_mid_tiles`) sizes block-M from
    modelled row work (:func:`_min_row_work_block_m`); the rest of that range
    minimizes the total block count -- the regime there is block count and
    occupancy, not per-expert padding.
    """

    tokens_per_expert = valid_shape_m / max(num_experts, 1)
    if _uses_mid_tiles(valid_shape_m, num_experts):
        return _min_row_work_block_m(valid_shape_m, num_experts)
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


def _h200_tp8_block_m(valid_shape_m: int, num_experts: int, shape_k: int) -> int:
    """Size TP8 block-M from routed tokens per expert.

    TP8 slices ``moe_intermediate`` instead of the expert list, so both
    projections are narrow in the dimension that decides how many n-tiles a
    single m-block covers: gate/up is ``N=512`` and down is ``K=256``.  With
    ``N / block_n`` as low as 1 the grid is not kept full by output width the way
    the EP8 shapes are, so block count and padding have to be traded directly,
    and the EP8 padding model would over-grow block-M.

    - Mid-size window (``tok_e`` ~20..70, see :func:`_uses_mid_tiles`): the
      row-work argmin (:func:`_min_row_work_block_m`).
    - Other chunks below ``tok_e`` 80: the block-count argmin, as EP8.
    - ``tok_e`` 80..128: gate/up keeps the row-work argmin (5-12% over the
      padded-rows size there); down sizes one block per expert to its
      padded rows.
    - ``tok_e`` 128..190: 96, then 128.  The 96..144 band beats the taller
      tiles EP8 grows into, because occupancy and not block count sets the
      limit at this output width.
    """

    tokens_per_expert = valid_shape_m / max(num_experts, 1)
    if _uses_mid_tiles(valid_shape_m, num_experts):
        return _min_row_work_block_m(valid_shape_m, num_experts)
    if tokens_per_expert < 80:
        return _min_block_count_block_m(valid_shape_m, num_experts)
    if tokens_per_expert <= 128:
        if shape_k > 512:
            return _min_row_work_block_m(valid_shape_m, num_experts)
        # 1.1x covers the per-expert overshoot a random router leaves behind.
        return round(tokens_per_expert * 1.1 / 8) * 8
    if tokens_per_expert <= 190:
        return 96
    return 128


# H200 SXM streaming multiprocessors; the wave model below sizes grids for it.
_H200_NUM_SMS = 132


def _wide_tile_has_partial_wave(
    valid_shape_m: int, num_experts: int, block_m: int, shape_n: int
) -> bool:
    """True when the mid-window ``block_m x 512`` one-pass grid leaves its last
    wave less than 90% full.

    The tile count comes from the same seeded routing as the block-M models.
    A partial last wave idles the SMs it does not reach for a whole deep-K
    tile, which the K-split rebalances: measured 4-14% faster at wave
    efficiency 0.76-0.80, and 1-5% slower near full waves.
    """

    import numpy as np

    counts = _sampled_expert_counts(valid_shape_m, num_experts)
    tiles = int(np.ceil(counts / block_m).sum()) * (shape_n // 512)
    waves = -(-tiles // _H200_NUM_SMS)
    return tiles < 0.9 * waves * _H200_NUM_SMS


def _h200_tp8_use_stream_k(valid_shape_m: int, num_experts: int, shape_k: int) -> bool:
    """Decide whether TP8 splits the K dimension across CTAs.

    Down (``K=256``) has only 4 K-blocks to split, so the K-split stays off
    except on the mid-size window's 256-wide tiles from block-M 56, where it
    measured 3-12% faster (and 10-12% slower below block-M 56).  Gate/up
    (``K=7168``) splits everywhere except the mid-size window's 512-wide
    one-pass tile, which covers the whole output per m-block and only splits
    when its grid ends in a partial wave.
    """

    block_m = _h200_tp8_block_m(valid_shape_m, num_experts, shape_k)
    if not _uses_mid_tiles(valid_shape_m, num_experts):
        return shape_k > 512
    if shape_k <= 512:
        return 56 <= block_m <= _TP8_DOWN_WIDE_MAX_BLOCK_M
    return _wide_tile_has_partial_wave(valid_shape_m, num_experts, block_m, 512)


def _h200_prefill_use_stream_k(
    valid_shape_m: int, num_experts: int, shape_n: int, shape_k: int
) -> bool:
    """Decide whether prefill splits the K dimension across CTAs.

    Stream-K helps the deep-K gate/up projection at every prefill size, but the
    mid-K down projection (512 < K < 4096) regresses once routed_m is large: the
    M*N tiles already fill the grid, so the K-split only adds reduction and lock
    overhead.  Below that crossover stream-K still balances the tail, so it
    stays on -- except in the mid-size window, where down's one-pass 2-CTA
    tiles fill the grid on their own and gate/up's 512-wide tile splits only
    when its grid ends in a partial wave (see
    :func:`_wide_tile_has_partial_wave`).
    """

    if _uses_mid_tiles(valid_shape_m, num_experts):
        if shape_k < 4096:
            return False
        block_m = _min_row_work_block_m(valid_shape_m, num_experts)
        return _wide_tile_has_partial_wave(
            valid_shape_m, num_experts, block_m, shape_n
        )
    if shape_n >= 4096 and 512 < shape_k < 4096 and valid_shape_m >= 5120:
        return False
    return True


# Largest block-M at which TP8 down's mid-size 256-wide tile still runs two
# CTAs per SM.
_TP8_DOWN_WIDE_MAX_BLOCK_M = 72

# H200 shared memory per SM, and the per-CTA reservation the driver adds on
# top of a kernel's dynamic allocation.
_H200_SMEM_PER_SM = 233472
_H200_SMEM_RESERVED_PER_CTA = 1024


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _indexed_smem_bytes(block_m: int, block_n: int, block_k: int, num_stages: int) -> int:
    """``sizeof(SharedStorage)`` of ``utils/storage.cuh`` for one tile."""

    stage = (
        _round_up(block_m * block_k * 2, 128)
        + _round_up(block_k * block_n // 2, 128)
        + _round_up(-(-block_k // 32) * block_n * 2, 128)
    )
    # The stage ring shares a 1024-aligned union with the epilogue buffer.
    union = _round_up(max(num_stages * stage, block_m * block_n * 2), 1024)
    return _round_up(union + block_m * 8, 1024)


def _h200_num_stages(block_m: int, block_n: int, block_k: int, num_ctas_per_sm: int) -> int:
    """Deepest cp.async ring (6 stages, else 5, else the default 4) that keeps
    the requested CTAs resident.

    A deeper ring hides more of the gather + dequant latency: 1-3% on the
    large prefill and TP8 tiles.  It must never cost a resident CTA: TP8 down
    ``128x128x64`` at 2 CTAs/SM takes 5 stages, where 6 would leave room for
    only one CTA and run 42% slower.
    """

    for num_stages in (6, 5):
        per_cta = (
            _indexed_smem_bytes(block_m, block_n, block_k, num_stages)
            + _H200_SMEM_RESERVED_PER_CTA
        )
        if num_ctas_per_sm * per_cta <= _H200_SMEM_PER_SM:
            return num_stages
    return 4


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
        use_stream_k = _h200_prefill_use_stream_k(m, experts, n, k)
        if _uses_mid_tiles(m, experts):
            # Mid-size window (tok_e ~20..70): one-pass wide tiles.  The
            # deep-K gate/up takes the full 512 output width per CTA (half the
            # activation gathers of 256x64 at 2 CTAs/SM); the short-K down
            # keeps 256x64 at 2 CTAs/SM, which still covers its latency.
            if k >= 4096:
                return _indexed_config(
                    meta, block_m=block_m, block_n=512, block_k=64,
                    num_stages=_h200_num_stages(block_m, 512, 64, 1),
                    use_stream_k=use_stream_k,
                )
            return _indexed_config(
                meta, block_m=block_m, block_n=256, block_k=64,
                num_stages=_h200_num_stages(block_m, 256, 64, 2),
                num_ctas_per_sm=2,
            )
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
        # Otherwise both projections use the wide 256x64 tile.  For gate/up
        # block-M 40..64 a generic N<=4096 heuristic would pick 128x128, but
        # 256x64 with the 2-CTA window measures 12-15% faster on H200 for
        # these EP8 shapes.  Forcing 2 CTAs/SM for the 40..80 block-M window
        # doubles resident warps to hide the cp.async + dequant latency; the
        # 1-CTA tiles from tok_e 80 take the deepest ring that fits.
        if 40 <= block_m <= 80:
            return _indexed_config(
                meta, block_m=block_m, block_n=256, block_k=64,
                num_ctas_per_sm=2, use_stream_k=use_stream_k,
            )
        # The one-pass down rows past the stream-K crossover keep the
        # default ring (6 stages measured 1.0-1.4% slower there).
        deep_ring = use_stream_k or k >= 4096
        return _indexed_config(
            meta, block_m=block_m, block_n=256, block_k=64,
            num_stages=_h200_num_stages(block_m, 256, 64, 1) if deep_ring else 4,
            use_stream_k=use_stream_k,
        )

    if profile == "h200_tp8" and (n, k) in {
        (512, 7168),
        (7168, 256),
    }:
        block_m = _h200_tp8_block_m(m, experts, k)
        use_stream_k = _h200_tp8_use_stream_k(m, experts, k)
        mid = _uses_mid_tiles(m, experts)
        if k <= 512:
            # down (N=7168 K=256).  The short K leaves the mainloop
            # latency-bound, so it always runs 2 CTAs/SM.  In the mid-size
            # window a 256-wide tile still fits two CTAs and halves the
            # activation re-reads; elsewhere block_n 128 keeps the WGMMA
            # accumulator plus B-smem small enough for 2 CTAs/SM (at 256 the
            # tall tiles pin occupancy at 1 CTA/SM).
            if mid and block_m <= _TP8_DOWN_WIDE_MAX_BLOCK_M:
                # The split rows keep the default 4-deep ring: with the
                # FP32 workspace in play 6 stages measured 3% slower.
                return _indexed_config(
                    meta, block_m=block_m, block_n=256, block_k=64,
                    num_stages=(
                        4 if use_stream_k
                        else _h200_num_stages(block_m, 256, 64, 2)
                    ),
                    num_ctas_per_sm=2, use_stream_k=use_stream_k,
                )
            if block_m <= 32:
                # Smallest chunks: K=256 is two 128-deep blocks.
                return _indexed_config(
                    meta, block_m=block_m, block_n=128, block_k=128,
                    num_ctas_per_sm=2, use_stream_k=use_stream_k,
                )
            return _indexed_config(
                meta, block_m=block_m, block_n=128, block_k=64,
                num_stages=_h200_num_stages(block_m, 128, 64, 2),
                num_ctas_per_sm=2, use_stream_k=use_stream_k,
            )
        if mid:
            # gate/up (N=512 K=7168) mid-size window: a single 512-wide
            # one-pass tile covers the whole output per m-block, so the
            # activation tile is gathered once instead of 2-4 times.
            return _indexed_config(
                meta, block_m=block_m, block_n=512, block_k=64,
                num_stages=_h200_num_stages(block_m, 512, 64, 1),
                use_stream_k=use_stream_k,
            )
        if block_m <= 64:
            # Smallest chunks: a 128-wide tile already covers this narrow
            # output in 4 n-blocks, and block-K compensates for the short
            # m-block.
            return _indexed_config(
                meta, block_m=block_m, block_n=128,
                block_k=256 if block_m <= 32 else 128,
                use_stream_k=use_stream_k,
            )
        return _indexed_config(
            meta, block_m=block_m, block_n=256, block_k=64,
            num_stages=_h200_num_stages(block_m, 256, 64, 1),
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
        # 3 CTAs/SM rather than 4: on the deep-K gate/up the fourth CTA
        # adds more smem and register pressure than latency cover (4-8%
        # slower at every decode size), and on down from block-M 16 (1%).
        # Only down's block-M 8 tile still gains from the fourth CTA.
        num_ctas_per_sm = 4 if k < 4096 and block_m == 8 else 3
        return _indexed_config(
            meta,
            block_m=block_m,
            block_n=256,
            num_ctas_per_sm=num_ctas_per_sm,
            swap_ab=True,
        )

    if profile == "blackwell_decode_ep8" and (n, k, experts) in {
        (4096, 7168, 48),
        (7168, 2048, 48),
    }:
        # Ported to csrc/indexed/select.cpp (CHORD_CPP_JIT=1): change both.
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
    """Parse the small tuning format used by Humming/vLLM.

    The row lookup is ported to csrc/indexed/select.cpp and the row parsing to
    cpp_jit._table_id (CHORD_CPP_JIT=1): change both."""
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
