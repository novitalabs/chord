# Derived from inclusionAI/humming; modified for chord_kernels.

"""Tuned indexed-kernel schedules: GPU, shape and routed-M to a config.

See docs/design.md for the rules and measurements behind the tables.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import torch

from chord_kernels.operator.api import IndexedKernelConfig, _compatible_block_n


@dataclasses.dataclass(frozen=True)
class _Table:
    """A tuned table's kernel family and fallback tile."""

    name: str
    layout: str  # "wgmma" or "mma"
    swap_ab: bool
    block_m: int


# Kimi K2.5-shaped MoE on 8 GPUs: EP8 holds 48 whole experts per GPU (gate/up
# 4096x7168, down 7168x2048); TP8 holds all 384 with moe_intermediate sliced 8
# ways (gate/up 512x7168, down 7168x256).
_TABLES = {
    t.name: t
    for t in (
        _Table("h200_prefill_ep8", "wgmma", False, 16),
        _Table("h200_tp8", "wgmma", False, 16),
        _Table("h200_decode_ep8", "mma", True, 8),
        _Table("blackwell_decode_ep8", "mma", True, 8),
    )
}
_EP8_SHAPES = frozenset({(4096, 7168), (7168, 2048)})
_TP8_SHAPES = frozenset({(512, 7168), (7168, 256)})

# H200 decode/prefill table crossover, measured on both EP8 projections.
_H200_DECODE_MAX_TOKENS_PER_EXPERT = 18.75


def _indexed_config(
    profile: _Table,
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
        swap_ab = profile.swap_ab
    warp_n = 32 if profile.layout == "wgmma" else 64
    return IndexedKernelConfig(
        block_shape=(block_m, block_n, block_k),
        warp_shape=(block_m, warp_n, min(block_k, 64)),
        num_stages=num_stages,
        num_ctas_per_sm=num_ctas_per_sm,
        swap_ab=swap_ab,
        use_stream_k=use_stream_k,
    )


# WGMMA accumulator register ceiling (~255 regs per thread at 176).
_H200_PREFILL_MAX_BLOCK_M = 176


def _min_block_count_block_m(routed_m: int, num_experts: int) -> int:
    """Block-M minimizing ``sum(ceil(count_e * 1.1 / block_m))`` over the
    sampled routing (Humming's draw, so the choice matches Humming's)."""

    import numpy as np

    counts = _sampled_expert_counts(routed_m, num_experts)
    best_block_m = 8
    best_blocks = None
    for block_m in range(8, _H200_PREFILL_MAX_BLOCK_M + 1, 8):
        blocks = int(np.ceil(counts * 1.1 / block_m).sum())
        if best_blocks is None or blocks < best_blocks:
            best_blocks, best_block_m = blocks, block_m
    return best_block_m


# Fixed per-block cost, in rows, of the row-work model.
_ROW_WORK_BLOCK_OVERHEAD = 32

# Row-work block-M window served by the wide mid-size tiles.
_MID_TILE_MIN_BLOCK_M = 32
_MID_TILE_MAX_BLOCK_M = 80


# ``RandomState.randint`` is prefix-stable, so one stream per expert count
# serves every routed-M up to this bound.
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
def _sampled_expert_counts(routed_m: int, num_experts: int):
    """``np.bincount(RandomState(0).randint(0, E, size=m))``, read-only."""

    import numpy as np

    experts = max(num_experts, 1)
    rows = max(routed_m, 1)
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
def _min_row_work_block_m(routed_m: int, num_experts: int) -> int:
    """Block-M minimizing ``sum(ceil(count_e / block_m)) * (block_m + 32)``."""

    import numpy as np

    counts = _sampled_expert_counts(routed_m, num_experts)
    block_ms = np.arange(8, _H200_PREFILL_MAX_BLOCK_M + 1, 8)
    # argmin keeps the smallest block-M on ties.
    blocks = (-(-counts[None, :] // block_ms[:, None])).sum(axis=1)
    work = blocks * (block_ms + _ROW_WORK_BLOCK_OVERHEAD)
    return int(block_ms[int(np.argmin(work))])


def _uses_mid_tiles(routed_m: int, num_experts: int) -> bool:
    """``tok_e < 80`` with a row-work block-M in 32..80 (wide one-pass tiles)."""

    tokens_per_expert = routed_m / max(num_experts, 1)
    if tokens_per_expert >= 80:
        return False
    block_m = _min_row_work_block_m(routed_m, num_experts)
    return _MID_TILE_MIN_BLOCK_M <= block_m <= _MID_TILE_MAX_BLOCK_M


def _h200_prefill_block_m(routed_m: int, num_experts: int, shape_k: int) -> int:
    """EP8 prefill block-M from ``tok_e``: row work in the mid window, block
    count below 80, else cover one expert's padded rows in the fewest blocks."""

    tokens_per_expert = routed_m / max(num_experts, 1)
    if _uses_mid_tiles(routed_m, num_experts):
        return _min_row_work_block_m(routed_m, num_experts)
    if tokens_per_expert < 80:
        return _min_block_count_block_m(routed_m, num_experts)

    # 1.1x covers the per-expert overshoot a random router leaves in each block.
    padded = tokens_per_expert * 1.1
    num_blocks = math.ceil(padded / _H200_PREFILL_MAX_BLOCK_M)
    if num_blocks <= 2:
        return math.ceil(padded / num_blocks / 8) * 8
    # Only deep-K gate/up wins from a third 176 window.
    if tokens_per_expert <= 352 and shape_k >= 4096:
        return _H200_PREFILL_MAX_BLOCK_M
    return 128


def _h200_tp8_block_m(routed_m: int, num_experts: int, shape_k: int) -> int:
    """TP8 block-M from ``tok_e``; narrow N makes occupancy, not padding, bind."""

    tokens_per_expert = routed_m / max(num_experts, 1)
    if _uses_mid_tiles(routed_m, num_experts):
        return _min_row_work_block_m(routed_m, num_experts)
    if tokens_per_expert < 80:
        return _min_block_count_block_m(routed_m, num_experts)
    if tokens_per_expert <= 128:
        if shape_k > 512:
            return _min_row_work_block_m(routed_m, num_experts)
        return round(tokens_per_expert * 1.1 / 8) * 8
    if tokens_per_expert <= 190:
        return 96
    return 128


# H200 SXM streaming multiprocessors; the wave model below sizes grids for it.
_H200_NUM_SMS = 132


def _wide_tile_has_partial_wave(
    routed_m: int, num_experts: int, block_m: int, shape_n: int
) -> bool:
    """True when the ``block_m x 512`` grid's last wave is under 90% full."""

    import numpy as np

    counts = _sampled_expert_counts(routed_m, num_experts)
    tiles = int(np.ceil(counts / block_m).sum()) * (shape_n // 512)
    waves = -(-tiles // _H200_NUM_SMS)
    return tiles < 0.9 * waves * _H200_NUM_SMS


def _h200_tp8_use_stream_k(routed_m: int, num_experts: int, shape_k: int) -> bool:
    """TP8 stream-K: gate/up on (mid window: partial waves only), down only on
    the mid window's 256-wide tiles from block-M 56."""

    block_m = _h200_tp8_block_m(routed_m, num_experts, shape_k)
    if not _uses_mid_tiles(routed_m, num_experts):
        return shape_k > 512
    if shape_k <= 512:
        return 56 <= block_m <= _TP8_DOWN_WIDE_MAX_BLOCK_M
    return _wide_tile_has_partial_wave(routed_m, num_experts, block_m, 512)


def _h200_prefill_use_stream_k(
    routed_m: int, num_experts: int, shape_n: int, shape_k: int
) -> bool:
    """EP8 prefill stream-K: gate/up on, down below routed_m 5120; in the mid
    window down is off and gate/up splits only on a partial wave."""

    if _uses_mid_tiles(routed_m, num_experts):
        if shape_k < 4096:
            return False
        block_m = _min_row_work_block_m(routed_m, num_experts)
        return _wide_tile_has_partial_wave(
            routed_m, num_experts, block_m, shape_n
        )
    if shape_n >= 4096 and 512 < shape_k < 4096 and routed_m >= 5120:
        return False
    return True


# Largest block-M where TP8 down's mid-size 256-wide tile fits 2 CTAs/SM.
_TP8_DOWN_WIDE_MAX_BLOCK_M = 72

# H200 shared memory per SM and the driver's per-CTA reservation.
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
    """Deepest cp.async ring (6, else 5, else 4) that keeps the CTAs resident."""

    for num_stages in (6, 5):
        per_cta = (
            _indexed_smem_bytes(block_m, block_n, block_k, num_stages)
            + _H200_SMEM_RESERVED_PER_CTA
        )
        if num_ctas_per_sm * per_cta <= _H200_SMEM_PER_SM:
            return num_stages
    return 4


@functools.lru_cache(maxsize=4096)
def _select_indexed_kernel_config(
    table: str, n: int, k: int, experts: int, routed_m: int
) -> IndexedKernelConfig:
    """One table at one routed M; untuned shapes get the table's fallback."""
    m = max(routed_m, 1)
    prof = _TABLES[table]
    profile = prof.name

    if profile == "h200_prefill_ep8" and (n, k) in {
        (4096, 7168),
        (7168, 2048),
    }:
        block_m = _h200_prefill_block_m(m, experts, k)
        use_stream_k = _h200_prefill_use_stream_k(m, experts, n, k)
        if _uses_mid_tiles(m, experts):
            # Mid window: gate/up 512-wide one-pass, down 256x64 at 2 CTAs/SM.
            if k >= 4096:
                return _indexed_config(
                    prof, block_m=block_m, block_n=512, block_k=64,
                    num_stages=_h200_num_stages(block_m, 512, 64, 1),
                    use_stream_k=use_stream_k,
                )
            return _indexed_config(
                prof, block_m=block_m, block_n=256, block_k=64,
                num_stages=_h200_num_stages(block_m, 256, 64, 2),
                num_ctas_per_sm=2,
            )
        if block_m <= 32:
            # gate/up 128x256, down 256x128 (swapped costs down 13-16%).
            if n <= 4096:
                return _indexed_config(
                    prof, block_m=block_m, block_n=128, block_k=256,
                    use_stream_k=use_stream_k,
                )
            return _indexed_config(
                prof, block_m=block_m, block_n=256, block_k=128,
                use_stream_k=use_stream_k,
            )
        # 256x64; 2 CTAs/SM at block-M 40..80 hides cp.async + dequant latency.
        if 40 <= block_m <= 80:
            return _indexed_config(
                prof, block_m=block_m, block_n=256, block_k=64,
                num_ctas_per_sm=2, use_stream_k=use_stream_k,
            )
        # One-pass down past the stream-K crossover keeps 4 stages.
        deep_ring = use_stream_k or k >= 4096
        return _indexed_config(
            prof, block_m=block_m, block_n=256, block_k=64,
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
            # down (K=256) is latency-bound: always 2 CTAs/SM.
            if mid and block_m <= _TP8_DOWN_WIDE_MAX_BLOCK_M:
                # Stream-K rows keep 4 stages (6 measured 3% slower).
                return _indexed_config(
                    prof, block_m=block_m, block_n=256, block_k=64,
                    num_stages=(
                        4 if use_stream_k
                        else _h200_num_stages(block_m, 256, 64, 2)
                    ),
                    num_ctas_per_sm=2, use_stream_k=use_stream_k,
                )
            if block_m <= 32:
                return _indexed_config(
                    prof, block_m=block_m, block_n=128, block_k=128,
                    num_ctas_per_sm=2, use_stream_k=use_stream_k,
                )
            return _indexed_config(
                prof, block_m=block_m, block_n=128, block_k=64,
                num_stages=_h200_num_stages(block_m, 128, 64, 2),
                num_ctas_per_sm=2, use_stream_k=use_stream_k,
            )
        if mid:
            # gate/up mid window: one 512-wide tile covers the whole output.
            return _indexed_config(
                prof, block_m=block_m, block_n=512, block_k=64,
                num_stages=_h200_num_stages(block_m, 512, 64, 1),
                use_stream_k=use_stream_k,
            )
        if block_m <= 64:
            return _indexed_config(
                prof, block_m=block_m, block_n=128,
                block_k=256 if block_m <= 32 else 128,
                use_stream_k=use_stream_k,
            )
        return _indexed_config(
            prof, block_m=block_m, block_n=256, block_k=64,
            num_stages=_h200_num_stages(block_m, 256, 64, 1),
            use_stream_k=use_stream_k,
        )

    if profile == "h200_decode_ep8" and (n, k) in {
        (4096, 7168),
        (7168, 2048),
    }:
        tokens_per_expert = m / max(experts, 1) / 0.9
        # gate/up moves to block-M 16 a little earlier than down (30 tokens/GPU:
        # 2.5% faster; at 25 block-M 8 is still 7% faster).
        bm8_max = 5.5 if k >= 4096 else 6
        block_m = (
            8 if tokens_per_expert <= bm8_max
            else (16 if tokens_per_expert <= 13 else 24)
        )
        # 3 CTAs/SM; only down's block-M 8 tile gains from a fourth.
        num_ctas_per_sm = 4 if k < 4096 and block_m == 8 else 3
        return _indexed_config(
            prof,
            block_m=block_m,
            block_n=256,
            num_ctas_per_sm=num_ctas_per_sm,
            swap_ab=True,
        )

    if profile == "blackwell_decode_ep8" and (n, k, experts) in {
        (4096, 7168, 48),
        (7168, 2048, 48),
    }:
        # SM100 and SM103 share this table.
        if (n, k) == (7168, 2048):
            # down: stream-K only on the non-swap tiles.
            if m <= 272:
                return _indexed_config(
                    prof, block_m=8, block_n=256, num_ctas_per_sm=4,
                    swap_ab=True,
                )
            if m <= 848:
                return _indexed_config(
                    prof, block_m=16, block_n=256, num_ctas_per_sm=4,
                    swap_ab=True,
                )
            if m <= 1424:
                return _indexed_config(
                    prof, block_m=32, block_n=512, swap_ab=False,
                    use_stream_k=True,
                )
            return _indexed_config(
                prof, block_m=48, block_n=512, swap_ab=False,
                use_stream_k=True,
            )

        # gate/up: stream-K at every size.
        if m <= 320:
            return _indexed_config(
                prof, block_m=8, block_n=256, num_ctas_per_sm=4,
                swap_ab=True, use_stream_k=True,
            )
        if m <= 736:
            # swap-AB 16x256 at 4 CTAs/SM: 2.6-2.8% faster than the non-swap
            # 16x512x128 from 50 to 80 tokens/GPU.
            return _indexed_config(
                prof, block_m=16, block_n=256, num_ctas_per_sm=4,
                swap_ab=True, use_stream_k=True,
            )
        if m <= 1424:
            return _indexed_config(
                prof, block_m=32, block_n=512, swap_ab=False,
                use_stream_k=True,
            )
        return _indexed_config(
            prof, block_m=48, block_n=512, swap_ab=False,
            use_stream_k=True,
        )

    return IndexedKernelConfig.default(
        layout=prof.layout,
        block_m=prof.block_m,
        block_n=_compatible_block_n(n),
        swap_ab=prof.swap_ab,
    )


def _with_block_m(config: IndexedKernelConfig, block_m: int) -> IndexedKernelConfig:
    """Keep a projection's N/K tile while matching a shared routing block-M."""
    if block_m == config.block_m:
        return config
    swap_ab = config.swap_ab
    if config.warp_shape[1] == 64 and block_m % 16 == 8:
        swap_ab = True  # non-swap MMA has no m8n8k8 BF16 instruction
    return dataclasses.replace(
        config,
        block_shape=(block_m, config.block_n, config.block_k),
        warp_shape=(block_m, config.warp_shape[1], config.warp_shape[2]),
        swap_ab=swap_ab,
    )


def _table_for(capability: tuple[int, int], n: int, k: int, experts: int, m: int) -> str:
    """Which tuned table serves this GPU, projection shape and routed-M."""
    if capability in ((10, 0), (10, 3)):
        return "blackwell_decode_ep8"
    if capability != (9, 0):
        raise RuntimeError(
            "the indexed W4A16 kernel supports SM90, SM100 and SM103, got "
            f"compute capability {capability[0]}.{capability[1]}"
        )
    if (n, k) in _TP8_SHAPES:
        return "h200_tp8"
    if m / experts < _H200_DECODE_MAX_TOKENS_PER_EXPERT:
        return "h200_decode_ep8"
    return "h200_prefill_ep8"


@functools.lru_cache(maxsize=None)
def _device_capability(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def select_indexed_config(
    shape_n: int,
    shape_k: int,
    num_experts: int,
    routed_m: int,
    *,
    block_m: int | None = None,
    device: torch.device | int | str | None = None,
    capability: tuple[int, int] | None = None,
) -> IndexedKernelConfig:
    """Tuned config for one call; ``routed_m = num_tokens * top_k``.

    ``capability`` plans for a GPU that is not present.  ``block_m`` replaces
    the M tile and keeps N/K.  Cached.
    """
    for name, value in (
        ("shape_n", shape_n),
        ("shape_k", shape_k),
        ("num_experts", num_experts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if isinstance(routed_m, bool) or not isinstance(routed_m, int):
        raise TypeError(f"routed_m must be an int, got {type(routed_m).__name__}")
    if routed_m < 0:
        raise ValueError(f"routed_m must be non-negative, got {routed_m}")
    if capability is None:
        index = torch.device(device if device is not None else "cuda").index
        capability = _device_capability(
            torch.cuda.current_device() if index is None else index
        )
    m = max(routed_m, 1)
    config = _select_indexed_kernel_config(
        _table_for(tuple(capability), shape_n, shape_k, num_experts, m),
        shape_n, shape_k, num_experts, routed_m,
    )
    if block_m is not None:
        if isinstance(block_m, bool) or not isinstance(block_m, int) or block_m <= 0 or block_m % 8:
            raise ValueError(
                f"routing block_m must be a positive multiple of 8, got {block_m!r}"
            )
        config = _with_block_m(config, block_m)
    return config


__all__ = ["select_indexed_config"]
