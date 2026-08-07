# Derived from deepseek-ai/DeepGEMM; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.
"""SM90 launch-config selection for the vendored DeepGEMM W4A16 kernel.

This is a Python port of the ``is_w4a16`` branches of
``csrc/jit_kernels/heuristics/sm90.hpp`` (``SM90ArchSpec``), restricted to the
grouped-GEMM modes the chord layer dispatches: ``MGroupedMasked`` (decode)
and ``MGroupedContiguous`` (prefill).  The dense ``Normal`` mode and the
FP8/1D1D families are intentionally not ported.

Only BF16 activations, packed-INT4 weights (``weight_ratio == 2``), BF16
outputs without accumulation, and scale group 32 are representable here,
matching the vendored kernel instantiation.  The ``get_expected_*`` fallback
semantics are kept: the masked path compiles the per-group ``expected_m`` into
the heuristic while the contiguous one sees the concatenated total ``m`` with
``expected_num_groups == 1``.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Literal

# Fixed schema of the vendored instantiation (all site-asserted by the API).
W4A16_SCALE_GROUP = 32
# The weight atom the offline packer permutes within; the kernel consumes
# N in multiples of this atom.  Mirrored from the upstream generators.
W4A16_N_ATOM = 64
# The weight scale TMA descriptor requires N to meet the BF16 TMA alignment.
W4A16_TMA_N_ALIGN = 16

_SMEM_CAPACITY = 232448  # SM90 devices: 227 KiB usable per CTA
_NUM_MAX_STAGES = 16
_WGMMA_M = 64

W4A16GemmType = Literal["masked", "contiguous"]

# Tuning overrides for experiments (not production selection).  With both
# CHORD_W4A16_BM and CHORD_W4A16_BN set the candidate enumeration collapses
# to a single forced layout (BK/CM/CN fall back to their defaults:
# BK=64, CM=CN=1); CHORD_W4A16_STAGES pins the stage count after the smem
# ceiling.
_ENV_PREFIX = "CHORD_W4A16_"


def _env_int(name: str) -> int | None:
    value = os.environ.get(_ENV_PREFIX + name)
    if value is None:
        return None
    return int(value)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _get_swizzle_mode(block_size: int, elem_size: int) -> int:
    # First fitting mode out of {128, 64, 32, 16}; 16 means interleaving.
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise AssertionError("unreachable")


def _w4a16_masked_bm(expected_m: int, k: int) -> int:
    """Activation-M (MMA_N) tile for masked decode.

    Large K covers the worst-case masked draw floor(1.3*em): a spilled M-tile
    re-reads the whole K, which is expensive on deep-K gate/up.  Small K (down)
    instead wins with the tight ceil(1.25*em, 8) tile.
    """
    if k >= 4096:
        # Cover the worst-case masked draw floor(1.3*em): a spilled M-tile
        # re-reads the whole K, which is expensive on deep-K gate/up.
        bm = _align(max(16, (13 * expected_m) // 10), 8)
    else:
        # Small K (down): the redundant K-pass is cheap; a leaner tile wins.
        bm = _align(max(16, (expected_m * 5 + 3) // 4), 8)
    return min(bm, 128)


@dataclasses.dataclass(frozen=True)
class W4A16GemmDesc:
    """Host view of one W4A16 grouped-GEMM problem (mirrors ``GemmDesc``)."""

    gemm_type: W4A16GemmType
    m: int  # masked: per-group row capacity (a.size(1)); contiguous: total rows
    n: int
    k: int  # activation K; the packed weight stores K/2 bytes per row
    num_groups: int
    num_sms: int
    expected_m: int = 0  # masked only: nominal tokens per group; 0 -> m

    def __post_init__(self) -> None:
        if self.gemm_type not in ("masked", "contiguous"):
            raise ValueError(
                f"gemm_type must be 'masked' or 'contiguous', got {self.gemm_type!r}"
            )
        for name in ("m", "n", "k", "num_groups", "num_sms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_sms % 2:
            raise ValueError(f"num_sms must be even, got {self.num_sms}")
        if self.n % W4A16_TMA_N_ALIGN:
            raise ValueError(
                f"N must be a multiple of {W4A16_TMA_N_ALIGN} for the scale TMA "
                f"descriptor, got {self.n}"
            )
        if self.expected_m < 0 or (
            self.gemm_type == "contiguous" and self.expected_m not in (0, self.m)
        ):
            raise ValueError(
                "expected_m is only meaningful for the masked mode, got "
                f"{self.expected_m}"
            )

    @property
    def exp_m(self) -> int:
        # Mirror `desc.get_expected_m()`: masked uses the caller's expected_m,
        # contiguous uses the concatenated total m.
        return self.expected_m if self.expected_m > 0 else self.m

    @property
    def exp_num_groups(self) -> int:
        # The upstream contiguous path compiles with expected_num_groups == 1
        # because m is already the group-concatenated total.
        return self.num_groups if self.gemm_type == "masked" else 1


@dataclasses.dataclass(frozen=True)
class W4A16Layout:
    block_m: int
    block_n: int
    block_k: int
    cluster_m: int = 1
    cluster_n: int = 1

    @property
    def cluster_size(self) -> int:
        return self.cluster_m * self.cluster_n


@dataclasses.dataclass(frozen=True)
class W4A16StorageConfig:
    load_block_m: int
    load_block_n: int
    store_block_m: int
    store_block_n: int
    swizzle_a_mode: int
    swizzle_b_mode: int
    swizzle_cd_mode: int


@dataclasses.dataclass(frozen=True)
class W4A16PipelineConfig:
    smem_size: int
    num_stages: int


@dataclasses.dataclass(frozen=True)
class W4A16LaunchConfig:
    num_sms: int  # grid-dim X (persistent kernel)
    num_threads: int
    num_tma_threads: int
    num_math_threads: int


@dataclasses.dataclass(frozen=True)
class W4A16GemmConfig:
    layout: W4A16Layout
    storage: W4A16StorageConfig
    pipeline: W4A16PipelineConfig
    launch: W4A16LaunchConfig

    @property
    def compute_m(self) -> int:
        # W4A16 runs RS-WGMMA with operands swapped: BLOCK_N is the WGMMA-M.
        return self.layout.block_n


def _get_storage_config(desc: W4A16GemmDesc, layout: W4A16Layout) -> W4A16StorageConfig:
    # A/B are K-major, so swizzling follows the K extent: the BF16 activation
    # at 2 bytes and the packed INT4 weight at half a byte (BK/2 bytes per row).
    swizzle_a = _get_swizzle_mode(layout.block_k, 2)
    swizzle_b = _get_swizzle_mode(layout.block_k // 2, 1)
    swizzle_cd = _get_swizzle_mode(layout.block_n, 2)
    return W4A16StorageConfig(
        load_block_m=layout.block_m,
        load_block_n=layout.block_n,
        store_block_m=layout.block_m,
        store_block_n=layout.block_n,
        swizzle_a_mode=swizzle_a,
        swizzle_b_mode=swizzle_b,
        swizzle_cd_mode=swizzle_cd,
    )


def _get_pipeline_config(
    desc: W4A16GemmDesc, layout: W4A16Layout, storage: W4A16StorageConfig
) -> W4A16PipelineConfig:
    smem_cd = _align(layout.block_m * layout.block_n * 2, 1024)
    smem_barriers = _NUM_MAX_STAGES * 8 * 2
    smem_a_per_stage = storage.load_block_m * layout.block_k * 2
    smem_b_per_stage = storage.load_block_n * layout.block_k // 2
    # The SFA slot carries BF16 weight scales for the COMPUTE_M (== BLOCK_N)
    # tile at scale-group granularity along K: BK/32 sub-groups per stage.
    scale_sub = layout.block_k // W4A16_SCALE_GROUP
    smem_sfa_per_stage = _align(scale_sub * layout.block_n * 2, 128)

    smem_extra = smem_cd + smem_barriers
    smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage
    smem_max_stages = min(
        (_SMEM_CAPACITY - smem_extra) // smem_per_stage, _NUM_MAX_STAGES
    )

    # W4A16 stage tuning: masked decode is latency-bound at 1 block/SM, so
    # hold a fixed ~512-element buffered-K depth (BK64 -> 8, BK128 -> 4)
    # instead of filling smem; large-BM masked decode is barrier-bound and
    # wants ~768.
    target_stages = max(3, 512 // layout.block_k)
    if desc.gemm_type == "masked" and layout.block_m >= 72:
        target_stages = max(3, 768 // layout.block_k)
    num_stages = min(smem_max_stages, target_stages)

    env_stages = _env_int("STAGES")
    if env_stages is not None:
        num_stages = min(
            (_SMEM_CAPACITY - smem_extra) // smem_per_stage, env_stages
        )
    return W4A16PipelineConfig(
        smem_size=smem_extra + num_stages * smem_per_stage,
        num_stages=num_stages,
    )


def _get_launch_config(desc: W4A16GemmDesc, layout: W4A16Layout) -> W4A16LaunchConfig:
    num_tma_threads = 128
    # One consumer warpgroup for COMPUTE_M <= 64, two above; more consumer
    # warpgroups measured no speedup (the bottleneck is the single TMA producer
    # and memory latency at 1 block/SM, not consumer issue capacity).
    num_math_threads = 128 if layout.block_n <= 64 else 256
    return W4A16LaunchConfig(
        num_sms=desc.num_sms,
        num_threads=num_tma_threads + num_math_threads,
        num_tma_threads=num_tma_threads,
        num_math_threads=num_math_threads,
    )


def _layout_candidates(desc: W4A16GemmDesc) -> list[W4A16Layout]:
    # Environment-forced single layout (CHORD_W4A16_BM/BN are required
    # together; BK defaults to 64, CM/CN default to 1).
    env_bm, env_bn = _env_int("BM"), _env_int("BN")
    if env_bm is not None and env_bn is not None:
        return [
            W4A16Layout(
                block_m=env_bm,
                block_n=env_bn,
                block_k=_env_int("BK") or 64,
                cluster_m=_env_int("CM") or 1,
                cluster_n=_env_int("CN") or 1,
            )
        ]

    if desc.gemm_type == "masked":
        block_m_candidates = [_w4a16_masked_bm(desc.exp_m, desc.k)]
        block_k = 128
        # BN rule: BN=256 amortizes int4 dequant when enough tiles exist to
        # fill the machine (wave-quantization guard); small tiles keep BN=128.
        bm = block_m_candidates[0]
        n_tiles_256 = desc.exp_num_groups * _ceil_div(desc.n, 256)
        enough_2w = n_tiles_256 >= 2 * desc.num_sms
        enough_3w = n_tiles_256 >= 3 * desc.num_sms
        if desc.k >= 4096:
            use_bn256 = enough_2w
        elif bm < 32:
            use_bn256 = enough_2w
        elif bm < 72:
            use_bn256 = enough_3w
        else:
            use_bn256 = False
        block_n_candidates = [256] if use_bn256 else [128]
    else:
        block_k = 64
        # Prefill (contiguous): BM=128/BK=64 is the tensor-throughput optimum;
        # a single small problem falls back to BM=64 to fill the SMs.  m is the
        # group-concatenated total here (expected_num_groups == 1).
        bm128_tiles = _ceil_div(desc.exp_m, 128) * _ceil_div(desc.n, 128)
        block_m_candidates = [64] if bm128_tiles < desc.num_sms else [128]
        bn128_tiles = bm128_tiles
        block_n_candidates = (
            [64] if bn128_tiles * 3 < desc.num_sms * 2 else [128]
        )

    candidates: list[W4A16Layout] = []
    for cluster_m in (1, 2):
        for cluster_n in (1, 2):
            if cluster_m * cluster_n > 2:
                continue
            if desc.num_sms % (cluster_m * cluster_n):
                continue
            for block_m in block_m_candidates:
                for block_n in block_n_candidates:
                    # Multicast legality for the masked layout.
                    if desc.gemm_type == "masked" and (
                        _ceil_div(desc.n, block_n) % (cluster_m * cluster_n)
                    ):
                        continue
                    # Register ceiling: at least one compute dim must be <= 128.
                    if block_n > 128 and block_m > 128:
                        continue
                    layout = W4A16Layout(block_m, block_n, block_k, cluster_m, cluster_n)
                    storage = _get_storage_config(desc, layout)
                    # 32B weight swizzle is allowed for the INT4 B rows.
                    if storage.swizzle_a_mode % 64 or storage.swizzle_b_mode % 32:
                        continue
                    stages = _get_pipeline_config(desc, layout, storage).num_stages
                    if stages < 3 or (block_m * block_n < 128 * 192 and stages < 4):
                        continue
                    candidates.append(layout)
    if not candidates:
        raise RuntimeError(
            f"no legal W4A16 layout for desc {dataclasses.asdict(desc)}"
        )
    return candidates


def _num_predicted_cycles(desc: W4A16GemmDesc, layout: W4A16Layout) -> float:
    """L1/L2 roofline cost model (ported ``SM90ArchSpec::get_layout_info``)."""
    num_blocks = (
        _ceil_div(desc.exp_m, layout.block_m)
        * _ceil_div(desc.n, layout.block_n)
        * desc.exp_num_groups
    )
    num_waves = _ceil_div(num_blocks, desc.num_sms)
    wave_eff = num_blocks / (num_waves * desc.num_sms)

    l2_bw_per_cycle = int(min(64.0 * desc.num_sms, 8e6 / 1.3e3))  # C++ int truncation
    l1_bw_per_cycle = 128 * desc.num_sms
    expected_k = desc.k
    # All byte counts at 2x scale: A/outs 2B*2, weight 1B*2/2.
    bytes_a, bytes_b, bytes_cd = 4, 1, 4
    l2_ab = expected_k * (
        layout.block_m // layout.cluster_n * bytes_a
        + layout.block_n // layout.cluster_m * bytes_b
    )
    l1_ab = expected_k * (layout.block_m * bytes_a + layout.block_n * bytes_b)
    l1_tc = expected_k * (
        max(_WGMMA_M, layout.block_m) * bytes_a + layout.block_n * bytes_b
    ) + layout.block_m * layout.block_n * bytes_cd
    l1_l2_cd = layout.block_m * layout.block_n * bytes_cd  # no accumulation

    # int64 division in the upstream model.
    l2_cycles = (l2_ab + l1_l2_cd) * num_blocks // l2_bw_per_cycle
    l1_cycles = (l1_ab + l1_tc + l1_l2_cd) * num_blocks // l1_bw_per_cycle
    cycles = max(l1_cycles, l2_cycles) / wave_eff

    # Multicast loses to the cluster sync overhead when one wave fills the SMs,
    # and for masked GEMM once each group fits in a single M-block.
    if layout.cluster_size > 1 and (
        num_waves <= 1
        or (desc.gemm_type == "masked" and desc.exp_m <= layout.block_m)
    ):
        cycles = math.inf
    return cycles


def select_w4a16_config(desc: W4A16GemmDesc) -> W4A16GemmConfig:
    """Resolve the launch config exactly like ``get_best_config<SM90ArchSpec>``."""
    candidates = _layout_candidates(desc)
    best = min(candidates, key=lambda layout: _num_predicted_cycles(desc, layout))
    storage = _get_storage_config(desc, best)
    # Mirror the host launch asserts of the upstream 1D2D dispatch: the vendored
    # kernel enumerates only these swizzle/block-K pairings.
    if storage.swizzle_a_mode != min(best.block_k * 2, 128):
        raise RuntimeError(
            f"swizzle-a mismatch: got {storage.swizzle_a_mode}, expected "
            f"{min(best.block_k * 2, 128)}"
        )
    if storage.swizzle_b_mode != best.block_k // 2:
        raise RuntimeError(
            f"swizzle-b mismatch: got {storage.swizzle_b_mode}, expected "
            f"{best.block_k // 2}"
        )
    return W4A16GemmConfig(
        layout=best,
        storage=storage,
        pipeline=_get_pipeline_config(desc, best, storage),
        launch=_get_launch_config(desc, best),
    )


__all__ = [
    "W4A16GemmConfig",
    "W4A16GemmDesc",
    "W4A16GemmType",
    "W4A16LaunchConfig",
    "W4A16Layout",
    "W4A16PipelineConfig",
    "W4A16_SCALE_GROUP",
    "W4A16StorageConfig",
    "select_w4a16_config",
]
