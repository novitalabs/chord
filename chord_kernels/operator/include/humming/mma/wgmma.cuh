// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/datatype/dequant.cuh>
#include <humming/utils/base.cuh>
#include <humming/utils/ptx/wgmma.cuh>

#include <cstddef>


CUDA_INLINE uint64_t make_wgmma_smem_desc(uint32_t addr) {
  constexpr uint64_t desc_base = (uint64_t{1} << 62) | (uint64_t{64} << 32);

  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t *>(&desc)[0] = (addr >> 4);

  return desc;
};


template <
    class MmaOpClass_, class SharedStorage, class ArithClass,
    class BlockShape, class WarpShape>
struct WGMMA {
public:
  using MmaOpClass = MmaOpClass_;
  using MmaShape = typename MmaOpClass::MmaShape;

  static constexpr uint32_t kPartMmaShapeK = 16;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kSwizzleBytes = 128;

  // WGMMA never uses the swap-AB token-tile schedule.  Both members are still
  // required by the shared mainloop's `if constexpr (MMA::kSwapAb ...)`.
  static constexpr bool kSwapAb = false;
  static constexpr uint32_t kNumBColTiles = 1;

  static_assert(BlockShape::K >= 64);

  SharedStorage &smem;
  ArithClass &arith;
  uint32_t regs_qb[2][4];
  typename MmaOpClass::BRegisters regs_b[2][WarpShape::N * 4 / MmaShape::N][kPartMmaShapeK / MmaShape::K];
  typename MmaOpClass::CRegisters regs_c[WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];
  uint32_t smem_offset = 0;

  CUDA_INLINE
  WGMMA(SharedStorage &smem, ArithClass &arith)
      : smem(smem), arith(arith) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t m_warp_id = warp_id / N_WARPS % M_WARPS;
    uint32_t k_warp_id = warp_id / (N_WARPS * M_WARPS);

    constexpr uint32_t kSwizzleSizeK = 64;
    static_assert(kSwizzleSizeK >= WarpShape::K);

    const uint32_t row_offset = M_WARPS > 1 ? WarpShape::M * m_warp_id : 0;
    const uint32_t col_offset = K_WARPS > 1 ? WarpShape::K * k_warp_id : 0;

    smem_offset = row_offset * (kSwizzleBytes / 16);
    smem_offset += (col_offset % kSwizzleSizeK) / 8;
    smem_offset += (col_offset / kSwizzleSizeK) * (BlockShape::M * kSwizzleBytes / 16);
    smem_offset = smem_offset * sizeof(int4);
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::N / (MmaShape::N / 4); i++) {
      uint32_t *regs_b_ptr =
          reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 64 / MmaShape::N]);
      dequant_uint4_to_bf16(regs_qb[buffer_id], regs_b_ptr, i);
      arith.apply_group_scale_on_b(regs_b_ptr, i, buffer_id);
    };
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id, uint32_t num_token_tiles = 0) {
    static_assert(WarpShape::M == MmaShape::M);
    (void)num_token_tiles;
    uint32_t buffer_id = iter_id % 2;

    const uint32_t smem_base = cast_smem_ptr_to_uint(&smem);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      uint32_t smem_addr = smem_base + offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
      smem_addr += (iter_id * 2 + k) * sizeof(int4) + smem_offset;
      uint64_t desc = make_wgmma_smem_desc(smem_addr);

      constexpr uint32_t kNumIters = WarpShape::N / (MmaShape::N / 4);

      wgmma_fence();
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        MmaOpClass::fma(
            desc, regs_b[buffer_id][j][k], regs_c[j][0], true);
      }
      wgmma_commit();
      // regs_b is double-buffered and regs_c is only read by the epilogue, so
      // keeping one group in flight overlaps WGMMA with the next load/dequant.
      wgmma_wait<1>();
    }
  };

  CUDA_INLINE uint32_t *regs_qb_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<uint32_t *>(regs_qb[buffer_id]);
  };

  CUDA_INLINE uint32_t *regs_c_as_ptr() {
    return reinterpret_cast<uint32_t *>(regs_c);
  };

  CUDA_INLINE uint32_t *final_regs_c_as_ptr() {
    // Drain the group left in flight before the epilogue reads the accumulator.
    wgmma_wait<0>();
    return regs_c_as_ptr();
  };
};
