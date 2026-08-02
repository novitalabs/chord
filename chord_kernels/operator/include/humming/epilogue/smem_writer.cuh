// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <cuda_bf16.h>
#include <humming/utils/base.cuh>

#include <cstddef>
#include <type_traits>


CUDA_INLINE void shlf_trans_mma_c(float2 &vals) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(&vals);

  uint32_t val;
  uint32_t idx = (threadIdx.x / 4) % 2;
  switch (idx) {
    case 0: {
      val = vals_uint_ptr[1];
      break;
    };
    case 1: {
      val = vals_uint_ptr[0];
      break;
    }
  }

  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);

  switch (idx) {
    case 0: {
      vals_uint_ptr[1] = swapped_val;
      break;
    };
    case 1: {
      vals_uint_ptr[0] = swapped_val;
      break;
    }
  }
}


template <
    class SharedStorage, class MmaOpClass,
    class BlockShape, class WarpShape, class TuningConfig>
class EpilogueSmemWriter {
private:
  static constexpr bool kUseWgmma = MmaOpClass::kUseWgmma;
  // swap-AB: regs_c is [n_out, token] (n_out-major), like the WGMMA path.
  static constexpr bool kSwapAb = TuningConfig::kSwapAb;

  using scalar_t2 = nv_bfloat162;
  using MmaShape = typename MmaOpClass::MmaShape;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using MMA_CRegistersArrayType = CRegistersType[MAX(WarpShape::M / MmaShape::M, 1)][MAX(WarpShape::N / MmaShape::N, 1)];
  using WGMMA_CRegistersArrayType = CRegistersType[WarpShape::N * 4 / MmaShape::N][MAX(WarpShape::M / MmaShape::M, 1)];
  // swap: n_out tiles (WarpShape::N/16) x token tiles (WarpShape::M/8).
  using SWAP_CRegistersArrayType = CRegistersType[MAX(WarpShape::N / 16, 1)][MAX(WarpShape::M / 8, 1)];
  using CRegistersArrayType = std::conditional_t<
      kSwapAb, SWAP_CRegistersArrayType,
      std::conditional_t<kUseWgmma, WGMMA_CRegistersArrayType, MMA_CRegistersArrayType>>;

  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

public:
  int4 *smem_ptr;

  CUDA_INLINE
  EpilogueSmemWriter(int4 *smem_ptr)
      : smem_ptr(smem_ptr) {}

  CUDA_INLINE
  void write(uint32_t *regs_ptr) {
    if (threadIdx.x >= kNumThreads / K_WARPS) return;

    auto &regs = *reinterpret_cast<CRegistersArrayType *>(regs_ptr);
    scalar_t2 *smem_bf16_pair_ptr =
        reinterpret_cast<scalar_t2 *>(smem_ptr);
    uint32_t smem = offsetof(SharedStorage, reduce) / 128 % 8;
    using PackTypeC = float2;

    uint32_t laneid = threadIdx.x % 32;
    uint32_t warpid = threadIdx.x / 32;
    uint32_t warp_delta_row = (warpid / N_WARPS % M_WARPS) * WarpShape::M;
    uint32_t n_warp_id = warpid % N_WARPS;
    auto write_to_smem = [&](PackTypeC val, uint32_t row_8x8block, uint32_t col_8x8block) {
      scalar_t2 val_bf16_pair;

      if constexpr (kUseWgmma || kSwapAb) shlf_trans_mma_c(val);
      val_bf16_pair = __float22bfloat162_rn(val);

      if constexpr (!kUseWgmma && !kSwapAb) {
        uint32_t sub_row = laneid / 4;
        uint32_t row = warp_delta_row + 8 * row_8x8block + sub_row;
        uint32_t col = col_8x8block * 4 + WarpShape::N / 2 * n_warp_id;

        row = row + BlockShape::M * (col / 32);
        col = ((col % 32 / 4) ^ ((sub_row + smem) % 8)) * 4 + laneid % 4;

        uint32_t idx = row * 32 + col;
        smem_bf16_pair_ptr[idx] = val_bf16_pair;
      } else {
        uint32_t sub_row = (laneid % 4) * 2 + (laneid % 8) / 4;
        uint32_t row = warp_delta_row + 8 * col_8x8block + sub_row;

        uint32_t count = (64 / WarpShape::N);
        uint32_t col1 = ((n_warp_id % count * (8 / count) + row_8x8block) ^ ((sub_row + smem) % 8)) * 4 + laneid / 8;
        uint32_t col2 = (n_warp_id / count) * (BlockShape::M * 64 / 2);
        uint32_t idx = row * 32 + col1 + col2;
        smem_bf16_pair_ptr[idx] = val_bf16_pair;
      }
    };

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs) / sizeof(regs[0]); i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < sizeof(regs[0]) / sizeof(regs[0][0]); j++) {
        auto part_regs = reinterpret_cast<PackTypeC *>(&regs[i][j]);
        if constexpr (kSwapAb) {
          // regs[i][j] = n_out-tile i (16 rows = two 8-blocks p) x token-tile j.
          // C-frag float[4]: d0,d1 -> n_out row lane/4 (8-block p=0); d2,d3 ->
          // row lane/4+8 (8-block p=1); both at token col (lane%4)*2 {+0,+1}.
          // Feed write_to_smem(val, row_8x8block=2*i+p, col_8x8block=j) so the
          // shared kUseWgmma formula transposes [n_out,token] -> [token,n_out].
          PRAGMA_UNROLL
          for (uint32_t p = 0; p < 2; p++) {
            write_to_smem(part_regs[p], 2 * i + p, j);
          }
          continue;
        }
        constexpr uint32_t inner_m = (kUseWgmma ? (MmaShape::N / 4) : MmaShape::M) / 8;
        constexpr uint32_t inner_n = sizeof(regs[0][0]) / sizeof(PackTypeC) / inner_m;

        PRAGMA_UNROLL
        for (uint32_t m = 0; m < inner_m; m++) {
          PRAGMA_UNROLL
          for (uint32_t n = 0; n < inner_n; n++) {
            uint32_t row_index = i * inner_m + m;
            uint32_t col_index = j * inner_n + n;
            write_to_smem(part_regs[n * inner_m + m], row_index, col_index);
          }
        }
      }
    }
  }
};
