// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/utils/base.cuh>
#include <humming/utils/ptx/shared.cuh>

#include <cstddef>


template <
    class SharedStorage, class MmaOpClass,
    class BlockShape, class WarpShape>
class S2RMemoryLoaderA {
private:
  using MmaShape = typename MmaOpClass::MmaShape;
  static constexpr uint32_t kPartMmaShapeK = 16;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;

  static_assert(BlockShape::K >= 64);

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id, uint32_t stage_id = 0) {
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t m_iter_id = M_WARPS > 1 ? warp_id / N_WARPS % M_WARPS : 0;
    const uint32_t k_warp_id = warp_id / (M_WARPS * N_WARPS);
    uint32_t smem_uint = offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
    uint32_t smem_base = smem_uint / 128 % 8;

    PRAGMA_UNROLL
    for (uint32_t load_iter_id = 0; load_iter_id < CEIL_DIV(WarpShape::M, 16); load_iter_id++) {

      uint32_t row = m_iter_id * (BlockShape::M / M_WARPS) + load_iter_id * 16;
      uint32_t col = iter_id * 2 + k_warp_id * (kWarpItersK * 2);

      if constexpr (MmaShape::M == 8) {
        row += (lane_id / 16) * 8 + lane_id % 8;
        col += (lane_id / 8) % 2;
      } else {
        row += lane_id % 16;
        col += lane_id / 16;
      }

      if constexpr (BlockShape::K > 64) {
        row = BlockShape::M * (col / 8) + row;
        col = (col % 8) ^ ((row + smem_base) % 8);
      } else {
        static_assert(BlockShape::K == 64);
        col = col ^ ((row + smem_base) % 8);
      }

      uint32_t a_sh_rd = row * 8 + col;

      if ((load_iter_id == CEIL_DIV(WarpShape::M, 16) - 1) && WarpShape::M % 16 == 8) {
        ld_shared<2>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      } else {
        ld_shared<4>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      }
    };
  };

  // swap-AB: read the activation tile (tokens, k) into mma.sync B-frag (tokens
  // on N=8). ldmatrix.x2 (non-trans) over an 8x16 tile gives lane-ownership
  // token=lane/4, k=(lane%4)*2 — exactly the B-frag layout. Reuses the same
  // smem swizzle the g2s loader wrote. Only valid for the (BlockShape::K == 64)
  // decode tile; static_assert guards other shapes.
  CUDA_INLINE
  void load_swap_act(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    static_assert(BlockShape::K == 64,
                  "swap-AB activation read only implemented for the 64-wide-K decode tile");
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t k_warp_id = warp_id / (M_WARPS * N_WARPS);
    uint32_t smem = cast_smem_ptr_to_uint(smem_ptr) / 128;

    // lanes 0..15 address the two 8x8 matrices (k-block 0/1); lanes 16..31
    // alias the same rows (ldmatrix.x2 only consumes the first 16 addresses).
    uint32_t token = lane_id % 8;
    uint32_t kblk = (lane_id / 8) % 2;  // 0 -> k[0,8), 1 -> k[8,16)
    uint32_t col = iter_id * 2 + k_warp_id * (kWarpItersK * 2) + kblk;
    // Tokens on mma-N=8: WarpShape::M tokens span WarpShape::M/8 tiles, each
    // taking its own 8 token-rows [8t,8t+8) into B-frag tile t (2 uint32).
    // block_m=8 fills one tile, block_m=16 both; either way the weight is
    // dequanted once in transform_b and reused across the token tiles in run().
    PRAGMA_UNROLL
    for (uint32_t t = 0; t < MAX(WarpShape::M / MmaShape::N, 1u); t++) {
      uint32_t row = token + t * MmaShape::N;
      uint32_t col_sw = col ^ ((row + smem) % 8);
      uint32_t a_sh_rd = row * 8 + col_sw;
      ld_shared<2>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr + t * 2));
    }
  };
};
