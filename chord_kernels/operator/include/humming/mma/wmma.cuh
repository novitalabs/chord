// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/utils/base.cuh>


template <
    class MmaOpClass_, class SharedStorage, class ArithClass,
    class WarpShape,
    class TuningConfig>
struct WMMA {
public:
  static constexpr uint32_t kPartMmaShapeK = 16;

  // swap-AB: weights fill mma.sync operand A (M=16=n_out), activations fill
  // operand B (N=tokens). The PTX m16n8k16 is symmetric so only which data
  // lands in a[]/b[] (and the regs_c orientation) changes — see the emitter in
  // chord_kernels/operator/config/mma.py.
  static constexpr bool kSwapAb = TuningConfig::kSwapAb;

  using MmaOpClass = MmaOpClass_;
  using MmaShape = typename MmaOpClass::MmaShape;

  // Non-swap (default):
  //   regs_a = activations [M tokens], regs_b = weights [N n_out].
  // Swap:
  //   regs_a = weights, tiled by n_out on mma-M (WarpShape::N / 16 tiles);
  //   regs_b = activations, tiled by tokens on mma-N (WarpShape::M / 8 tiles);
  //   regs_c is n_out-major (mirror of WGMMA): [n_out tiles][token tiles].
  static constexpr uint32_t kNumARowTiles =
      kSwapAb ? MAX(WarpShape::N / MmaShape::M, 1u) : MAX(WarpShape::M / MmaShape::M, 1u);
  static constexpr uint32_t kNumBColTiles =
      kSwapAb ? MAX(WarpShape::M / MmaShape::N, 1u) : MAX(WarpShape::N / MmaShape::N, 1u);
  static constexpr uint32_t kNumCRowTiles = kNumARowTiles;
  static constexpr uint32_t kNumCColTiles = kNumBColTiles;

  ArithClass &arith;
  typename MmaOpClass::ARegisters regs_a[2][kNumARowTiles][kPartMmaShapeK / MmaShape::K];
  uint32_t regs_qb[2][4];
  typename MmaOpClass::BRegisters regs_b[2][kNumBColTiles][kPartMmaShapeK / MmaShape::K];
  typename MmaOpClass::CRegisters regs_c[kNumCRowTiles][kNumCColTiles];

  CUDA_INLINE
  WMMA(SharedStorage &smem, ArithClass &arith)
      : arith(arith) { (void)smem; }

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
    for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
      if constexpr (kSwapAb) {
        // swap-AB: dequant n_out-block i directly into A-frag M-tile i.
        uint32_t *regs_a_ptr = reinterpret_cast<uint32_t *>(regs_a[buffer_id][i]);
        arith.dequant_uint4_bf16_and_apply_group_bs_on_a(
            regs_qb[buffer_id], regs_a_ptr, i, buffer_id);
      } else {
        uint32_t *regs_b_ptr =
            reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 16 / MmaShape::N]);
        arith.dequant_uint4_bf16_and_apply_group_bs_on_b(
            regs_qb[buffer_id], regs_b_ptr, i, buffer_id);
      }
    };
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id, uint32_t num_token_tiles = 0) {
    (void)stage_id;
    uint32_t buffer_id = iter_id % 2;
    if constexpr (kSwapAb) {
      // Weight (dequanted once in transform_b) stays in regs_a; iterate it over
      // the token mma-N tiles. num_token_tiles is this m-block's populated
      // 8-token tile count (counted in humming.cuh); 0 means run all tiles.
      //
      // Semi-static schedule: j=0 is unconditional because every m-block holds
      // at least one populated tile, which gives ptxas a predicate-free anchor
      // to schedule the guarded j>=1 tiles around. Padding rows are masked
      // again in the epilogue via the wr_row_index sentinel.
      const uint32_t jmax = num_token_tiles ? num_token_tiles : kNumBColTiles;
      for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kNumBColTiles; j++) {
          if (j != 0 && j >= jmax) continue;
          PRAGMA_UNROLL
          for (uint32_t m = 0; m < kNumARowTiles; m++) {
            MmaOpClass::fma(
                regs_a[buffer_id][m][k], regs_b[buffer_id][j][k],
                regs_c[m][j], regs_c[m][j]);
          }
        }
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kNumBColTiles; j++) {
          PRAGMA_UNROLL
          for (uint32_t m = 0; m < kNumARowTiles; m++) {
            MmaOpClass::fma(
                regs_a[buffer_id][m][k], regs_b[buffer_id][j][k],
                regs_c[m][j], regs_c[m][j]);
          }
        }
      }
    }
  };

  CUDA_INLINE uint32_t *regs_a_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<uint32_t *>(regs_a[buffer_id]);
  };

  CUDA_INLINE uint32_t *regs_qb_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<uint32_t *>(regs_qb[buffer_id]);
  };

  // Per-buffer B-frag pointer (used by the swap-AB activation s2r read).
  CUDA_INLINE uint32_t *regs_b_buf_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
  };

  CUDA_INLINE uint32_t *regs_c_as_ptr() {
    return reinterpret_cast<uint32_t *>(regs_c);
  };

  CUDA_INLINE uint32_t *final_regs_c_as_ptr() {
    return regs_c_as_ptr();
  };
};
