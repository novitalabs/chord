// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <cuda_bf16.h>
#include <humming/utils/base.cuh>
#include <humming/utils/ptx/math.cuh>


template <class WarpShape>
class MainloopArithmetic {
private:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;

  static constexpr uint32_t kNumScaleWords = MAX(WarpShape::N / 8, 8) / 2;

public:
  // Two buffers overlap scale loading with the current MMA/WGMMA iteration;
  // each word holds two BF16 group scales.
  uint32_t bs[2][kNumScaleWords];

  CUDA_INLINE
  void apply_group_scale_on_b(
      uint32_t *regs_b, uint32_t j, uint32_t buffer_id) {
    scalar_t2 *values = reinterpret_cast<scalar_t2 *>(regs_b);
    // Take the scale pair through the single word; see the note on
    // dequant_and_scale below.
    scalar_t2 bs_word = *reinterpret_cast<scalar_t2 *>(&bs[buffer_id][j]);
    scalar_t2 scale[2] = {__low2bfloat162(bs_word), __high2bfloat162(bs_word)};

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; ++i) {
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < 2; ++k) {
        values[i * 2 + k] = __hmul2(values[i * 2 + k], scale[k]);
      }
    }
  }

  CUDA_INLINE
  void dequant_uint4_bf16_and_apply_group_bs_on_b(
      const uint32_t *regs_qb, uint32_t *regs_b,
      uint32_t scale_index, uint32_t buffer_id) {
    dequant_and_scale<false>(regs_qb, regs_b, scale_index, buffer_id);
  }

  // swap-AB variant: identical dequant + group-scale math, but scatters the
  // four result registers into mma.sync A-frag slots instead of B-frag.
  CUDA_INLINE
  void dequant_uint4_bf16_and_apply_group_bs_on_a(
      const uint32_t *regs_qb, uint32_t *regs_a,
      uint32_t scale_index, uint32_t buffer_id) {
    dequant_and_scale<true>(regs_qb, regs_a, scale_index, buffer_id);
  }

  CUDA_INLINE uint32_t *regs_bs_as_ptr(uint32_t buffer_id) {
    return bs[buffer_id];
  }

private:
  // Scale extraction must go through __low2bfloat162/__high2bfloat162: their
  // inline asm splits the 32-bit scale word with a single `mov.b32 {lo,hi}`.
  // Reading the two halves through a scalar_t* (or the array base) makes NVVM
  // materialize a 64-bit value and extract with SHF.R.U64 funnel shifts inside
  // the hot dequant loop, costing 1-2% on decode.
  template <bool kWriteMmaA>
  CUDA_INLINE void dequant_and_scale(
      const uint32_t *regs_qb, uint32_t *regs,
      uint32_t j, uint32_t buffer_id) {
    constexpr uint32_t base = 0x43004300;
    constexpr uint32_t mask = 0x000F000F;
    // Subtract-then-scale, NOT hfma2(x, bs, -136*bs): (128+w)-136 = w-8 is
    // exact in bf16, so the result is the single-rounding RN((w-8)*bs),
    // bit-identical to the unfused path.  The fma form rounds -136*bs, a bias
    // that is constant within a scale group and accumulates coherently over K.
    constexpr uint32_t fixed_zp = 0x43084308;
    const scalar_t2 zp_bf162 = *reinterpret_cast<const scalar_t2 *>(&fixed_zp);
    // A-frag and B-frag share per-lane k-ownership; only the register slot
    // order differs.  This maps dequant output i -> A-frag register slot.
    constexpr uint32_t kAFragPerm[4] = {0, 2, 1, 3};

    scalar_t2 bs_word = *reinterpret_cast<scalar_t2 *>(&bs[buffer_id][j]);
    scalar_t2 bs_vals[2] = {__low2bfloat162(bs_word), __high2bfloat162(bs_word)};

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      const uint32_t index = j * 4 + i;
      uint32_t qb_val = regs_qb[index / 4];
      const uint32_t shift_count = 4 * i;
      if (shift_count) qb_val = qb_val >> shift_count;

      const uint32_t extracted_val = lop3_and_or(qb_val, mask, base);
      const scalar_t2 extracted_val_bf162 =
          *reinterpret_cast<const scalar_t2 *>(&extracted_val);
      const scalar_t2 scaled_val = __hmul2(
          __hsub2(extracted_val_bf162, zp_bf162),
          bs_vals[i / 2]);
      regs[kWriteMmaA ? kAFragPerm[i] : i] =
          *reinterpret_cast<const uint32_t *>(&scaled_val);
    }
  }
};
