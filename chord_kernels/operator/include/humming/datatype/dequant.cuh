// Derived from inclusionAI/humming; modified for chord_kernels.

#pragma once

#include <cuda_bf16.h>
#include <humming/utils/base.cuh>
#include <humming/utils/ptx/math.cuh>


// Each packed word holds four nibble pairs; pair p (shift 4p) is the two
// adjacent-K values of the p-th (row-half, K-half) slot in mma.sync B-fragment
// order: p = row-half * 2 + K-half.  ``kWgmmaOrder`` writes result[i] from
// pair {0, 2, 1, 3}[i] instead, which is WGMMA's register-A order
// (K-half * 2 + row-half).  Only the shift immediates differ, so both orders
// cost the same instructions.
template <bool kWgmmaOrder = false>
CUDA_INLINE void dequant_uint4_to_bf16(
    const uint32_t *packed, uint32_t *result, uint32_t tile) {
  constexpr uint32_t kBf16Base = 0x43004300;
  // Codes are stored unsigned and re-centered at 8 -- giving the signed INT4
  // range [-8, 7] -- before the BF16 group scale is applied.
  constexpr uint32_t kBf16Center = 0x43084308;
  constexpr uint32_t kNibbleMask = 0x000F000F;

  const uint32_t values = packed[tile];
  const nv_bfloat162 center =
      *reinterpret_cast<const nv_bfloat162 *>(&kBf16Center);
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t slot = kWgmmaOrder ? (i % 2) * 2 + i / 2 : i;
    const uint32_t shifted = values >> (4 * slot);
    const uint32_t extracted = lop3_and_or(shifted, kNibbleMask, kBf16Base);
    nv_bfloat162 pair = *reinterpret_cast<const nv_bfloat162 *>(&extracted);
    pair = __hsub2(pair, center);
    result[i] = *reinterpret_cast<uint32_t *>(&pair);
  }
}
