// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

// Packed INT4 shared-memory to register loader.

#pragma once

#include <humming/utils/base.cuh>


template <
    class BlockShape, class WarpShape,
    class TuningConfig>
class S2RMemoryLoaderB {
private:
  static constexpr uint32_t kPartMmaShapeK = 16;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  // WGMMA uses Warp-N 32: pairs of warps each load half of one packed
  // 16-byte weight fragment.  MMA uses Warp-N 64 and loads a full fragment.
  static constexpr bool kLoadHalfFragment = WarpShape::N == 32;
  static constexpr uint32_t kTrueNWarps =
      kLoadHalfFragment ? N_WARPS / 2 : N_WARPS;
  static constexpr uint32_t kSmemStride =
      BlockShape::N * kPartMmaShapeK * 4 / 32 / 4;
  static constexpr uint32_t kLoadBytes =
      kLoadHalfFragment ? sizeof(uint2) : sizeof(uint4);
  using LoadType = typename LoadTypeChooser<kLoadBytes>::Type;

  static_assert(WarpShape::N == 32 || WarpShape::N == 64);
  static_assert(!kLoadHalfFragment || N_WARPS % 2 == 0);

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    uint32_t n_warp_id = warp_id % N_WARPS;
    if constexpr (kLoadHalfFragment) n_warp_id /= 2;

    uint32_t index = 32 * n_warp_id + lane_id;
    if constexpr (K_WARPS > 1) {
      const uint32_t k_warp_id =
          threadIdx.x / (TuningConfig::kNumThreads / K_WARPS);
      index +=
          kTrueNWarps * 32 * kWarpItersK * k_warp_id;
    }

    const LoadType *source = reinterpret_cast<const LoadType *>(
        smem_ptr + kSmemStride * iter_id);
    LoadType *destination = reinterpret_cast<LoadType *>(regs_ptr);
    if constexpr (kLoadHalfFragment) {
      destination[0] = source[index * 2 + warp_id % 2];
    } else {
      destination[0] = source[index];
    }
  }
};
