// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

// BF16 group-32 scale shared-memory to register loader.

#pragma once

#include <humming/utils/base.cuh>


template <
    class MmaOpClass,
    class BlockShape, class WarpShape>
class S2RMemoryLoaderBS {
private:
  static constexpr uint32_t kGroupSize = 32;
  static constexpr uint32_t kPartMmaShapeK = 16;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;

  // The packed scale permutation supplies two BF16 scales per 16 output
  // columns.  WGMMA Warp-N 32 shares a mini-block between two warps, while
  // MMA Warp-N 64 consumes it from one warp.
  static constexpr uint32_t kNumScalesPerSubBlock = 2;
  static constexpr uint32_t kNumScales = WarpShape::N / 8;
  static constexpr uint32_t kNumBytesPerThread = kNumScales * 2;
  static constexpr uint32_t kNumWarpsPerMiniBlock = 64 / WarpShape::N;
  static constexpr uint32_t kLoadBytes = 16 / kNumWarpsPerMiniBlock;
  using LoadType = typename LoadTypeChooser<kLoadBytes>::Type;

  static constexpr uint32_t kSmemStride = BlockShape::N / 8;
  static constexpr uint32_t kSmemStrideLoadType =
      kSmemStride * sizeof(int4) / sizeof(LoadType);

  static_assert(WarpShape::N == 32 || WarpShape::N == 64);
  static_assert(kNumBytesPerThread == sizeof(LoadType));
  static_assert(MmaOpClass::kUseWgmma == (WarpShape::N == 32));

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t n_warp_id =
        (warp_id % N_WARPS) / kNumWarpsPerMiniBlock;

    constexpr uint32_t kWarpLoadDelta = 16 / kNumScalesPerSubBlock;
    uint32_t source_index =
        kWarpLoadDelta * kNumWarpsPerMiniBlock * n_warp_id;
    source_index +=
        (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock +
        warp_id % kNumWarpsPerMiniBlock;

    const uint32_t k_index =
        (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K +
        iter_id * kPartMmaShapeK;
    source_index += (k_index / kGroupSize) * kSmemStrideLoadType;

    const LoadType *source =
        reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *destination = reinterpret_cast<LoadType *>(regs_ptr);
    destination[0] = source[source_index];
  }
};
