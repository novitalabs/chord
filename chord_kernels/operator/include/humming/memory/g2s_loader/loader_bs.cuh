// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

// BF16 group-32 weight-scale loader for indexed W4A16.

#pragma once

#include <humming/utils/base.cuh>
#include <humming/utils/ptx/cp_async.cuh>


template <
    class ProblemShape, class BlockShape,
    class TuningConfig>
class G2SMemoryLoaderBS {
private:
  static constexpr uint32_t kGroupSize = 32;
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr uint32_t kSmemStride = BlockShape::N / 8;
  static constexpr uint32_t kGmemStride = ProblemShape::N / 8;
  static constexpr uint32_t kProblemNumGroups =
      ProblemShape::K / kGroupSize;
  static constexpr uint32_t kGmemExpertStride =
      kGmemStride * kProblemNumGroups;
  static constexpr uint32_t kNumGroups = BlockShape::K / kGroupSize;
  static constexpr uint32_t kNumInt4s = kSmemStride * kNumGroups;

  static_assert(ProblemShape::K % kGroupSize == 0);
  static_assert(BlockShape::K % kGroupSize == 0);

public:
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  CUDA_INLINE
  explicit G2SMemoryLoaderBS(const void *ptr)
      : gmem_ptr_raw(reinterpret_cast<const int4 *>(ptr)) {}

  CUDA_INLINE void load(int4 *smem_ptr) {
    cp_async_load_2d<
        kNumInt4s, kNumThreads,
        kGmemStride, kSmemStride>(
        gmem_ptr, smem_ptr);
    advance();
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id = 0) {
    const uint32_t group_offset =
        expert_id * kProblemNumGroups;
    uint64_t gmem_offset =
        static_cast<uint64_t>(group_offset) * kGmemStride +
        n_block_id * kSmemStride;
    // Skip the scale groups for the k-blocks before this stream-K segment.
    gmem_offset += static_cast<uint64_t>(k_block_id) * kKBlockStride;
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }

private:
  static constexpr uint32_t kKBlockStride = kGmemStride * kNumGroups;

  CUDA_INLINE
  void advance() {
    gmem_ptr += kKBlockStride;
  }
};
