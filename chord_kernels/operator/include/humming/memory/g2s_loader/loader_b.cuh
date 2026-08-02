// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

// Indexed W4A16 packed-weight loader.

#pragma once

#include <humming/utils/base.cuh>
#include <humming/utils/ptx/cp_async.cuh>


template <
    class ProblemShape, class BlockShape,
    class TuningConfig>
class G2SMemoryLoaderB {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr uint32_t kPartMmaShapeK = 16;
  static constexpr uint32_t kSmemStride = BlockShape::N / 2;
  static constexpr uint32_t kGmemStride = ProblemShape::N / 2;
  static constexpr uint32_t kGmemExpertStride =
      ProblemShape::N * ProblemShape::K / 32;
  static constexpr uint32_t kNumInt4s =
      kSmemStride * BlockShape::K / kPartMmaShapeK;

public:
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  CUDA_INLINE
  explicit G2SMemoryLoaderB(const void *ptr)
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
    uint64_t gmem_offset =
        static_cast<uint64_t>(expert_id) * kGmemExpertStride;
    gmem_offset += n_block_id * kSmemStride;
    // A stream-K segment starts k_block_id blocks into the K reduction.
    gmem_offset += static_cast<uint64_t>(k_block_id) * kKBlockStride;
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }

private:
  static constexpr uint32_t kKBlockStride =
      kGmemStride * BlockShape::K / kPartMmaShapeK;

  CUDA_INLINE
  void advance() {
    gmem_ptr += kKBlockStride;
  }
};
