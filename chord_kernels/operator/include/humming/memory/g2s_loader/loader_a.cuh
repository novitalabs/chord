// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

// Indexed W4A16 activation loader.
//
// The indexed kernel always gathers BF16 activation rows through the routing
// indices loaded by Scheduler, using the 128-byte shared-memory swizzle and
// cp.async.

#pragma once

#include <humming/utils/base.cuh>
#include <humming/utils/ptx/cp_async.cuh>

#include <cstddef>


template <
    class SharedStorage, class ProblemShape, class BlockShape,
    class TuningConfig>
class G2SMemoryLoaderA {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;

  // BF16 activation tiles are always at least 64 columns wide (see the
  // static_assert below), so the 128-byte swizzle is the only layout needed.
  static constexpr uint32_t kSmemStride = BlockShape::K / 8;
  static constexpr uint32_t kGmemStride = ProblemShape::K / 8;
  static constexpr uint32_t kNumInt4s = kSmemStride * BlockShape::M;
  static constexpr uint32_t kNumLoadIters = CEIL_DIV(kNumInt4s, kNumThreads);

  static_assert(BlockShape::K >= 64);

public:
  SharedStorage &smem;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t shape_m;
  uint32_t load_row_index[kNumLoadIters];

  CUDA_INLINE
  G2SMemoryLoaderA(const void *ptr, SharedStorage &smem, uint32_t shape_m)
      : smem(smem),
        gmem_ptr_raw(reinterpret_cast<const int4 *>(ptr)),
        shape_m(shape_m) {}

  CUDA_INLINE void load(int4 *smem_ptr, uint32_t stage_id) {
    load_cp_async(smem_ptr, stage_id);
    advance();
  }

private:
  CUDA_INLINE
  void load_cp_async(int4 *smem_ptr, uint32_t stage_id) {
    const uint32_t smem_uint =
        offsetof(SharedStorage, stages) +
        stage_id * sizeof(typename SharedStorage::StageStorage);
    const uint32_t smem_base = smem_uint / 128 % 8;
    const uint32_t smem_swizzled_col =
        (threadIdx.x % 8) ^ (((threadIdx.x % 64) / 8 + smem_base) % 8);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumLoadIters; i++) {
      const uint32_t smem_offset = i * kNumThreads + threadIdx.x;
      const uint32_t smem_row = smem_offset / 8;
      const uint32_t smem_col = smem_offset % 8;
      const uint32_t smem_swizzled_offset =
          smem_row * 8 + smem_swizzled_col;

      const uint32_t gmem_col = smem_row / BlockShape::M * 8 + smem_col;
      const uint32_t gmem_row = load_row_index[i];
      const uint32_t gmem_offset = gmem_row * kGmemStride + gmem_col;

      const bool pred_load =
          kNumInt4s % kNumThreads == 0 ||
          i != kNumLoadIters - 1 ||
          smem_offset < kNumInt4s;
      const bool pred_row = gmem_row < shape_m;

      cp_async_load_pred(
          gmem_ptr + gmem_offset,
          smem_ptr + smem_swizzled_offset,
          pred_load && pred_row);
    }
  }

  CUDA_INLINE
  void advance() {
    gmem_ptr += kSmemStride;
  }

public:
  CUDA_INLINE
  void seek(uint32_t k_block_id = 0) {
    // Data-parallel blocks start at K=0; a stream-K segment starts at its
    // assigned k-block, offset by that many BlockShape::K columns.
    gmem_ptr = gmem_ptr_raw + k_block_id * kSmemStride;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumLoadIters; i++) {
      const uint32_t smem_offset = i * kNumThreads + threadIdx.x;
      const uint32_t smem_row = smem_offset / 8;
      const uint32_t gmem_row = smem_row % BlockShape::M;
      load_row_index[i] = smem.rd_row_index[gmem_row];
    }
  }
};
