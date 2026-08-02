// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <cuda_bf16.h>

#include <humming/utils/base.cuh>


// Non-atomic add of two int4-packed BF16x2 registers.  Used on the serial-chain
// stream-K path, where the lock already serializes the CTAs sharing a tile.
CUDA_INLINE int4 reduce_add_bf162(int4 a, int4 b) {
  __nv_bfloat162 *ap = reinterpret_cast<__nv_bfloat162 *>(&a);
  __nv_bfloat162 *bp = reinterpret_cast<__nv_bfloat162 *>(&b);
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(int4) / 4; ++i) ap[i] = __hadd2(ap[i], bp[i]);
  return a;
}

// Atomic add of an int4-packed BF16x2 register into C.  Used on the counter
// stream-K path, where CTAs accumulate concurrently.
CUDA_INLINE void atomic_reduce_add_bf162(int4 a, int4 *dst) {
  __nv_bfloat162 *ap = reinterpret_cast<__nv_bfloat162 *>(&a);
  __nv_bfloat162 *dp = reinterpret_cast<__nv_bfloat162 *>(dst);
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(int4) / 4; ++i) atomicAdd(&dp[i], ap[i]);
}


template <
    class SharedStorage, class ProblemShape, class BlockShape,
    class TuningConfig>
class EpilogueGmemWriter {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr bool kUseStreamK = TuningConfig::kUseStreamK;

public:
  SharedStorage &smem;
  int4 *smem_ptr;
  int4 *gmem_ptr_raw;
  int4 *gmem_ptr;
  uint32_t output_shape_m;

  // Stream-K state, set per output tile before write(); slice_count==1 is the
  // data-parallel fast path (plain store, no reduction).
  uint32_t slice_count = 1;
  uint32_t slice_id = 0;

  CUDA_INLINE
  EpilogueGmemWriter(
      int4 *smem_ptr, void *output_ptr, SharedStorage &smem,
      uint32_t shape_m, uint32_t top_k)
      : smem(smem),
        smem_ptr(smem_ptr),
        gmem_ptr_raw(reinterpret_cast<int4 *>(output_ptr)),
        output_shape_m(shape_m * top_k) {}

  CUDA_INLINE
  void set_streamk_state(uint32_t slice_count_, uint32_t slice_id_) {
    slice_count = slice_count_;
    slice_id = slice_id_;
  }

  CUDA_INLINE
  void write() {
    constexpr uint32_t kTotalWriteInt4s =
        BlockShape::M * BlockShape::N * 2 / 16;
    constexpr bool kEvenIterations =
        kTotalWriteInt4s % kNumThreads == 0;
    constexpr uint32_t kIterations =
        CEIL_DIV(kTotalWriteInt4s, kNumThreads);
    constexpr uint32_t kOutputStride = ProblemShape::N / 8;
    const uint32_t smem_base = offsetof(SharedStorage, reduce) / 128 % 8;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kIterations; ++i) {
      const uint32_t smem_offset = threadIdx.x + kNumThreads * i;
      if (!kEvenIterations && i == kIterations - 1 &&
          smem_offset >= kTotalWriteInt4s) {
        continue;
      }

      const uint32_t smem_row = smem_offset / 8;
      const uint32_t smem_col = smem_offset % 8;
      const uint32_t smem_col_swizzled =
          smem_col ^ ((smem_row + smem_base) % 8);
      const uint32_t smem_offset_swizzled =
          smem_row * 8 + smem_col_swizzled;
      const uint32_t routed_row = smem.wr_row_index[smem_row % BlockShape::M];
      const uint32_t output_col =
          smem_row / BlockShape::M * 8 + smem_col;

      if (routed_row >= output_shape_m) continue;

      const uint32_t output_offset = routed_row * kOutputStride + output_col;
      const int4 val = smem_ptr[smem_offset_swizzled];
      if (!kUseStreamK || slice_count == 1 || slice_id == 0) {
        // Data-parallel, or the segment that initializes this tile: plain store.
        gmem_ptr[output_offset] = val;
      } else if (slice_count > 3) {
        // Many segments accumulate concurrently under the counter protocol.
        atomic_reduce_add_bf162(val, &gmem_ptr[output_offset]);
      } else {
        // Few segments run serially under the chain protocol, so a plain
        // read-modify-write is safe and cheaper than an atomic.
        gmem_ptr[output_offset] =
            reduce_add_bf162(val, gmem_ptr[output_offset]);
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t n_block_id) {
    gmem_ptr = gmem_ptr_raw + n_block_id * (BlockShape::N * 2 / 16);
  }
};
