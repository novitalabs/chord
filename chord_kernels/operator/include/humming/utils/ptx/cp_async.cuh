// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/utils/base.cuh>


CUDA_INLINE void cp_async_load(const int4 *gmem_ptr, int4 *smem_ptr) {
  const uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(smem), "l"(gmem_ptr)
      : "memory");
}


CUDA_INLINE void cp_async_load_pred(
    const int4 *gmem_ptr, int4 *smem_ptr, bool pred) {
  const uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.s32 p, %0, 0;\n"
      "  @p cp.async.cg.shared.global [%1], [%2], 16;\n"
      "}\n"
      :
      : "r"(static_cast<uint32_t>(pred)), "r"(smem), "l"(gmem_ptr)
      : "memory");
}


template <
    uint32_t kNumInt4s, uint32_t kNumThreads>
CUDA_INLINE void cp_async_load_1d(
    const int4 *gmem_ptr, int4 *smem_ptr) {
  constexpr uint32_t kIterations = CEIL_DIV(kNumInt4s, kNumThreads);
  const uint32_t thread_id = threadIdx.x;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kIterations; ++i) {
    const uint32_t index = i * kNumThreads + thread_id;
    if (kNumInt4s % kNumThreads == 0 ||
        i != kIterations - 1 ||
        index < kNumInt4s) {
      cp_async_load(gmem_ptr + index, smem_ptr + index);
    }
  }
}


template <
    uint32_t kNumInt4s, uint32_t kNumThreads,
    uint32_t kGmemStride, uint32_t kSmemStride>
CUDA_INLINE void cp_async_load_2d(
    const int4 *gmem_ptr, int4 *smem_ptr) {
  static_assert(kNumInt4s % kSmemStride == 0);
  const uint32_t thread_id = threadIdx.x;

  if constexpr (
      kSmemStride % kNumThreads == 0 ||
      kNumInt4s < kSmemStride) {
    constexpr uint32_t kLineIterations = kSmemStride / kNumThreads;
    constexpr uint32_t kNumLines = kNumInt4s / kSmemStride;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumLines; ++i) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLineIterations; ++j) {
        const uint32_t smem_offset =
            (i * kLineIterations + j) * kNumThreads + thread_id;
        const uint32_t gmem_offset =
            i * kGmemStride + j * kNumThreads + thread_id;
        cp_async_load(
            gmem_ptr + gmem_offset,
            smem_ptr + smem_offset);
      }
    }
  } else {
    constexpr uint32_t kIterations =
        CEIL_DIV(kNumInt4s, kNumThreads);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kIterations; ++i) {
      const uint32_t smem_offset = i * kNumThreads + thread_id;
      const uint32_t gmem_row = smem_offset / kSmemStride;
      const uint32_t gmem_offset =
          smem_offset + (kGmemStride - kSmemStride) * gmem_row;

      if (kNumInt4s % kNumThreads == 0 ||
          i != kIterations - 1) {
        cp_async_load(
            gmem_ptr + gmem_offset,
            smem_ptr + smem_offset);
      } else {
        cp_async_load_pred(
            gmem_ptr + gmem_offset,
            smem_ptr + smem_offset,
            smem_offset < kNumInt4s);
      }
    }
  }
}


CUDA_INLINE void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n");
}


template <uint32_t N>
CUDA_INLINE void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
