// Derived from inclusionAI/humming; modified for chord_kernels.

#pragma once

#include <humming/utils/base.cuh>

CUDA_INLINE void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::
                   : "memory");
}

CUDA_INLINE void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::
                   : "memory");
}

template <uint32_t N>
CUDA_INLINE void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N)
               : "memory");
}
