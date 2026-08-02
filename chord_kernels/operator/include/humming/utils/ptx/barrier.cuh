// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

//
// Global-memory barriers used by the stream-K epilogue to sequence the CTAs that
// share one output tile.  Two protocols, selected by slice count:
//
//   * serial chain (slice_count <= 3): each CTA waits until the lock word equals
//     its slice_id, adds its partial with a plain read-modify-write, then bumps
//     the lock by one; the final slice resets the word to 0.  The lock enforces
//     order, so the add need not be atomic.
//
//   * counter (slice_count > 3): the slice_id==0 CTA writes the tile and sets the
//     lock negative to release the rest, which then accumulate concurrently with
//     atomicAdd and bump the lock back toward 0.
//
// Both leave the lock at 0 when the tile is finished, so a zero-initialized lock
// buffer can be reused across launches without a reset.  Uses ld.acquire + red /
// st.cg + fence.acq_rel; no atom.cas.

#pragma once

#include <humming/utils/base.cuh>


// Every thread here is a math thread (no warp specialization), so sequencing the
// stream-K slices needs nothing narrower than a full block barrier.
CUDA_INLINE void sync_streamk_threads() {
  __syncthreads();
}

// Serial-chain acquire: spin until the lock reaches this slice's turn.
CUDA_INLINE void barrier_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state != count);
  }
  sync_streamk_threads();
}

// Counter acquire: spin until the lock has been released (state <= count).
CUDA_INLINE void barrier_acquire2(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = 1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state > count);
  }
  sync_streamk_threads();
}

// Serial-chain release: bump the lock by one, or reset it to 0 on the last slice.
CUDA_INLINE void barrier_release(int *lock, bool reset = false) {
  sync_streamk_threads();
  if (threadIdx.x == 0) {
    if (reset) {
      __stcg(&lock[0], 0);
    } else {
      int32_t val = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(val));
    }
  }
}

// Counter release: slice_id==0 stores the lock negative (1 - slice_count) to
// release the others; the rest add one, summing back to 0 when all have run.
CUDA_INLINE void barrier_release2(int *lock, int32_t val) {
  sync_streamk_threads();
  if (threadIdx.x == 0) {
    if (val < 0) {
      asm volatile("fence.acq_rel.gpu;\n");
      __stcg(&lock[0], val);
    } else {
      int32_t val2 = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(val2));
    }
  }
}
