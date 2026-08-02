// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/epilogue/gmem_writer.cuh>
#include <humming/epilogue/smem_reducer.cuh>
#include <humming/epilogue/smem_writer.cuh>
#include <humming/utils/ptx/barrier.cuh>


template <
    class MmaOpClass, class SharedStorage,
    class ProblemShape, class BlockShape, class WarpShape,
    class TuningConfig>
class EpiloguePipeline {
private:
  static constexpr bool kUseStreamK = TuningConfig::kUseStreamK;

  using SmemReducer =
      EpilogueSmemReducer<MmaOpClass, BlockShape, WarpShape, TuningConfig>;
  using SmemWriter = EpilogueSmemWriter<
      SharedStorage, MmaOpClass, BlockShape, WarpShape, TuningConfig>;
  using GmemWriter = EpilogueGmemWriter<
      SharedStorage, ProblemShape, BlockShape, TuningConfig>;

public:
  SmemReducer smem_reducer;
  SmemWriter smem_writer;
  GmemWriter gmem_writer;

  // Per-tile stream-K state.  slice_count==1 is the data-parallel fast path.
  int *locks = nullptr;
  uint32_t slice_count = 1;
  uint32_t slice_id = 0;
  uint32_t locks_offset = 0;

  CUDA_INLINE
  EpiloguePipeline(
      SharedStorage &smem, void *output_ptr,
      uint32_t output_shape_m, uint32_t top_k, int *locks)
      : smem_reducer(smem.reduce),
        smem_writer(smem.reduce),
        gmem_writer(smem.reduce, output_ptr, smem, output_shape_m, top_k),
        locks(locks) {
    sync_math_threads();
  }

  CUDA_INLINE
  void set_streamk_state(
      uint32_t slice_count_, uint32_t slice_id_, uint32_t locks_offset_) {
    slice_count = slice_count_;
    slice_id = slice_id_;
    locks_offset = locks_offset_;
    gmem_writer.set_streamk_state(slice_count_, slice_id_);
  }

  CUDA_INLINE
  void call(uint32_t *regs_c_ptr) {
    sync_math_threads();
    if constexpr (BlockShape::K > WarpShape::K) {
      smem_reducer.reduce(regs_c_ptr);
    }
    smem_writer.write(regs_c_ptr);
    sync_math_threads();
    // Serialize the CTAs sharing this output tile: acquire before writing,
    // release after, so each segment's partial lands in a defined order.
    if constexpr (kUseStreamK) {
      if (slice_count > 1) acquire_gmem_barrier();
    }
    gmem_writer.write();
    if constexpr (kUseStreamK) {
      if (slice_count > 1) release_gmem_barrier();
    }
    sync_math_threads();
  }

  CUDA_INLINE
  void acquire_gmem_barrier() {
    if (slice_count > 3) {
      // Counter protocol: slice_id==0 passes immediately (count 0), the rest
      // wait until it releases the lock negative.
      const int count = slice_id == 0 ? 0 : -1;
      barrier_acquire2(&locks[locks_offset], count);
    } else {
      // Serial chain: wait until it is this slice's turn.
      barrier_acquire(&locks[locks_offset], slice_id);
    }
  }

  CUDA_INLINE
  void release_gmem_barrier() {
    if (slice_count > 3) {
      const int32_t val =
          slice_id == 0 ? 1 - static_cast<int32_t>(slice_count) : 0;
      barrier_release2(&locks[locks_offset], val);
    } else {
      barrier_release(&locks[locks_offset], slice_id == slice_count - 1);
    }
  }

  CUDA_INLINE
  void sync_math_threads() {
    __syncthreads();
  }

  CUDA_INLINE
  void seek(uint32_t n_block_id) {
    gmem_writer.seek(n_block_id);
  }
};
