// Derived from inclusionAI/humming; modified for chord_kernels.

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
  static constexpr bool kStreamKFp32 = TuningConfig::kStreamKFp32;

  using SmemReducer =
      EpilogueSmemReducer<MmaOpClass, BlockShape, WarpShape, TuningConfig>;
  using SmemWriter = EpilogueSmemWriter<
      SharedStorage, MmaOpClass, BlockShape, WarpShape, TuningConfig>;
  using GmemWriter = EpilogueGmemWriter<
      SharedStorage, ProblemShape, BlockShape, TuningConfig>;

  // FP32 stream-K workspace: one tile of floats per stream-K tile, laid out
  // [register][thread] so every CTA sharing a tile (same kernel, same thread ->
  // element mapping) reads back exactly the floats it would have held.
  static constexpr uint32_t kTileElems = BlockShape::M * BlockShape::N;
  static_assert(
      !kStreamKFp32 ||
          SmemWriter::kNumCRegs * SmemWriter::kNumRegThreads == kTileElems,
      "FP32 stream-K expects the reduced accumulators to cover the tile once");

public:
  SmemReducer smem_reducer;
  SmemWriter smem_writer;
  GmemWriter gmem_writer;

  // Per-tile stream-K state.  slice_count==1 is the data-parallel fast path.
  int *locks = nullptr;
  float *streamk_ws = nullptr;
  uint32_t slice_count = 1;
  uint32_t slice_id = 0;
  uint32_t locks_offset = 0;

  CUDA_INLINE
  EpiloguePipeline(
      SharedStorage &smem, void *output_ptr,
      uint32_t output_shape_m, uint32_t top_k, int *locks, float *streamk_ws)
      : smem_reducer(smem.reduce),
        smem_writer(smem.reduce),
        gmem_writer(smem.reduce, output_ptr, smem, output_shape_m, top_k),
        locks(locks),
        streamk_ws(streamk_ws) {
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
    if constexpr (kUseStreamK && kStreamKFp32) {
      // Sum the slices' FP32 partials in the workspace in slice order, so the
      // tile is rounded to BF16 once and the result is deterministic.  Only
      // the last slice goes on to write C.
      if (slice_count > 1 && !reduce_fp32_partial(regs_c_ptr)) return;
    }
    smem_writer.write(regs_c_ptr);
    sync_math_threads();
    // Serialize the CTAs sharing this output tile: acquire before writing,
    // release after, so each segment's partial lands in a defined order.
    if constexpr (kUseStreamK && !kStreamKFp32) {
      if (slice_count > 1) acquire_gmem_barrier();
    }
    gmem_writer.write();
    if constexpr (kUseStreamK && !kStreamKFp32) {
      if (slice_count > 1) release_gmem_barrier();
    }
    sync_math_threads();
  }

  // Stream-K geometry for the FP32 fixup (see reduce_fp32_partial).
  uint32_t k_block_id = 0;
  uint32_t tile_mnk_begin = 0;
  uint32_t streamk_base = 0;
  uint32_t streamk_iters_per_cta = 0;

  CUDA_INLINE
  void set_streamk_geometry(
      uint32_t k_block_id_, uint32_t tile_mnk_begin_, uint32_t streamk_base_,
      uint32_t streamk_iters_per_cta_) {
    k_block_id = k_block_id_;
    tile_mnk_begin = tile_mnk_begin_;
    streamk_base = streamk_base_;
    streamk_iters_per_cta = streamk_iters_per_cta_;
  }

  // Only segments that start mid-tile (k_block_id != 0) store a partial, and a
  // CTA has at most one: its first segment.  Every later segment starts at
  // k = 0, so one workspace tile per CTA suffices.
  CUDA_INLINE
  float *partial_slot(uint32_t cta) {
    return streamk_ws + cta * kTileElems;
  }

  // FP32 fixup.  Every segment but the k = 0 one stores its FP32 partial to
  // its slot and bumps the tile's lock, without waiting.  The k = 0 segment
  // (the last segment of its CTA, so normally the last to finish) waits until
  // the other slice_count - 1 have arrived, then adds their partials to its
  // own in CTA order: one fixed summation order, one BF16 rounding.  Returns
  // true on the CTA that goes on to write the tile.  Slots are read only after
  // being written in the same launch, so the workspace needs no zeroing; the
  // lock returns to 0.
  CUDA_INLINE
  bool reduce_fp32_partial(uint32_t *regs_c_ptr) {
    float *regs = reinterpret_cast<float *>(regs_c_ptr);
    const bool active = threadIdx.x < SmemWriter::kNumRegThreads;
    const uint32_t cta_first =
        (tile_mnk_begin - streamk_base) / streamk_iters_per_cta;
    if (k_block_id != 0) {
      const uint32_t cta =
          (tile_mnk_begin + k_block_id - streamk_base) / streamk_iters_per_cta;
      if (active) {
        float *slot = partial_slot(cta) + threadIdx.x;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < SmemWriter::kNumCRegs; i++) {
          __stcg(slot + i * SmemWriter::kNumRegThreads, regs[i]);
        }
      }
      barrier_release2(&locks[locks_offset], 1);
      return false;
    }
    barrier_acquire(&locks[locks_offset], static_cast<int>(slice_count) - 1);
    if (active) {
      for (uint32_t c = cta_first + 1; c < cta_first + slice_count; c++) {
        const float *slot = partial_slot(c) + threadIdx.x;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < SmemWriter::kNumCRegs; i++) {
          regs[i] += __ldcg(slot + i * SmemWriter::kNumRegThreads);
        }
      }
    }
    if (threadIdx.x == 0) __stcg(&locks[locks_offset], 0);
    return true;
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
