// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/utils/base.cuh>
#include <humming/utils/ptx/cp_async.cuh>


template <
    class SharedStorage,
    class ProblemShape, class BlockShape,
    class TuningConfig>
class Scheduler {
private:
  static constexpr bool kUseStreamK = TuningConfig::kUseStreamK;
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr uint32_t kNBlocks = ProblemShape::N / BlockShape::N;
  static constexpr uint32_t kKBlocks = ProblemShape::K / BlockShape::K;

  // Data-parallel iteration state: whole-K blocks handed out round-robin.
  uint32_t dp_mn_iters;
  uint32_t dp_mn_next_index;
  uint32_t dp_mn_total_iters;

  // Stream-K iteration state: a flat run of mnk (mn-tile x k-block) iterations
  // split evenly across CTAs, so one output tile's K dimension is shared.
  uint32_t mn_blocks;
  uint32_t mnk_blocks;
  uint32_t streamk_mnk_total_iters;
  uint32_t streamk_mnk_iters;
  uint32_t streamk_mnk_next_index;

  const uint32_t *sorted_ids;
  const uint32_t *expert_ids;
  uint32_t top_k;

public:
  SharedStorage &smem;
  uint32_t expert_id;
  uint32_t m_block_id;
  uint32_t n_block_id;
  uint32_t k_block_id;
  uint32_t slice_iters;

  // Stream-K reduction bookkeeping consumed by the epilogue.  slice_count==1 is
  // the data-parallel fast path (one CTA owns the whole tile, plain write).
  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  CUDA_INLINE
  Scheduler(
      SharedStorage &smem,
      uint32_t top_k,
      const uint32_t *sorted_ids,
      const uint32_t *expert_ids,
      const uint32_t *num_tokens_padded)
      : smem(smem),
        sorted_ids(sorted_ids),
        expert_ids(expert_ids),
        top_k(top_k) {
    const uint32_t m_blocks =
        CEIL_DIV(num_tokens_padded[0], BlockShape::M);
    mn_blocks = m_blocks * kNBlocks;
    mnk_blocks = mn_blocks * kKBlocks;

    if constexpr (kUseStreamK) {
      // Peel off a tail of mn-tiles for K-splitting; the rest stay data
      // parallel.  When the tail is a small fraction of the grid, fold in one
      // more grid's worth so each stream-K CTA still gets a useful slice.
      uint32_t streamk_mn_blocks = mn_blocks;
      if (mn_blocks > gridDim.x) {
        streamk_mn_blocks = mn_blocks % gridDim.x;
        if (streamk_mn_blocks && streamk_mn_blocks * 10 <= gridDim.x) {
          streamk_mn_blocks += gridDim.x;
        }
      }

      dp_mn_iters = (mn_blocks - streamk_mn_blocks) / gridDim.x;

      const uint32_t streamk_mnk_blocks = streamk_mn_blocks * kKBlocks;
      streamk_mnk_total_iters = CEIL_DIV(streamk_mnk_blocks, gridDim.x);
      streamk_mnk_next_index =
          gridDim.x * dp_mn_iters * kKBlocks + streamk_mnk_total_iters * blockIdx.x;

      if (streamk_mnk_next_index >= mnk_blocks) {
        streamk_mnk_iters = 0;
      } else {
        streamk_mnk_iters = mnk_blocks - streamk_mnk_next_index;
        if (streamk_mnk_iters > streamk_mnk_total_iters) {
          streamk_mnk_iters = streamk_mnk_total_iters;
        }
      }
    } else {
      dp_mn_iters = mn_blocks / gridDim.x;
      if (blockIdx.x < mn_blocks % gridDim.x) ++dp_mn_iters;
    }

    dp_mn_total_iters = dp_mn_iters;
    slice_count = 1;
    slice_id = 0;
    if (dp_mn_iters) dp_mn_next_index = blockIdx.x;
  }

  CUDA_INLINE
  bool get_next_block() {
    bool has_next_block = false;
    if (dp_mn_iters) {
      // Data-parallel tile: one CTA computes the whole K in one pass.
      slice_iters = kKBlocks;
      m_block_id = dp_mn_next_index / kNBlocks;
      n_block_id = dp_mn_next_index % kNBlocks;
      k_block_id = 0;
      slice_count = 1;
      slice_id = 0;
      dp_mn_next_index += gridDim.x;
      --dp_mn_iters;
      has_next_block = true;
    } else if constexpr (kUseStreamK) {
      has_next_block = get_streamk_next_block();
    }

    if (has_next_block) load_routing_block();
    return has_next_block;
  }

private:
  CUDA_INLINE
  bool get_streamk_next_block() {
    if (!streamk_mnk_iters) return false;

    const uint32_t streamk_mn_index = streamk_mnk_next_index / kKBlocks;
    m_block_id = streamk_mn_index / kNBlocks;
    n_block_id = streamk_mn_index % kNBlocks;
    k_block_id = streamk_mnk_next_index - streamk_mn_index * kKBlocks;

    // Run from k_block_id to the end of this tile's K, but no further than the
    // even share this CTA was handed.
    slice_iters = kKBlocks - k_block_id;
    if (slice_iters > streamk_mnk_iters) slice_iters = streamk_mnk_iters;

    streamk_mnk_iters -= slice_iters;
    streamk_mnk_next_index += slice_iters;

    // How many CTAs cover this tile, and which one this is.  The flip makes
    // slice_id==0 the segment that plain-writes C (the others accumulate).
    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = CEIL_DIV(kKBlocks - slice_iters, streamk_mnk_total_iters) + 1;
    } else {
      slice_id = k_block_id / streamk_mnk_total_iters;
      const uint32_t slice_first_block_iters =
          k_block_id - slice_id * streamk_mnk_total_iters;
      slice_count = CEIL_DIV(kKBlocks - slice_first_block_iters, streamk_mnk_total_iters);
      if (slice_first_block_iters) {
        ++slice_id;
        ++slice_count;
      }
    }
    slice_id = slice_count - 1 - slice_id;

    // One lock word per stream-K tile, indexed past the data-parallel tiles.
    locks_offset = streamk_mn_index - dp_mn_total_iters * gridDim.x;
    return true;
  }

  CUDA_INLINE
  void load_routing_block() {
    expert_id = expert_ids[m_block_id];

    const uint32_t *global = sorted_ids + m_block_id * BlockShape::M;
    cp_async_load_1d<BlockShape::M / 4, kNumThreads>(
        reinterpret_cast<const int4 *>(global),
        reinterpret_cast<int4 *>(smem.wr_row_index));
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumThreads); ++i) {
      const uint32_t index = kNumThreads * i + threadIdx.x;
      if (index < BlockShape::M) {
        smem.rd_row_index[index] = smem.wr_row_index[index] / top_k;
      }
    }
    __syncthreads();
  }
};
