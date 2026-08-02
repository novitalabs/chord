// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/base.cuh>
#include <humming/utils/storage.cuh>

#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>

#include <type_traits>


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape,
    class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const void *A,
    const void *B,
    void *C,
    const void *BS,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    int *locks,
    uint32_t shape_m,
    uint32_t top_k) {

  constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using Storage = SharedStorage<BlockShape, WarpShape, TuningConfig>;
  using Scheduler = Scheduler<
      Storage, ProblemShape, BlockShape, TuningConfig>;
  using ProducerPipeline = ProducerPipeline<
      Storage, ProblemShape, BlockShape, TuningConfig>;
  using Consumer = ConsumerPipeline<TuningConfig>;
  using Arithmetic = MainloopArithmetic<WarpShape>;
  using MmaSync = WMMA<
      MmaOpClass, Storage, Arithmetic,
      WarpShape, TuningConfig>;
  using WarpGroupMma = WGMMA<
      MmaOpClass, Storage, Arithmetic,
      BlockShape, WarpShape>;
  using Mma = std::conditional_t<
      MmaOpClass::kUseWgmma, WarpGroupMma, MmaSync>;
  using Epilogue = EpiloguePipeline<
      MmaOpClass, Storage, ProblemShape, BlockShape, WarpShape,
      TuningConfig>;
  using S2RMemoryPipeline = S2RMemoryPipeline<
      Storage, Mma, BlockShape, WarpShape, TuningConfig>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<Storage *>(shared_memory);

  auto scheduler = Scheduler(
      smem, top_k, sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr);
  auto mainloop_arith = Arithmetic();
  auto mma = Mma(smem, mainloop_arith);
  auto epilogue = Epilogue(smem, C, shape_m, top_k, locks);
  auto producer = ProducerPipeline(smem, A, B, BS, shape_m);
  auto consumer = Consumer();
  auto s2r_pipe = S2RMemoryPipeline(smem, mma);

  __syncthreads();

  while (scheduler.get_next_block()) {
    mma.zero_accum();
    __syncthreads();

    uint32_t &slice_iters = scheduler.slice_iters;
    // A stream-K segment starts partway into K; a data-parallel block has
    // k_block_id == 0 and covers the whole reduction.
    producer.seek(scheduler.expert_id, scheduler.n_block_id, scheduler.k_block_id);
    epilogue.seek(scheduler.n_block_id);
    epilogue.set_streamk_state(
        scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

    // swap-AB: count populated 8-token mma-N tiles of this m-block. Tokens are
    // front-packed per expert with sentinel >= shape_m*top_k, so tile j is all
    // padding iff its first token wr_row_index[8j] is the sentinel. Feeds the
    // semi-static run() guard (see wmma.cuh): bounds j>=1 tiles to the populated
    // count so 1-tile blocks stay correct, while j=0 always runs.
    uint32_t num_token_tiles = 0;
    if constexpr (Mma::kSwapAb) {
      const uint32_t output_shape_m = shape_m * top_k;
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < Mma::kNumBColTiles; j++) {
        if (smem.wr_row_index[MmaOpClass::MmaShape::N * j] < output_shape_m) {
          num_token_tiles = j + 1;
        }
      }
    }

    producer.load_stage(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.wait_stage();
    s2r_pipe.load_stage_iter(0, 0);
    mma.transform_b(0);

    while (slice_iters) {
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        constexpr uint32_t kPartMmaShapeK = 16;
        constexpr uint32_t warp_k_iters = WarpShape::K / kPartMmaShapeK;

        PRAGMA_UNROLL
        for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_k_iter_id + 1);
          mma.run(stage_id, warp_k_iter_id, num_token_tiles);
          if (warp_k_iter_id == warp_k_iters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (slice_iters > 1) consumer.wait_stage();
              producer.load_stage(stage_id, slice_iters > kNumStages);
            } else {
              producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
              if (slice_iters > 1) consumer.wait_stage();
            }
          }

          mma.transform_b((warp_k_iter_id + 1) % 2);
        }

        slice_iters--;
        if (!slice_iters) break;
      };
    };

    __syncthreads();
    epilogue.call(mma.final_regs_c_as_ptr());
  }
};
