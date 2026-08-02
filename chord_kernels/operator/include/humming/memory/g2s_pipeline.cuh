// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/memory/g2s_loader/loader_a.cuh>
#include <humming/memory/g2s_loader/loader_b.cuh>
#include <humming/memory/g2s_loader/loader_bs.cuh>
#include <humming/utils/ptx/cp_async.cuh>


template <
    class SharedStorage,
    class ProblemShape, class BlockShape,
    class TuningConfig>
class ProducerPipeline {
private:
  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using LoaderA = G2SMemoryLoaderA<
      SharedStorage, ProblemShape, BlockShape, TuningConfig>;
  using LoaderB = G2SMemoryLoaderB<
      ProblemShape, BlockShape, TuningConfig>;
  using LoaderBS = G2SMemoryLoaderBS<
      ProblemShape, BlockShape, TuningConfig>;

public:
  SharedStorage &smem;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderBS loader_bs;

  CUDA_INLINE
  ProducerPipeline(
      SharedStorage &smem,
      const void *ptr_a, const void *ptr_b, const void *ptr_bs,
      uint32_t shape_m)
      : smem(smem),
        loader_a(ptr_a, smem, shape_m),
        loader_b(ptr_b),
        loader_bs(ptr_bs) {
  }

  CUDA_INLINE void load_stage(uint32_t stage_id, bool pred = true) {
    stage_id %= kNumStages;
    if (pred) {
      loader_a.load(smem.stages[stage_id].a, stage_id);
      loader_b.load(smem.stages[stage_id].b);
      loader_bs.load(smem.stages[stage_id].bs);
    }
    cp_async_commit_group();
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id = 0) {
    loader_a.seek(k_block_id);
    loader_b.seek(expert_id, n_block_id, k_block_id);
    loader_bs.seek(expert_id, n_block_id, k_block_id);
  }
};


template <class TuningConfig>
class ConsumerPipeline {
private:
  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;

public:
  CUDA_INLINE void wait_stage() {
    cp_async_wait_group<kNumStages - 2>();
    __syncthreads();
  }
};
