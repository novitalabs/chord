// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/memory/s2r_loader/loader_a.cuh>
#include <humming/memory/s2r_loader/loader_b.cuh>
#include <humming/memory/s2r_loader/loader_bs.cuh>


template <
    class SharedStorage, class MMA,
    class BlockShape, class WarpShape,
    class TuningConfig>
class S2RMemoryPipeline {
private:
  static constexpr bool kUseWgmma = MMA::MmaOpClass::kUseWgmma;
  static constexpr bool kSwapAb = TuningConfig::kSwapAb;
  static constexpr uint32_t kPartMmaShapeK = 16;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using MmaOpClass = typename MMA::MmaOpClass;
  using LoaderA = S2RMemoryLoaderA<
      SharedStorage, MmaOpClass, BlockShape, WarpShape>;
  using LoaderB = S2RMemoryLoaderB<
      BlockShape, WarpShape, TuningConfig>;
  using LoaderBS = S2RMemoryLoaderBS<
      MmaOpClass, BlockShape, WarpShape>;

public:
  SharedStorage &smem;
  MMA &mma;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderBS loader_bs;

  CUDA_INLINE
  S2RMemoryPipeline(SharedStorage &smem, MMA &mma)
      : smem(smem), mma(mma) {}

  CUDA_INLINE void load_stage_iter(uint32_t stage_id, uint32_t iter_id) {
    stage_id = (stage_id + iter_id / kWarpItersK) % kNumStages;
    iter_id %= kWarpItersK;
    const uint32_t buffer_id = iter_id % 2;

    loader_b.load(
        smem.stages[stage_id].b,
        mma.regs_qb_as_ptr(buffer_id),
        iter_id);
    if constexpr (kSwapAb) {
      loader_a.load_swap_act(
          smem.stages[stage_id].a,
          mma.regs_b_buf_as_ptr(buffer_id),
          iter_id);
    } else if constexpr (!kUseWgmma) {
      loader_a.load(
          smem.stages[stage_id].a,
          mma.regs_a_as_ptr(buffer_id),
          iter_id,
          stage_id);
    }
    loader_bs.load(
        smem.stages[stage_id].bs,
        mma.arith.regs_bs_as_ptr(buffer_id),
        iter_id);
  }
};
