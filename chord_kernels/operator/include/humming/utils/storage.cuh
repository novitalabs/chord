// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/utils/base.cuh>


template <
    class BlockShape, class WarpShape,
    class TuningConfig>
struct SharedStorage {
  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  static constexpr uint32_t kStageSizeA =
      BlockShape::M * BlockShape::K / 8;
  static constexpr uint32_t kStageSizeB =
      BlockShape::K * BlockShape::N / 32;
  static constexpr uint32_t kStageSizeBS =
      CEIL_DIV(BlockShape::K, 32) * BlockShape::N / 8;

  static constexpr uint32_t kMWarps = BlockShape::M / WarpShape::M;
  static constexpr uint32_t kKWarps = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kWarpReduceSize =
      kMWarps * 16 * BlockShape::N * 32 / 128 * (kKWarps / 2);
  static constexpr uint32_t kBlockOutputSize =
      BlockShape::M * BlockShape::N / 8;

  struct StageStorage {
    alignas(128) int4 a[kStageSizeA];
    alignas(128) int4 b[kStageSizeB];
    alignas(128) int4 bs[kStageSizeBS];
  };

  union alignas(1024) {
    StageStorage stages[kNumStages];
    alignas(128) int4 reduce[MAX(kWarpReduceSize, kBlockOutputSize)];
  };

  uint32_t rd_row_index[BlockShape::M];
  uint32_t wr_row_index[BlockShape::M];
};
