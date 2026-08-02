// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/utils/base.cuh>

#include <type_traits>

template <class MmaOpClass, class BlockShape, class WarpShape, class TuningConfig>
class EpilogueSmemReducer {
private:
  using MmaShape = typename MmaOpClass::MmaShape;

  static constexpr bool kUseWgmma = MmaOpClass::kUseWgmma;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using MMA_CRegistersArrayType = CRegistersType[MAX(WarpShape::M / MmaShape::M, 1)][MAX(WarpShape::N / MmaShape::N, 1)];
  using WGMMA_CRegistersArrayType = CRegistersType[WarpShape::N * 4 / MmaShape::N][MAX(WarpShape::M / MmaShape::M, 1)];
  using CRegistersArrayType = std::conditional_t<kUseWgmma, WGMMA_CRegistersArrayType, MMA_CRegistersArrayType>;

public:
  int4 *smem_ptr;

  CUDA_INLINE
  EpilogueSmemReducer(int4 *smem_ptr)
      : smem_ptr(smem_ptr) {}

  CUDA_INLINE
  void reduce(uint32_t *regs_ptr) {
    constexpr uint32_t num_int4s = sizeof(CRegistersArrayType) / 16;
    constexpr uint32_t num_int4s_per_time = num_int4s / MAX(WarpShape::M / 16, 1);
    constexpr uint32_t group_num_warps = BlockShape::K / WarpShape::K;
    constexpr uint32_t num_groups = TuningConfig::kNumThreads / 32 / group_num_warps;
    uint32_t group_id = threadIdx.x / 32 % num_groups;
    uint32_t group_warp_id = threadIdx.x / (32 * num_groups);
    uint32_t laneid = threadIdx.x % 32;

    using ReductionSmemType = int4[group_num_warps / 2][num_groups][num_int4s_per_time][32];
    auto &smem_arr = *reinterpret_cast<ReductionSmemType *>(smem_ptr);

    auto write_to_smem = [&](uint32_t buffer_id, uint32_t m) {
      int4 *regs_int4_ptr = reinterpret_cast<int4 *>(regs_ptr) + m * num_int4s_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_int4s_per_time; i++) {
        smem_arr[buffer_id][group_id][i][laneid] = regs_int4_ptr[i];
      };
    };

    auto read_from_smem_and_reduce = [&](uint32_t buffer_id, uint32_t m) {
      int4 *regs_int4_ptr = reinterpret_cast<int4 *>(regs_ptr) + m * num_int4s_per_time;

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_int4s_per_time; i++) {
        int4 val = smem_arr[buffer_id][group_id][i][laneid];

        float *sval_scalar_ptr = reinterpret_cast<float *>(&val);
        float *regs_scalar_ptr = reinterpret_cast<float *>(regs_int4_ptr + i);

        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 4; j++) {
          regs_scalar_ptr[j] += sval_scalar_ptr[j];
        }
      };
    };

    PRAGMA_UNROLL
    for (uint32_t m = 0; m < MAX(WarpShape::M / 16, 1); m++) {
      PRAGMA_UNROLL
      for (uint32_t i = 1; i < group_num_warps; i *= 2) {
        uint32_t buffer_id = group_warp_id % (group_num_warps / (2 * i));
        if (group_warp_id >= group_num_warps / i) {
          __syncthreads();
        } else if (group_warp_id >= group_num_warps / (2 * i)) {
          write_to_smem(buffer_id, m);
          __syncthreads();
        } else {
          __syncthreads();
          read_from_smem_and_reduce(buffer_id, m);
        };

        __syncthreads();
      };
    };
  };
};
