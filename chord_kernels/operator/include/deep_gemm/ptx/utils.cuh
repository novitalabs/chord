#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::ptx {

CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

} // namespace deep_gemm::ptx
