#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::epilogue::transform {

struct EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }
};

} // namespace deep_gemm::epilogue::transform
