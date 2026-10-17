#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    CUTLASS_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    CUTLASS_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

} // namespace deep_gemm::utils
