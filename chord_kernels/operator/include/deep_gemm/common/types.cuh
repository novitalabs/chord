#pragma once

#include <cutlass/detail/helper_macros.hpp>

namespace deep_gemm {

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
};

} // namespace deep_gemm
