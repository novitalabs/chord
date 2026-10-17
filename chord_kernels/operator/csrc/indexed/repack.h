// The repack_w4a16 kernel (INT4 checkpoint -> indexed kernel weight layout) on
// the generic C++ JIT.  No torch, no Python; errors throw std::runtime_error.

#pragma once

#include <cuda.h>

#include <cstdint>

namespace chord::indexed {

struct RepackArgs {
  const void* input;  // int32 [E, N, K] codes, or [E, N, K/8] packed words
  void* output;       // int32 [E, K/16, 2N]
  int64_t num_experts, shape_n, shape_k;  // shape_k: logical K
  bool packed_input;  // MMA and WGMMA read the same layout: the input form is the only choice
};

// Compile (or read the cache), load and launch on `stream`.
void repack_w4a16(const RepackArgs& args, CUstream stream);

}  // namespace chord::indexed
