// Configuration choice of the indexed W4A16 GEMM on SM100 / SM103: what
// chord's Python dispatch (dispatch.forward_w4a16, tuning.py) picks per call,
// for the profiles that reach Blackwell.  The result goes to register_gemm
// (humming.h).  No torch, no Python, no CUDA; a bad request throws
// std::invalid_argument.

#pragma once

#include <cstdint>
#include <vector>

#include "humming.h"

namespace chord::indexed {

// The static part of one indexed layer (IndexedLayerMeta, MMA layout).
struct LayerDesc {
  int64_t shape_n, shape_k, num_experts;
  bool blackwell_table;    // profile is blackwell_decode_ep8: its published table applies
  int64_t profile_block_m;  // the profile's fallback tile
  bool profile_swap_ab;
};

// One row of a framework tuning table (vLLM's tuning_config): routed-M in
// (lower, upper], and the config fields (IndexedKernelConfig, already checked).
struct TableRow {
  int64_t lower, upper;
  int64_t block_m, block_n, block_k;
  int64_t warp_m, warp_n, warp_k;
  int64_t num_stages, num_ctas_per_sm;
  bool swap_ab, use_stream_k;
};

// IndexedKernelConfig.default for the MMA layout, block-N from N.
GemmConfig default_config(int64_t shape_n, int64_t shape_k, int64_t num_experts, int64_t block_m,
                          bool swap_ab);

// Store a tuning table; returns its id for select_config.
int64_t register_table(const std::vector<TableRow>& rows);

// The config for routed-M `m`: the profile's published table / fallback, or
// table `table` when table >= 0, then the routing block-M override when
// `has_block_m` (dispatch.forward_w4a16).
GemmConfig select_config(const LayerDesc& layer, int64_t m, int64_t table, bool has_block_m,
                         int64_t block_m);

}  // namespace chord::indexed
