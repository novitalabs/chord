// The indexed W4A16 GEMM (humming kernel: WGMMA on SM90, MMA / swap-AB MMA on
// SM90 / SM100 / SM103) on
// the generic C++ JIT: kernel source, compile / cache / load, registration and
// launch.  The configuration is chosen by the caller (select_indexed_config).
// No torch, no Python; errors throw std::invalid_argument for a bad
// configuration and std::runtime_error otherwise.

#pragma once

#include <cuda.h>

#include <cstdint>
#include <vector>

namespace chord::indexed {

// One kernel: problem N / K / experts plus the IndexedKernelConfig fields.
struct GemmConfig {
  int64_t shape_n, shape_k, num_experts;
  int64_t block_m, block_n, block_k;
  int64_t warp_m, warp_n, warp_k;
  int64_t num_stages;
  int64_t num_ctas_per_sm;
  bool swap_ab;
  bool use_stream_k;
  bool stream_k_fp32;  // stream-K partials reduced in an FP32 workspace (MMA and WGMMA)
};

// Validate, compile (or read the cache), load and register on the current
// device.  Returns a process-wide id; the same (device, config) returns the same id.
int64_t register_gemm(const GemmConfig& config);

// register_gemm for every config, with one progress bar over the cache misses.
std::vector<int64_t> precompile_gemm(const std::vector<GemmConfig>& configs);

// The config an id was registered with.
const GemmConfig& gemm_config(int64_t id);

struct GemmArgs {
  const void* input;              // bf16 [M, K]
  const void* weight;             // int32 [E, K/16, 2N] (pack_w4a16_indexed)
  const void* weight_scale;       // bf16 [E, K/32, N]
  const int32_t* sorted_ids;      // moe_align_block_size routing
  const int32_t* expert_ids;
  const int32_t* num_tokens_padded;
  void* output;                   // bf16 [M * top_k, N]
  int64_t shape_m;
  int64_t top_k;
  // Stream-K lock words, zero-initialized, owned by the caller and kept across
  // its launches (the kernel leaves them zeroed); required for a stream-K
  // kernel, unused otherwise.
  int32_t* locks = nullptr;
  int64_t num_locks = 0;
  // FP32 stream-K partials, gemm_streamk_workspace_elems(id) floats (no
  // zeroing needed); required when that is non-zero.
  float* streamk_workspace = nullptr;
};

// Floats of FP32 stream-K workspace one launch of kernel `id` needs on the
// current device: one block-M x block-N tile per CTA, 0 when it has none.
int64_t gemm_streamk_workspace_elems(int64_t id);

// Launch kernel `id` on `stream`; the current device must be the one it was
// registered on.
void gemm(int64_t id, const GemmArgs& args, CUstream stream);

}  // namespace chord::indexed
