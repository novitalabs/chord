// The SM90 grouped W4A16 GEMM (vendored DeepGEMM kernel, masked decode and
// contiguous prefill) on the generic C++ JIT: kernel source, compile / cache /
// load, registration and launch.  The launch config is chosen by the caller
// (grouped/heuristics.py).  No torch, no Python; errors throw
// std::invalid_argument for a bad configuration and std::runtime_error otherwise.

#pragma once

#include <cuda.h>

#include <cstdint>

namespace chord::grouped {

// One kernel: the problem desc plus the W4A16GemmConfig fields the cubin bakes in.
struct GemmConfig {
  int64_t gemm_type;  // 1 = contiguous (prefill), 2 = masked (decode)
  int64_t shape_n, shape_k, num_groups, num_sms;
  int64_t block_m, block_n, block_k;
  int64_t swizzle_a, swizzle_b, swizzle_cd;
  int64_t num_stages, smem_size;
  int64_t num_tma_threads, num_math_threads, num_threads;
  int64_t cluster_size;
  bool multicast_on_a;
};

// Validate, compile (or read the cache), load and register on the current
// device.  Returns a process-wide id; the same (device, config) returns the same id.
int64_t register_gemm(const GemmConfig& config);

struct GemmArgs {
  void* a;               // bf16; masked [G, max_m, K], contiguous [m, K]
  void* weight;          // float8_e4m3fn bytes [G, N, K/2] (pack_w4a16_grouped)
  void* scale;           // bf16 [G, K/32, N]
  int32_t* layout;       // masked_m [G] or m_indices [m]
  void* d;               // bf16; masked [G, max_m, N], contiguous [m, N]
  int64_t m;             // masked: per-group max_m; contiguous: total rows
  int64_t n, k;
  int64_t num_groups;
  bool enable_pdl;
};

// Launch kernel `id` on `stream`; the current device must be the one it was
// registered on.
void gemm(int64_t id, const GemmArgs& args, CUstream stream);

// The gemm_type an id was registered with.
int64_t gemm_type(int64_t id);

}  // namespace chord::grouped
