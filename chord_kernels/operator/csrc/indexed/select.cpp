// Configuration choice of the indexed W4A16 GEMM; see select.h.  Ported from
// tuning.py (_indexed_config, the blackwell_decode_ep8 rows of
// _select_indexed_kernel_config, _resolve_tuning_config), api.py
// (_compatible_block_n, IndexedKernelConfig.default) and dispatch.py
// (_config_with_block_m).  Keep each in step with its Python counterpart: the
// Python one still serves CHORD_CPP_JIT=0, SM90 and the published tuning rows.

#include "select.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>

namespace chord::indexed {

namespace {

[[noreturn]] void invalid(const std::string& message) { throw std::invalid_argument(message); }

std::string str(int64_t v) { return std::to_string(v); }

// api.py _compatible_block_n
int64_t compatible_block_n(int64_t shape_n) {
  const int64_t capped = std::min<int64_t>(shape_n, 256);
  int64_t candidate = 1;
  while (candidate * 2 <= capped) candidate *= 2;
  while (candidate > 1 && shape_n % candidate) candidate /= 2;
  if (candidate < 128) {
    invalid("shape_n must have a compatible power-of-two tile >= 128, got " + str(shape_n));
  }
  return candidate;
}

// api.py _default_stream_k_fp32: FP32 wherever the kernel implements it, which
// on the MMA layout (warp-N 64) is every stream-K config.
// tuning.py _indexed_config, MMA layout (warp-N 64).
GemmConfig indexed_config(const LayerDesc& l, int64_t block_m, int64_t block_n,
                          int64_t num_ctas_per_sm, bool swap_ab, bool use_stream_k,
                          int64_t block_k = 64) {
  return {l.shape_n, l.shape_k,         l.num_experts, block_m, block_n, block_k, block_m,
          64,        std::min<int64_t>(block_k, 64), 4, num_ctas_per_sm, swap_ab, use_stream_k,
          use_stream_k};
}

// tuning.py _select_indexed_kernel_config, for the profiles that reach SM100 / SM103.
GemmConfig profile_config(const LayerDesc& l, int64_t valid_shape_m) {
  if (valid_shape_m < 0) invalid("valid_shape_m must be non-negative, got " + str(valid_shape_m));
  const int64_t m = std::max<int64_t>(valid_shape_m, 1);
  const bool gate_up = l.shape_n == 4096 && l.shape_k == 7168;
  const bool down = l.shape_n == 7168 && l.shape_k == 2048;
  if (l.blackwell_table && l.num_experts == 48 && (gate_up || down)) {
    if (down) {
      if (m <= 272) return indexed_config(l, 8, 256, 4, true, false);
      if (m <= 848) return indexed_config(l, 16, 256, 4, true, false);
      if (m <= 1424) return indexed_config(l, 32, 512, 1, false, true);
      return indexed_config(l, 48, 512, 1, false, true);
    }
    if (m <= 320) return indexed_config(l, 8, 256, 4, true, true);
    if (m <= 736) return indexed_config(l, 16, 512, 1, false, true, 128);
    if (m <= 1424) return indexed_config(l, 32, 512, 1, false, true);
    return indexed_config(l, 48, 512, 1, false, true);
  }
  return default_config(l.shape_n, l.shape_k, l.num_experts, l.profile_block_m,
                        l.profile_swap_ab);
}

std::mutex g_mutex;  // guards g_tables
std::deque<std::vector<TableRow>> g_tables;

// tuning.py _resolve_tuning_config, a list of rows.
GemmConfig table_config(const LayerDesc& l, int64_t table, int64_t m) {
  std::lock_guard<std::mutex> hold(g_mutex);
  if (table >= static_cast<int64_t>(g_tables.size())) {
    throw std::runtime_error("indexed tuning table is not registered: " + str(table));
  }
  for (const TableRow& r : g_tables[table]) {
    if (m > r.lower && m <= r.upper) {
      return {l.shape_n,  l.shape_k,  l.num_experts, r.block_m,    r.block_n,
              r.block_k,  r.warp_m,   r.warp_n,      r.warp_k,     r.num_stages,
              r.num_ctas_per_sm, r.swap_ab, r.use_stream_k, r.stream_k_fp32};
    }
  }
  invalid("no indexed tuning row covers valid_shape_m=" + str(m));
}

// dispatch.py _config_with_block_m, MMA layout.
GemmConfig with_block_m(GemmConfig c, int64_t block_m) {
  if (block_m == c.block_m) return c;
  if (block_m <= 0 || block_m % 8) {
    invalid("routing block_m must be a positive multiple of 8, got " + str(block_m));
  }
  // MMA's non-swap path cannot issue an m8n8k8 BF16 instruction.
  if (block_m % 16 == 8) c.swap_ab = true;
  c.block_m = block_m;
  c.warp_m = block_m;
  return c;
}

}  // namespace

GemmConfig default_config(int64_t shape_n, int64_t shape_k, int64_t num_experts, int64_t block_m,
                          bool swap_ab) {
  return {shape_n, shape_k, num_experts, block_m, compatible_block_n(shape_n), 64, block_m,
          64,      64,      4,           swap_ab ? 4 : 1, swap_ab, false, false};
}

int64_t register_table(const std::vector<TableRow>& rows) {
  std::lock_guard<std::mutex> hold(g_mutex);
  g_tables.push_back(rows);
  return static_cast<int64_t>(g_tables.size()) - 1;
}

GemmConfig select_config(const LayerDesc& layer, int64_t m, int64_t table, bool has_block_m,
                         int64_t block_m) {
  // The Python path builds the profile default first, even when a table follows.
  GemmConfig c = profile_config(layer, m);
  if (table >= 0) c = table_config(layer, table, m);
  if (has_block_m) c = with_block_m(c, block_m);
  return c;
}

}  // namespace chord::indexed
