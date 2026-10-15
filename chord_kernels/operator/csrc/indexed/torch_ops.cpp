// Thin torch ops over the pure C++ indexed library (humming.h, repack.h):
// check the tensors (as launch_kernel in csrc/launcher/launcher.cpp, with the
// same tensor.h helpers), take the current stream, and hand the pointers on.
// Built by chord_kernels/operator/cpp_jit.py.

#define USE_CUDA 1

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../launcher/tensor.h"
#include "humming.h"
#include "repack.h"
#include "select.h"

namespace {

cudaStream_t current_stream(int64_t device) {
#if USE_TORCH_STABLE_API
  void* stream_ptr = nullptr;
  aoti_torch_get_current_cuda_stream(device, &stream_ptr);
  return static_cast<cudaStream_t>(stream_ptr);
#else
  return at::cuda::getCurrentCUDAStream(device);
#endif
}

void check_current_device(int64_t device) {
  int current = -1;
  ASSERT_CHECK(cudaGetDevice(&current) == cudaSuccess, "cudaGetDevice failed");
  ASSERT_CHECK(current == device, "tensors are on device ", device, " but the current device is ",
               current);
}

Tensor indexed_gemm(int64_t kernel_id, Tensor input, Tensor weight, std::optional<Tensor> output_,
                    Tensor weight_scale, Tensor sorted_ids, Tensor expert_ids,
                    Tensor num_tokens_padded, int64_t top_k) {
  ASSERT_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  ASSERT_CHECK(input.dim() == 2, "input must be a 2D tensor");
  ASSERT_CHECK(top_k > 0, "top_k must be positive");
  const int64_t device = input.get_device();
  check_current_device(device);
  KernelData kernel_data = {};
  try {
    const chord::indexed::GemmConfig& config = chord::indexed::gemm_config(kernel_id);
    kernel_data.problem_shape_n = static_cast<uint32_t>(config.shape_n);
    kernel_data.problem_shape_k = static_cast<uint32_t>(config.shape_k);
    kernel_data.num_experts = static_cast<uint32_t>(config.num_experts);
  } catch (const std::exception& e) {
    ASSERT_CHECK(false, e.what());
  }
  Tensor output = make_output_tensor(output_, input, kernel_data, top_k);
  check_indexed_w4a16_tensors(input, weight, output, weight_scale, sorted_ids, expert_ids,
                              num_tokens_padded, kernel_data, top_k);

  chord::indexed::GemmArgs args;
  args.input = input.data_ptr();
  args.weight = weight.data_ptr();
  args.weight_scale = weight_scale.data_ptr();
  args.sorted_ids = static_cast<const int32_t*>(sorted_ids.data_ptr());
  args.expert_ids = static_cast<const int32_t*>(expert_ids.data_ptr());
  args.num_tokens_padded = static_cast<const int32_t*>(num_tokens_padded.data_ptr());
  args.output = output.data_ptr();
  args.shape_m = input.size(0);
  args.top_k = top_k;
  try {
    chord::indexed::gemm(kernel_id, args, current_stream(device));
  } catch (const std::exception& e) {
    ASSERT_CHECK(false, e.what());
  }
  return output;
}

// output = repack(input); returns output.  input: int32 [E, N, K] codes or
// [E, N, K/8] packed words; output: int32 [E, K/16, 2N].
Tensor repack_w4a16(Tensor input, Tensor output, bool packed_input, bool wgmma_layout) {
  ASSERT_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 3 &&
                   input.scalar_type() == ScalarType::Int,
               "repack input must be a contiguous CUDA int32 [E, N, K] (or [E, N, K/8]) tensor");
  const int64_t device = input.get_device();
  check_current_device(device);
  const int64_t num_experts = input.size(0);
  const int64_t shape_n = input.size(1);
  const int64_t shape_k = packed_input ? input.size(2) * 8 : input.size(2);
  check_tensor(output, "repack output", device, ScalarType::Int,
               {num_experts, shape_k / 16, shape_n * 2});

  chord::indexed::RepackArgs args;
  args.input = input.data_ptr();
  args.output = output.data_ptr();
  args.num_experts = num_experts;
  args.shape_n = shape_n;
  args.shape_k = shape_k;
  args.packed_input = packed_input;
  args.wgmma_layout = wgmma_layout;
  try {
    chord::indexed::repack_w4a16(args, current_stream(device));
  } catch (const std::exception& e) {
    ASSERT_CHECK(false, e.what());
  }
  return output;
}

}  // namespace

COMMON_TORCH_LIBRARY(chord_cpp, m) {
  m.def(
      "indexed_gemm(int kernel_id, Tensor input, Tensor weight, Tensor? output, "
      "Tensor weight_scale, Tensor sorted_ids, Tensor expert_ids, "
      "Tensor num_tokens_padded, int top_k) -> Tensor");
  m.def("repack_w4a16(Tensor input, Tensor output, bool packed_input, bool wgmma_layout) -> Tensor");
}

COMMON_TORCH_LIBRARY_IMPL(chord_cpp, CUDA, m) {
  m.impl("indexed_gemm", COMMON_TORCH_BOX(&indexed_gemm));
  m.impl("repack_w4a16", COMMON_TORCH_BOX(&repack_w4a16));
}

// Registration and startup precompile (cpp_jit.py).  Not torch ops: errors are
// plain C++ exceptions, which pybind11 raises as ValueError (bad config,
// std::invalid_argument) or RuntimeError.  Each makes `device` current
// (creating its primary context if needed), then the library does the rest.
namespace {

constexpr size_t kConfigFields = 13;

chord::indexed::GemmConfig to_config(const std::vector<int64_t>& v) {
  if (v.size() != kConfigFields) {
    throw std::invalid_argument("an indexed config has " + std::to_string(kConfigFields) +
                                " fields, got " + std::to_string(v.size()));
  }
  return {v[0], v[1], v[2],  v[3],  v[4],      v[5],      v[6],
          v[7], v[8], v[9], v[10], v[11] != 0, v[12] != 0};
}

void set_device(int64_t device) {
  if (cudaSetDevice(static_cast<int>(device)) != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice failed");
  }
}

// config: (shape_n, shape_k, num_experts, block_m, block_n, block_k, warp_m,
// warp_n, warp_k, num_stages, num_ctas_per_sm, swap_ab, use_stream_k).
int64_t register_gemm(int64_t device, const std::vector<int64_t>& config) {
  set_device(device);
  return chord::indexed::register_gemm(to_config(config));
}

std::vector<int64_t> precompile_gemm(int64_t device,
                                     const std::vector<std::vector<int64_t>>& configs) {
  set_device(device);
  std::vector<chord::indexed::GemmConfig> list;
  for (const auto& c : configs) list.push_back(to_config(c));
  return chord::indexed::precompile_gemm(list);
}

// Configuration choice (select.h).  A config goes back as its 10
// IndexedKernelConfig fields: block_m, block_n, block_k, warp_m, warp_n,
// warp_k, num_stages, num_ctas_per_sm, swap_ab, use_stream_k.
std::vector<int64_t> config_fields(const chord::indexed::GemmConfig& c) {
  return {c.block_m, c.block_n, c.block_k,         c.warp_m,  c.warp_n,
          c.warp_k,  c.num_stages, c.num_ctas_per_sm, c.swap_ab, c.use_stream_k};
}

// rows: (lower, upper, then the 10 config fields above) each.
int64_t register_table(const std::vector<std::vector<int64_t>>& rows) {
  std::vector<chord::indexed::TableRow> list;
  for (const auto& r : rows) {
    if (r.size() != 12) throw std::invalid_argument("a tuning row has 12 fields");
    list.push_back({r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10] != 0,
                    r[11] != 0});
  }
  return chord::indexed::register_table(list);
}

// layer: (shape_n, shape_k, num_experts, blackwell_table, profile_block_m, profile_swap_ab).
std::vector<int64_t> select_config(const std::vector<int64_t>& layer, int64_t m, int64_t table,
                                   bool has_block_m, int64_t block_m) {
  if (layer.size() != 6) throw std::invalid_argument("a layer has 6 fields");
  const chord::indexed::LayerDesc desc = {layer[0],      layer[1], layer[2],
                                          layer[3] != 0, layer[4], layer[5] != 0};
  return config_fields(chord::indexed::select_config(desc, m, table, has_block_m, block_m));
}

std::vector<int64_t> default_config(int64_t shape_n, int64_t shape_k, int64_t num_experts,
                                    int64_t block_m, bool swap_ab) {
  return config_fields(
      chord::indexed::default_config(shape_n, shape_k, num_experts, block_m, swap_ab));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_gemm", &register_gemm, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("precompile_gemm", &precompile_gemm,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("register_table", &register_table);
  m.def("select_config", &select_config);
  m.def("default_config", &default_config);
}
