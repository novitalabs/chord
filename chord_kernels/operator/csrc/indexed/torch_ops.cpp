// Thin torch ops over the pure C++ libraries (indexed: humming.h, repack.h;
// grouped: ../grouped/grouped.h): check the tensors (as the launch functions in
// csrc/launcher/launcher.cpp, with the same tensor.h helpers), take the current
// stream, and hand the pointers on.
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
#include "../grouped/grouped.h"
#include "humming.h"
#include "repack.h"

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
                    Tensor num_tokens_padded, int64_t top_k, std::optional<Tensor> locks) {
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
  if (locks.has_value()) {
    check_tensor(locks.value(), "locks", device, ScalarType::Int, {locks->numel()});
    args.locks = static_cast<int32_t*>(locks->data_ptr());
    args.num_locks = locks->numel();
  }
  // As launcher.cpp: the FP32 stream-K workspace comes from the PyTorch caching
  // allocator per launch, which recycles it in stream order and, under CUDA-graph
  // capture, draws it from the graph's private pool.
  std::optional<Tensor> streamk_ws;
  try {
    const int64_t ws_elems = chord::indexed::gemm_streamk_workspace_elems(kernel_id);
    if (ws_elems > 0) {
      streamk_ws = torch_empty({ws_elems}, ScalarType::Float, input.device());
      args.streamk_workspace = static_cast<float*>(streamk_ws->data_ptr());
    }
    chord::indexed::gemm(kernel_id, args, current_stream(device));
  } catch (const std::exception& e) {
    ASSERT_CHECK(false, e.what());
  }
  return output;
}

// output = repack(input); returns output.  input: int32 [E, N, K] codes or
// [E, N, K/8] packed words; output: int32 [E, K/16, 2N].
Tensor repack_w4a16(Tensor input, Tensor output, bool packed_input) {
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
  try {
    chord::indexed::repack_w4a16(args, current_stream(device));
  } catch (const std::exception& e) {
    ASSERT_CHECK(false, e.what());
  }
  return output;
}

// launcher.cpp check_dg_common.
void check_grouped_common(const Tensor& a, const Tensor& packed_w, const Tensor& scale,
                          int64_t num_groups, int64_t shape_n, int64_t shape_k, int64_t device) {
  ASSERT_CHECK(a.is_cuda() && a.is_contiguous(), "a must be contiguous CUDA");
  ASSERT_CHECK(a.scalar_type() == ScalarType::BFloat16, "a must be BFloat16");
  ASSERT_CHECK(a.size(-1) == shape_k, "a K mismatch");
  ASSERT_CHECK(packed_w.is_cuda() && packed_w.is_contiguous(),
               "packed weight must be contiguous CUDA");
  ASSERT_CHECK(packed_w.dim() == 3, "packed weight must be [G, N, K/2]");
  ASSERT_CHECK(packed_w.scalar_type() == ScalarType::Float8_e4m3fn,
               "packed weight must be Float8_e4m3fn, got ",
               DTYPE_TO_STRING(packed_w.scalar_type()));
  ASSERT_CHECK(packed_w.size(0) == num_groups, "weight G mismatch");
  ASSERT_CHECK(packed_w.size(1) == shape_n, "weight N mismatch");
  ASSERT_CHECK(packed_w.size(2) == shape_k / 2, "weight K/2 mismatch");
  ASSERT_CHECK(scale.is_cuda() && scale.is_contiguous(), "scale must be contiguous CUDA");
  ASSERT_CHECK(scale.scalar_type() == ScalarType::BFloat16, "scale must be BFloat16");
  ASSERT_CHECK(scale.dim() == 3 && scale.size(0) == num_groups &&
                   scale.size(1) == shape_k / 32 && scale.size(2) == shape_n,
               "scale must be MN-major BF16 [G, K/32, N]");
  ASSERT_CHECK(packed_w.get_device() == device && scale.get_device() == device,
               "weights must live on the activation device");
  ASSERT_CHECK(shape_n % 16 == 0, "N must be a multiple of 16 for the SF TMA");
}

void run_grouped(int64_t kernel_id, int64_t expected_type, chord::grouped::GemmArgs& args,
                 int64_t device) {
  try {
    ASSERT_CHECK(chord::grouped::gemm_type(kernel_id) == expected_type,
                 expected_type == 2 ? "kernel was compiled for the contiguous path"
                                    : "kernel was compiled for the masked path");
    chord::grouped::gemm(kernel_id, args, current_stream(device));
  } catch (const std::exception& e) {
    ASSERT_CHECK(false, e.what());
  }
}

// launcher.cpp launch_grouped_w4a16_masked.
Tensor grouped_masked(int64_t kernel_id, Tensor a, Tensor packed_w, Tensor scale, Tensor masked_m,
                      std::optional<Tensor> out_, bool enable_pdl) {
  ASSERT_CHECK(a.dim() == 3, "masked a must be [G, max_m, K]");
  const int64_t device = a.get_device();
  check_current_device(device);
  const int64_t num_groups = a.size(0), shape_m = a.size(1), shape_k = a.size(2);
  const int64_t shape_n = packed_w.size(1);
  check_grouped_common(a, packed_w, scale, num_groups, shape_n, shape_k, device);
  ASSERT_CHECK(masked_m.is_cuda() && masked_m.is_contiguous() &&
                   masked_m.scalar_type() == ScalarType::Int && masked_m.numel() == num_groups,
               "masked_m must be contiguous CUDA int32 [G]");
  ASSERT_CHECK(shape_k % 32 == 0, "K must be a multiple of the scale group");
  Tensor d = out_.has_value()
                 ? out_.value()
                 : torch_empty({num_groups, shape_m, shape_n}, ScalarType::BFloat16, a.device());
  ASSERT_CHECK(d.is_contiguous() && d.dim() == 3 && d.size(0) == num_groups &&
                   d.size(1) == shape_m && d.size(2) == shape_n &&
                   d.scalar_type() == ScalarType::BFloat16,
               "out must be BFloat16 [G, max_m, N]");
  chord::grouped::GemmArgs args = {a.data_ptr(), packed_w.data_ptr(), scale.data_ptr(),
                                   static_cast<int32_t*>(masked_m.data_ptr()), d.data_ptr(),
                                   shape_m, shape_n, shape_k, num_groups, enable_pdl};
  run_grouped(kernel_id, 2, args, device);
  return d;
}

// launcher.cpp launch_grouped_w4a16_contiguous.
Tensor grouped_contiguous(int64_t kernel_id, Tensor a, Tensor packed_w, Tensor scale,
                          Tensor m_indices, std::optional<Tensor> out_, bool enable_pdl) {
  ASSERT_CHECK(a.dim() == 2, "contiguous a must be [m, K]");
  const int64_t device = a.get_device();
  check_current_device(device);
  const int64_t shape_m = a.size(0), shape_k = a.size(1);
  const int64_t num_groups = packed_w.size(0), shape_n = packed_w.size(1);
  check_grouped_common(a, packed_w, scale, num_groups, shape_n, shape_k, device);
  ASSERT_CHECK(m_indices.is_cuda() && m_indices.is_contiguous() &&
                   m_indices.scalar_type() == ScalarType::Int && m_indices.numel() == shape_m,
               "m_indices must be contiguous CUDA int32 [m]");
  ASSERT_CHECK(shape_m % 128 == 0,
               "contiguous m must be 128-aligned (per-expert padded), got ", shape_m);
  Tensor d = out_.has_value() ? out_.value()
                              : torch_empty({shape_m, shape_n}, ScalarType::BFloat16, a.device());
  ASSERT_CHECK(d.is_contiguous() && d.dim() == 2 && d.size(0) == shape_m &&
                   d.size(1) == shape_n && d.scalar_type() == ScalarType::BFloat16,
               "out must be BFloat16 [m, N]");
  chord::grouped::GemmArgs args = {a.data_ptr(), packed_w.data_ptr(), scale.data_ptr(),
                                   static_cast<int32_t*>(m_indices.data_ptr()), d.data_ptr(),
                                   shape_m, shape_n, shape_k, num_groups, enable_pdl};
  run_grouped(kernel_id, 1, args, device);
  return d;
}

}  // namespace

COMMON_TORCH_LIBRARY(chord_cpp, m) {
  m.def(
      "indexed_gemm(int kernel_id, Tensor input, Tensor weight, Tensor? output, "
      "Tensor weight_scale, Tensor sorted_ids, Tensor expert_ids, "
      "Tensor num_tokens_padded, int top_k, Tensor? locks=None) -> Tensor");
  m.def("repack_w4a16(Tensor input, Tensor output, bool packed_input) -> Tensor");
  m.def(
      "grouped_masked(int kernel_id, Tensor a, Tensor packed_weight, Tensor scale, "
      "Tensor masked_m, Tensor? output, bool enable_pdl) -> Tensor");
  m.def(
      "grouped_contiguous(int kernel_id, Tensor a, Tensor packed_weight, Tensor scale, "
      "Tensor m_indices, Tensor? output, bool enable_pdl) -> Tensor");
}

COMMON_TORCH_LIBRARY_IMPL(chord_cpp, CUDA, m) {
  m.impl("indexed_gemm", COMMON_TORCH_BOX(&indexed_gemm));
  m.impl("repack_w4a16", COMMON_TORCH_BOX(&repack_w4a16));
  m.impl("grouped_masked", COMMON_TORCH_BOX(&grouped_masked));
  m.impl("grouped_contiguous", COMMON_TORCH_BOX(&grouped_contiguous));
}

// Registration and startup precompile (cpp_jit.py).  Not torch ops: errors are
// plain C++ exceptions, which pybind11 raises as ValueError (bad config,
// std::invalid_argument) or RuntimeError.  Each makes `device` current
// (creating its primary context if needed), then the library does the rest.
namespace {

constexpr size_t kConfigFields = 14;

chord::indexed::GemmConfig to_config(const std::vector<int64_t>& v) {
  if (v.size() != kConfigFields) {
    throw std::invalid_argument("an indexed config has " + std::to_string(kConfigFields) +
                                " fields, got " + std::to_string(v.size()));
  }
  return {v[0], v[1], v[2],  v[3],  v[4],      v[5],      v[6],
          v[7], v[8], v[9], v[10], v[11] != 0, v[12] != 0, v[13] != 0};
}

void set_device(int64_t device) {
  if (cudaSetDevice(static_cast<int>(device)) != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice failed");
  }
}

// config: (shape_n, shape_k, num_experts, block_m, block_n, block_k, warp_m,
// warp_n, warp_k, num_stages, num_ctas_per_sm, swap_ab, use_stream_k,
// stream_k_fp32).
int64_t register_gemm(int64_t device, const std::vector<int64_t>& config) {
  set_device(device);
  return chord::indexed::register_gemm(to_config(config));
}

// grouped config: (gemm_type, shape_n, shape_k, num_groups, num_sms, block_m,
// block_n, block_k, swizzle_a, swizzle_b, swizzle_cd, num_stages, smem_size,
// num_tma_threads, num_math_threads, num_threads, cluster_size, multicast_on_a).
int64_t register_grouped(int64_t device, const std::vector<int64_t>& v) {
  if (v.size() != 18) throw std::invalid_argument("a grouped config has 18 fields");
  set_device(device);
  return chord::grouped::register_gemm({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                                        v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16],
                                        v[17] != 0});
}

std::vector<int64_t> precompile_gemm(int64_t device,
                                     const std::vector<std::vector<int64_t>>& configs) {
  set_device(device);
  std::vector<chord::indexed::GemmConfig> list;
  for (const auto& c : configs) list.push_back(to_config(c));
  return chord::indexed::precompile_gemm(list);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_gemm", &register_gemm, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("precompile_gemm", &precompile_gemm,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("register_grouped", &register_grouped,
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
