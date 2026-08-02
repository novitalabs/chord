// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include "./torch_api.h"
#include "./utils.h"

#include <optional>
#include <vector>

inline Tensor make_output_tensor(
    std::optional<Tensor> &output,
    const Tensor &input,
    const KernelData &kernel_data,
    int64_t top_k) {
  if (output.has_value()) return output.value();
  const int64_t output_m = input.size(0) * top_k;
  const int64_t output_n = kernel_data.problem_shape_n;
  return torch_empty(
      {output_m, output_n}, ScalarType::BFloat16, input.device());
}

inline void check_tensor(
    const Tensor &tensor,
    const char *name,
    int64_t expected_device,
    ScalarType expected_dtype,
    const std::vector<int64_t> &expected_shape) {
  ASSERT_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  ASSERT_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  ASSERT_CHECK(
      tensor.get_device() == expected_device, name,
      " must be on the input device");
  ASSERT_CHECK(
      tensor.scalar_type() == expected_dtype, name, " has an invalid dtype: ",
      DTYPE_TO_STRING(tensor.scalar_type()), " != ",
      DTYPE_TO_STRING(expected_dtype));
  ASSERT_CHECK(
      tensor.dim() == static_cast<int64_t>(expected_shape.size()), name,
      " has an invalid rank");
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    ASSERT_CHECK(
        tensor.size(dim) == expected_shape[dim], name, ".size(", dim,
        ") is invalid: ", tensor.size(dim), " != ", expected_shape[dim]);
  }
}

inline void check_indexed_w4a16_tensors(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &output,
    const Tensor &weight_scale,
    const Tensor &sorted_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_padded,
    const KernelData &kernel_data,
    int64_t top_k) {
  const int64_t device = input.get_device();
  const int64_t shape_m = input.size(0);
  const int64_t shape_n = kernel_data.problem_shape_n;
  const int64_t shape_k = kernel_data.problem_shape_k;
  const int64_t num_experts = kernel_data.num_experts;

  check_tensor(
      input, "input", device, ScalarType::BFloat16, {shape_m, shape_k});
  check_tensor(
      weight, "weight", device, ScalarType::Int,
      {num_experts, shape_k / 16, shape_n * 2});
  check_tensor(
      output, "output", device, ScalarType::BFloat16,
      {shape_m * top_k, shape_n});
  check_tensor(
      weight_scale, "weight_scale", device, ScalarType::BFloat16,
      {num_experts, shape_k / 32, shape_n});

  check_tensor(
      sorted_ids, "sorted_ids", device, ScalarType::Int,
      {sorted_ids.numel()});
  check_tensor(
      expert_ids, "expert_ids", device, ScalarType::Int,
      {expert_ids.numel()});
  ASSERT_CHECK(
      num_tokens_padded.dim() == 0 ||
          (num_tokens_padded.dim() == 1 && num_tokens_padded.numel() == 1),
      "num_tokens_padded must be a scalar or one-element tensor");
  check_tensor(
      num_tokens_padded, "num_tokens_padded", device, ScalarType::Int,
      num_tokens_padded.dim() == 0 ? std::vector<int64_t>{}
                                   : std::vector<int64_t>{1});
}
