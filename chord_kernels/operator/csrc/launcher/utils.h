// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include "./torch_api.h"

#include <cuda.h>
#include <cstdint>
#include <string>

inline void check_curesult(const CUresult res, const char *func_name) {
  if (res != CUDA_SUCCESS) {
    const char *err_name;
    const char *err_string;
    cuGetErrorName(res, &err_name);
    cuGetErrorString(res, &err_string);
    ASSERT_CHECK(
        false, func_name, " failed with error: ", err_name, " (",
        err_string, ")");
  }
}

inline uint32_t manual_crc32(const std::string &data) {
  static uint32_t table[256];
  static bool table_computed = false;

  if (!table_computed) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t value = i;
      for (int bit = 0; bit < 8; ++bit) {
        value = value & 1 ? 0xEDB88320L ^ (value >> 1) : value >> 1;
      }
      table[i] = value;
    }
    table_computed = true;
  }

  uint32_t crc = 0xFFFFFFFFL;
  for (unsigned char byte : data) {
    crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFL;
}

struct KernelData {
  CUmodule module;
  CUfunction func;
  uint32_t smem_size;
  uint32_t num_threads;
  uint32_t problem_shape_n;
  uint32_t problem_shape_k;
  uint32_t num_experts;
  uint32_t num_ctas_per_sm;
  bool use_stream_k;
};
