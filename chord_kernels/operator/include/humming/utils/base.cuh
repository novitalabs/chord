// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <cstdint>
#include <vector_types.h>


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(a, b) ((a + b - 1) / (b))

#define STR(x) #x
#define PRAGMA_UNROLL _Pragma(STR(unroll))
#define CUDA_INLINE __device__ __forceinline__


template <typename T>
CUDA_INLINE uint32_t cast_smem_ptr_to_uint(T *smem_ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
};

template <int kBytes>
struct LoadTypeChooser;

template <>
struct LoadTypeChooser<8> {
  using Type = uint2;
};

template <>
struct LoadTypeChooser<16> {
  using Type = uint4;
};

template <uint32_t M_, uint32_t N_, uint32_t K_>
struct Shape {
  static constexpr uint32_t M = M_;
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
};
