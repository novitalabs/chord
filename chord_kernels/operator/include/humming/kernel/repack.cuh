// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#pragma once

#include <humming/utils/base.cuh>


CUDA_INLINE void pack_w4a16_fragment(
    const uint32_t *input, uint32_t *output, uint32_t interleave_mode) {
  auto interleaved_index = [](uint32_t index) {
    return (index % 4) * 2 + index / 4;
  };

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; ++i) {
    PRAGMA_UNROLL
    for (uint32_t bit = 0; bit < 4; ++bit) {
      uint32_t packed = 0;
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < 8; ++k) {
        const uint32_t permuted_k = interleaved_index(k);
        const uint32_t low_source =
            interleave_mode % 2 == 0 ? k : permuted_k;
        const uint32_t high_source =
            interleave_mode / 2 == 0 ? k : permuted_k;
        const uint32_t base = i * 32 + bit * 8;
        packed |= (input[base + low_source] & 0x7) << (k * 4);
        packed |= (input[base + high_source] & 0x8) << (k * 4);
      }
      output[i * 4 + bit] = packed;
    }
  }
}


template <bool kPackedInput>
CUDA_INLINE uint32_t load_uint4(const uint32_t *row, uint32_t index) {
  if constexpr (kPackedInput) {
    return (row[index / 8] >> ((index % 8) * 4)) & 0xF;
  }
  return row[index];
}


template <bool kPackedInput, bool kTransposeMiniBlock>
__global__ void repack_w4a16(
    const uint32_t *input, uint32_t *output,
    uint32_t shape_n, uint32_t shape_k,
    uint32_t padded_shape_n, uint32_t padded_shape_k,
    uint32_t interleave_mode) {
  constexpr uint32_t kInputBits = kPackedInput ? 4 : 32;
  constexpr uint32_t kSmemStride = 64 * kInputBits / 32;
  const uint32_t gmem_stride = shape_k * kInputBits / 32;

  static_assert(kSmemStride == 8 || kSmemStride == 64);
  assert(padded_shape_n % 64 == 0);
  // K % 32 is the BF16-activation packing requirement; it also covers the
  // padded_shape_k % 16 == 0 the output-row arithmetic below needs.
  assert(padded_shape_k % 32 == 0);
  assert(blockDim.x == 32);

  __shared__ uint32_t smem[64][kSmemStride];
  uint32_t *smem_flat = reinterpret_cast<uint32_t *>(smem);
  constexpr uint32_t kLoadIterations = sizeof(smem) / sizeof(uint32_t) / 32;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kLoadIterations; ++i) {
    const uint32_t smem_offset = i * 32 + threadIdx.x;
    const uint32_t row = smem_offset / kSmemStride + blockIdx.x * 64;
    const uint32_t col =
        smem_offset % kSmemStride + blockIdx.y * kSmemStride;
    const uint64_t gmem_offset =
        row * gmem_stride + col +
        blockIdx.z * static_cast<uint64_t>(gmem_stride) * shape_n;
    smem_flat[smem_offset] =
        row < shape_n && col < gmem_stride ? input[gmem_offset] : 0;
  }
  __syncthreads();

  // One repack tile covers 64 output rows by 16 logical K values.
  uint32_t fragment[4][1][4][2][2][2];
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t row = i * 8 + threadIdx.x / 4;
    const uint32_t *smem_row = smem[row];
    PRAGMA_UNROLL
    for (uint32_t j = 0; j < 8; ++j) {
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < 2; ++k) {
        const uint32_t index = j * 8 + (threadIdx.x % 4) * 2 + k;
        const uint32_t value = load_uint4<kPackedInput>(smem_row, index);
        const uint32_t i1 = j / 2;
        const uint32_t i3 = (i * 8) % 64 / 16;
        const uint32_t i4 = i % 2;
        const uint32_t i5 = j % 2;
        if constexpr (kTransposeMiniBlock) {
          fragment[i1][0][i3][i5][i4][k] = value;
        } else {
          fragment[i1][0][i3][i4][i5][k] = value;
        }
      }
    }
  }

  uint32_t packed[16];
  pack_w4a16_fragment(
      reinterpret_cast<const uint32_t *>(fragment), packed, interleave_mode);

  const uint32_t output_stride = padded_shape_n * 2;
  const uint32_t output_col_base = blockIdx.x * 128;
  const uint32_t output_rows = gridDim.z * padded_shape_k / 16;
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t row = blockIdx.y * 4 + blockIdx.z * padded_shape_k / 16 + i;
    if (row >= output_rows) continue;
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < 4; ++k) {
      const uint32_t col = output_col_base + threadIdx.x * 4 + k;
      output[row * output_stride + col] = packed[i * 4 + k];
    }
  }
}
