// Derived from inclusionAI/humming; modified for chord_kernels.
// Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

#define USE_CUDA 1

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <unordered_map>

#include "./elf.h"
#include "./tensor.h"
#include "./torch_api.h"
#include "./utils.h"

static std::unordered_map<int64_t, KernelData> g_kernel_data;

inline KernelData &find_kernel_data(int64_t kernel_id) {
  auto it = g_kernel_data.find(kernel_id);
  ASSERT_CHECK(it != g_kernel_data.end(), "kernel is not registered: ", kernel_id);
  return it->second;
}

// One int32 lock word per active stream-K output tile.  The barrier protocol
// leaves every word it touches back at 0, so a zero-initialized buffer is
// reused across launches without a reset.  The scheduler peels at most
// ~1.1 * gridDim.x stream-K tiles, so 1024 words leave a wide margin at the
// tile counts these profiles launch; launch_kernel still checks the bound
// before every stream-K launch.
//
// The buffer is keyed by (device, stream): the reuse-without-reset contract
// holds only when launches sharing a buffer are ordered, which CUDA
// guarantees within one stream but not across streams.
static constexpr int64_t kNumLocks = 1024;
static std::unordered_map<int64_t, std::unordered_map<cudaStream_t, int32_t *>>
    g_locks;

inline cudaStream_t get_current_cuda_stream(int64_t device) {
#if USE_TORCH_STABLE_API
  void *stream_ptr = nullptr;
  aoti_torch_get_current_cuda_stream(device, &stream_ptr);
  return static_cast<cudaStream_t>(stream_ptr);
#else
  return at::cuda::getCurrentCUDAStream(device);
#endif
}

inline int32_t *get_locks(int64_t device, cudaStream_t stream) {
  auto &device_locks = g_locks[device];
  auto it = device_locks.find(stream);
  if (it != device_locks.end()) return it->second;
  int32_t *locks = nullptr;
  ASSERT_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&locks), kNumLocks * sizeof(int32_t)) ==
          cudaSuccess,
      "cudaMalloc for stream-K locks failed");
  // Zero on the launch stream so the first stream-K kernel on this stream is
  // ordered after the fill; a host-side memset would race a non-blocking
  // stream.
  ASSERT_CHECK(
      cudaMemsetAsync(locks, 0, kNumLocks * sizeof(int32_t), stream) ==
          cudaSuccess,
      "cudaMemsetAsync for stream-K locks failed");
  device_locks[stream] = locks;
  return locks;
}

inline uint32_t get_num_sms(int64_t device) {
  int32_t num_sms = 0;
  ASSERT_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device) ==
          cudaSuccess,
      "cudaDeviceGetAttribute failed");
  return static_cast<uint32_t>(num_sms);
}

Tensor launch_kernel(
    int64_t kernel_id,
    Tensor input,
    Tensor weight,
    std::optional<Tensor> output_,
    Tensor weight_scale,
    Tensor sorted_ids,
    Tensor expert_ids,
    Tensor num_tokens_padded,
    int64_t top_k) {
  KernelData &kernel_data = find_kernel_data(kernel_id);
  ASSERT_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  ASSERT_CHECK(input.dim() == 2, "input must be a 2D tensor");
  ASSERT_CHECK(top_k > 0, "top_k must be positive");

  const int64_t device = input.get_device();
  const int64_t shape_m = input.size(0);
  uint32_t shape_m_u32 = static_cast<uint32_t>(shape_m);
  uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  ASSERT_CHECK(
      static_cast<int64_t>(shape_m_u32) == shape_m,
      "shape_m overflows uint32_t: ", shape_m);
  ASSERT_CHECK(
      static_cast<int64_t>(top_k_u32) == top_k,
      "top_k overflows uint32_t: ", top_k);

  Tensor output =
      make_output_tensor(output_, input, kernel_data, top_k);
  check_indexed_w4a16_tensors(
      input, weight, output, weight_scale, sorted_ids, expert_ids,
      num_tokens_padded, kernel_data, top_k);

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();
  void *weight_scale_ptr = weight_scale.data_ptr();
  void *sorted_ids_ptr = sorted_ids.data_ptr();
  void *expert_ids_ptr = expert_ids.data_ptr();
  void *num_tokens_padded_ptr = num_tokens_padded.data_ptr();

  const cudaStream_t stream = get_current_cuda_stream(device);
  const uint32_t num_ctas = kernel_data.num_ctas_per_sm * get_num_sms(device);
  // Only stream-K touches the locks; data-parallel launches pass the buffer but
  // never read it (slice_count==1 skips the barrier).  The scheduler peels at
  // most mn_blocks % gridDim.x stream-K tiles, folding in one extra grid only
  // while the remainder is <= gridDim.x / 10, so the lock requirement is
  // bounded by gridDim.x + gridDim.x / 10.
  int32_t *locks_ptr = nullptr;
  if (kernel_data.use_stream_k) {
    const int64_t max_streamk_tiles =
        static_cast<int64_t>(num_ctas) + num_ctas / 10;
    ASSERT_CHECK(
        max_streamk_tiles <= kNumLocks,
        "stream-K launch may need ", max_streamk_tiles,
        " lock words but only ", kNumLocks, " are allocated");
    locks_ptr = get_locks(device, stream);
  }
  void *kernel_args[] = {
      &input_ptr,
      &weight_ptr,
      &output_ptr,
      &weight_scale_ptr,
      &sorted_ids_ptr,
      &expert_ids_ptr,
      &num_tokens_padded_ptr,
      &locks_ptr,
      &shape_m_u32,
      &top_k_u32,
  };

  CUlaunchConfig config = {};
  config.gridDimX = num_ctas;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = kernel_data.num_threads;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = kernel_data.smem_size;
  config.hStream = stream;

  constexpr auto kSmemSizeAttribute =
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
  check_curesult(
      cuFuncSetAttribute(
          kernel_data.func, kSmemSizeAttribute, kernel_data.smem_size),
      "cuFuncSetAttribute");
  check_curesult(
      cuLaunchKernelEx(&config, kernel_data.func, kernel_args, nullptr),
      "cuLaunchKernelEx");
  return output;
}

int64_t register_kernel(
    const std::string &cubin_path,
    const std::string &function_name) {
  int device = 0;
  ASSERT_CHECK(
      cudaGetDevice(&device) == cudaSuccess,
      "cudaGetDevice failed while registering kernel");

  int64_t kernel_id = manual_crc32(cubin_path);
  const std::string device_function =
      function_name + "\n" + std::to_string(device);
  kernel_id = (kernel_id << 30) + manual_crc32(device_function);

  if (g_kernel_data.find(kernel_id) == g_kernel_data.end()) {
    CUmodule module;
    CUfunction function;
    check_curesult(cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
    check_curesult(
        cuModuleGetFunction(&function, module, function_name.c_str()),
        "cuModuleGetFunction");
    CubinReader reader(cubin_path);
    const uint32_t num_experts = reader.getUint32("NUM_EXPERTS");
    const uint32_t num_ctas_per_sm = reader.getUint32("NUM_CTAS_PER_SM");
    ASSERT_CHECK(num_experts > 0, "indexed kernel requires num_experts > 0");
    ASSERT_CHECK(
        num_ctas_per_sm > 0,
        "indexed kernel requires num_ctas_per_sm > 0");

    g_kernel_data[kernel_id] = {
        module,
        function,
        reader.getUint32("SMEM_SIZE"),
        reader.getUint32("NUM_THREADS"),
        reader.getUint32("PROBLEM_SHAPE_N"),
        reader.getUint32("PROBLEM_SHAPE_K"),
        num_experts,
        num_ctas_per_sm,
        reader.getUint32("USE_STREAM_K") != 0,
    };
  }
  return kernel_id;
}

// ==================== grouped W4A16 (TMA) launch path ====================
// Derived from deepseek-ai/DeepGEMM (secondarily developed from the public
// baseline recorded in chord_kernels/operator/SOURCE.md):
// csrc/jit_kernels/impls/runtime_utils.hpp (TMA descriptor builders) and
// csrc/jit/handle.hpp (cluster/PDL launch plumbing), specialized to the one
// vendored kernel family (SM90 W4A16, K-major A/B, BF16 D, MN-major BF16
// scales on the SFA slot, scale group 32).
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12010
#error "the grouped W4A16 launch path requires CUDA >= 12.1 driver headers (CUtensorMap)"
#endif

namespace {

struct DGKernelData {
  CUmodule module;
  CUfunction func;
  uint32_t smem_size;
  uint32_t num_threads;
  uint32_t grid_dim;
  uint32_t cluster_dim;
  uint32_t block_m;
  uint32_t block_n;
  uint32_t block_k;
  uint32_t swizzle_a;
  uint32_t swizzle_b;
  uint32_t swizzle_cd;
  uint32_t num_groups;
  uint32_t scale_group;
  uint32_t gemm_type;  // 1 = m-grouped contiguous (prefill), 2 = masked (decode)
};

std::unordered_map<int64_t, DGKernelData> g_grouped_kernel_data;

// Maps the byte-mode swizzle to the TMA enum; a non-zero mode overrides the
// inner smem box extent with swizzle_bytes / elem_size (upstream behavior).
CUtensorMap make_dg_tma_2d_desc(
    void *ptr,
    CUtensorMapDataType dtype,
    int64_t elem_size,
    int64_t gmem_inner_dim,
    int64_t gmem_outer_dim,
    int64_t smem_inner_dim,
    int64_t smem_outer_dim,
    int64_t gmem_outer_stride_elems,
    uint32_t swizzle_mode) {
  CUtensorMapSwizzle swizzle;
  switch (swizzle_mode) {
    case 0:
    case 16:
      swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
      break;
    case 32:
      swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
      break;
    case 64:
      swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
      break;
    case 128:
      swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
      break;
    default:
      ASSERT_CHECK(false, "unsupported TMA swizzle mode: ", swizzle_mode);
  }
  if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;

  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {
      static_cast<cuuint64_t>(gmem_inner_dim),
      static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {
      static_cast<cuuint32_t>(smem_inner_dim),
      static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride_elems * elem_size)};
  const cuuint32_t elem_strides[2] = {1, 1};
  check_curesult(
      cuTensorMapEncodeTiled(
          &tensor_map, dtype, 2, ptr, gmem_dims, gmem_strides, smem_dims,
          elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tensor_map;
}

void check_dg_common(
    const Tensor &a,
    const Tensor &packed_w,
    const Tensor &scale,
    int64_t num_groups,
    int64_t shape_n,
    int64_t shape_k,
    int64_t device) {
  ASSERT_CHECK(a.is_cuda() && a.is_contiguous(), "a must be contiguous CUDA");
  ASSERT_CHECK(a.scalar_type() == ScalarType::BFloat16, "a must be BFloat16");
  ASSERT_CHECK(a.size(-1) == shape_k, "a K mismatch");
  ASSERT_CHECK(
      packed_w.is_cuda() && packed_w.is_contiguous(),
      "packed weight must be contiguous CUDA");
  ASSERT_CHECK(packed_w.dim() == 3, "packed weight must be [G, N, K/2]");
  ASSERT_CHECK(
      packed_w.scalar_type() == ScalarType::Float8_e4m3fn,
      "packed weight must be Float8_e4m3fn, got ",
      DTYPE_TO_STRING(packed_w.scalar_type()));
  ASSERT_CHECK(packed_w.size(0) == num_groups, "weight G mismatch");
  ASSERT_CHECK(packed_w.size(1) == shape_n, "weight N mismatch");
  ASSERT_CHECK(packed_w.size(2) == shape_k / 2, "weight K/2 mismatch");
  ASSERT_CHECK(
      scale.is_cuda() && scale.is_contiguous(), "scale must be contiguous CUDA");
  ASSERT_CHECK(scale.scalar_type() == ScalarType::BFloat16, "scale must be BFloat16");
  ASSERT_CHECK(
      scale.dim() == 3 && scale.size(0) == num_groups &&
          scale.size(1) == shape_k / 32 && scale.size(2) == shape_n,
      "scale must be MN-major BF16 [G, K/32, N]");
  ASSERT_CHECK(
      packed_w.get_device() == device && scale.get_device() == device,
      "weights must live on the activation device");
  ASSERT_CHECK(shape_n % 16 == 0, "N must be a multiple of 16 for the SF TMA");
}

// Shared tail of both grouped entry points: four descriptors + launch.
void launch_grouped_w4a16(
    const DGKernelData &kd,
    void *a_ptr,
    void *w_ptr,
    void *scale_ptr,
    void *d_ptr,
    int *grouped_layout_ptr,
    int64_t num_groups,
    int64_t m_total,  // masked: per-group max_m; contiguous: concatenated rows
    int64_t n,
    int64_t k,
    bool a_has_groups,
    cudaStream_t stream,
    bool enable_pdl) {
  ASSERT_CHECK(kd.block_k == 64 || kd.block_k == 128, "bad GRP_BLOCK_K");

  // A: (K, m_total [* groups]) BF16 rows, K-major; box (swizzle A, BLOCK_M).
  CUtensorMap tm_a = make_dg_tma_2d_desc(
      a_ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, k,
      m_total * (a_has_groups ? num_groups : 1), kd.block_k, kd.block_m,
      k /* row stride in elements */, kd.swizzle_a);
  // B: (K/2, N * groups) packed UINT8 rows, K-major; box (swizzle B, BLOCK_N).
  CUtensorMap tm_b = make_dg_tma_2d_desc(
      w_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, k / 2, n * num_groups,
      kd.block_k / 2, kd.block_n, k / 2, kd.swizzle_b);
  // D: (N, m_total [* groups]) BF16 rows; box (swizzle CD, BLOCK_M).
  CUtensorMap tm_d = make_dg_tma_2d_desc(
      d_ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, n,
      m_total * (a_has_groups ? num_groups : 1), kd.block_n, kd.block_m, n,
      kd.swizzle_cd);
  // SFA: (N, (K/32) * groups), MN-major with N contiguous; box (BLOCK_N, 1).
  CUtensorMap tm_sfa = make_dg_tma_2d_desc(
      scale_ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, n,
      (k / 32) * num_groups, kd.block_n, 1, n, 0);

  void *sfb_ptr = nullptr;  // W4A16 scales ride the SFA slot; no global SFB.
  uint32_t m_u32 = static_cast<uint32_t>(m_total);
  uint32_t n_u32 = static_cast<uint32_t>(n);
  uint32_t k_u32 = static_cast<uint32_t>(k);
  void *kernel_args[] = {
      &sfb_ptr, &grouped_layout_ptr, &m_u32, &n_u32, &k_u32,
      &tm_a, &tm_b, &tm_d, &tm_sfa};

  // Launch attributes must outlive the launch call; keep them static like
  // upstream does.
  static CUlaunchAttribute attrs[2];
  CUlaunchConfig config = {};
  config.gridDimX = kd.grid_dim;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = kd.num_threads;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = kd.smem_size;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = 0;
  if (kd.cluster_dim > 1) {
    auto &attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim.x = kd.cluster_dim;
    attr.value.clusterDim.y = 1;
    attr.value.clusterDim.z = 1;
  }
  if (enable_pdl) {
    auto &attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr.value.programmaticStreamSerializationAllowed = 1;
  }
  check_curesult(
      cuLaunchKernelEx(&config, kd.func, kernel_args, nullptr),
      "cuLaunchKernelEx (grouped w4a16)");
}

}  // namespace

Tensor launch_grouped_w4a16_masked(
    int64_t kernel_id,
    Tensor a,
    Tensor packed_w,
    Tensor scale,
    Tensor masked_m,
    std::optional<Tensor> out_,
    bool enable_pdl) {
  auto it = g_grouped_kernel_data.find(kernel_id);
  ASSERT_CHECK(
      it != g_grouped_kernel_data.end(), "grouped kernel is not registered: ", kernel_id);
  const DGKernelData &kd = it->second;
  ASSERT_CHECK(kd.gemm_type == 2, "kernel was compiled for the masked path");
  ASSERT_CHECK(a.dim() == 3, "masked a must be [G, max_m, K]");
  const int64_t device = a.get_device();
  const int64_t num_groups = a.size(0);
  const int64_t shape_m = a.size(1);
  const int64_t shape_k = a.size(2);
  const int64_t shape_n = packed_w.size(1);
  check_dg_common(a, packed_w, scale, num_groups, shape_n, shape_k, device);
  ASSERT_CHECK(
      masked_m.is_cuda() && masked_m.is_contiguous() &&
          masked_m.scalar_type() == ScalarType::Int &&
          masked_m.numel() == num_groups,
      "masked_m must be contiguous CUDA int32 [G]");
  ASSERT_CHECK(
      static_cast<int64_t>(kd.num_groups) == num_groups,
      "kernel was compiled for ", kd.num_groups, " groups, got ", num_groups);
  ASSERT_CHECK(shape_k % 32 == 0, "K must be a multiple of the scale group");
  Tensor d = out_.has_value()
      ? out_.value()
      : torch_empty(
            {num_groups, shape_m, shape_n}, ScalarType::BFloat16, a.device());
  ASSERT_CHECK(
      d.is_contiguous() && d.dim() == 3 && d.size(0) == num_groups &&
          d.size(1) == shape_m && d.size(2) == shape_n &&
          d.scalar_type() == ScalarType::BFloat16,
      "out must be BFloat16 [G, max_m, N]");
  launch_grouped_w4a16(
      kd, a.data_ptr(), packed_w.data_ptr(), scale.data_ptr(), d.data_ptr(),
      static_cast<int *>(masked_m.data_ptr()), num_groups, shape_m, shape_n,
      shape_k, /*a_has_groups=*/true, get_current_cuda_stream(device),
      enable_pdl);
  return d;
}

Tensor launch_grouped_w4a16_contiguous(
    int64_t kernel_id,
    Tensor a,
    Tensor packed_w,
    Tensor scale,
    Tensor m_indices,
    std::optional<Tensor> out_,
    bool enable_pdl) {
  auto it = g_grouped_kernel_data.find(kernel_id);
  ASSERT_CHECK(
      it != g_grouped_kernel_data.end(), "grouped kernel is not registered: ", kernel_id);
  const DGKernelData &kd = it->second;
  ASSERT_CHECK(kd.gemm_type == 1, "kernel was compiled for the contiguous path");
  ASSERT_CHECK(a.dim() == 2, "contiguous a must be [m, K]");
  const int64_t device = a.get_device();
  const int64_t shape_m = a.size(0);
  const int64_t shape_k = a.size(1);
  const int64_t num_groups = packed_w.size(0);
  const int64_t shape_n = packed_w.size(1);
  check_dg_common(a, packed_w, scale, num_groups, shape_n, shape_k, device);
  ASSERT_CHECK(
      m_indices.is_cuda() && m_indices.is_contiguous() &&
          m_indices.scalar_type() == ScalarType::Int &&
          m_indices.numel() == shape_m,
      "m_indices must be contiguous CUDA int32 [m]");
  ASSERT_CHECK(
      static_cast<int64_t>(kd.num_groups) == num_groups,
      "kernel was compiled for ", kd.num_groups, " groups, got ", num_groups);
  ASSERT_CHECK(
      shape_m % 128 == 0,
      "contiguous m must be 128-aligned (per-expert padded), got ", shape_m);
  Tensor d = out_.has_value()
      ? out_.value()
      : torch_empty({shape_m, shape_n}, ScalarType::BFloat16, a.device());
  ASSERT_CHECK(
      d.is_contiguous() && d.dim() == 2 && d.size(0) == shape_m &&
          d.size(1) == shape_n && d.scalar_type() == ScalarType::BFloat16,
      "out must be BFloat16 [m, N]");
  launch_grouped_w4a16(
      kd, a.data_ptr(), packed_w.data_ptr(), scale.data_ptr(), d.data_ptr(),
      static_cast<int *>(m_indices.data_ptr()), num_groups, shape_m, shape_n,
      shape_k, /*a_has_groups=*/false, get_current_cuda_stream(device),
      enable_pdl);
  return d;
}

int64_t register_grouped_w4a16_kernel(
    const std::string &cubin_path,
    const std::string &function_name) {
  int device = 0;
  ASSERT_CHECK(
      cudaGetDevice(&device) == cudaSuccess,
      "cudaGetDevice failed while registering kernel");

  int64_t kernel_id = manual_crc32("grouped_w4a16\n" + cubin_path);
  const std::string device_function =
      function_name + "\n" + std::to_string(device);
  kernel_id = (kernel_id << 30) + manual_crc32(device_function);

  if (g_grouped_kernel_data.find(kernel_id) == g_grouped_kernel_data.end()) {
    CUmodule module;
    CUfunction function;
    check_curesult(cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
    check_curesult(
        cuModuleGetFunction(&function, module, function_name.c_str()),
        "cuModuleGetFunction (grouped w4a16)");
    CubinReader reader(cubin_path);
    DGKernelData data = {
        module,
        function,
        reader.getUint32("SMEM_SIZE"),
        reader.getUint32("NUM_THREADS"),
        reader.getUint32("GRID_DIM"),
        reader.getUint32("CLUSTER_DIM"),
        reader.getUint32("GRP_BLOCK_M"),
        reader.getUint32("GRP_BLOCK_N"),
        reader.getUint32("GRP_BLOCK_K"),
        reader.getUint32("GRP_SWIZZLE_A"),
        reader.getUint32("GRP_SWIZZLE_B"),
        reader.getUint32("GRP_SWIZZLE_CD"),
        reader.getUint32("GRP_NUM_GROUPS"),
        reader.getUint32("GRP_SCALE_GROUP"),
        reader.getUint32("GRP_GEMM_TYPE"),
    };
    ASSERT_CHECK(data.num_threads > 0, "missing DG NUM_THREADS metadata");
    ASSERT_CHECK(data.grid_dim > 0, "missing DG GRID_DIM metadata");
    ASSERT_CHECK(data.scale_group == 32, "vendored kernel requires scale group 32");
    ASSERT_CHECK(
        data.gemm_type == 1 || data.gemm_type == 2,
        "unsupported DG gemm type: ", data.gemm_type);
    check_curesult(
        cuFuncSetAttribute(
            function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            data.smem_size),
        "cuFuncSetAttribute (grouped w4a16 smem)");
    g_grouped_kernel_data[kernel_id] = data;
  }
  return kernel_id;
}

COMMON_TORCH_LIBRARY(chord, m) {
  m.def(
      "launch_kernel(int kernel_id, Tensor input, Tensor weight, "
      "Tensor? output, Tensor weight_scale, Tensor sorted_ids, "
      "Tensor expert_ids, Tensor num_tokens_padded, SymInt top_k) -> Tensor");
  m.def("register_kernel(str cubin_path, str func_name) -> int");
  m.def(
      "launch_grouped_w4a16_masked(int kernel_id, Tensor a, Tensor packed_weight, "
      "Tensor scale, Tensor masked_m, Tensor? output, bool enable_pdl) -> Tensor");
  m.def(
      "launch_grouped_w4a16_contiguous(int kernel_id, Tensor a, Tensor packed_weight, "
      "Tensor scale, Tensor m_indices, Tensor? output, bool enable_pdl) -> Tensor");
  m.def("register_grouped_w4a16_kernel(str cubin_path, str func_name) -> int");
}

COMMON_TORCH_LIBRARY_IMPL(chord, CUDA, m) {
  m.impl("launch_kernel", COMMON_TORCH_BOX(&launch_kernel));
  m.impl("launch_grouped_w4a16_masked", COMMON_TORCH_BOX(&launch_grouped_w4a16_masked));
  m.impl("launch_grouped_w4a16_contiguous", COMMON_TORCH_BOX(&launch_grouped_w4a16_contiguous));
}

COMMON_TORCH_LIBRARY_IMPL(chord, Undefined, m) {
  m.impl("register_kernel", COMMON_TORCH_BOX(&register_kernel));
  m.impl("register_grouped_w4a16_kernel", COMMON_TORCH_BOX(&register_grouped_w4a16_kernel));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
