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

COMMON_TORCH_LIBRARY(chord, m) {
  m.def(
      "launch_kernel(int kernel_id, Tensor input, Tensor weight, "
      "Tensor? output, Tensor weight_scale, Tensor sorted_ids, "
      "Tensor expert_ids, Tensor num_tokens_padded, SymInt top_k) -> Tensor");
  m.def("register_kernel(str cubin_path, str func_name) -> int");
}

COMMON_TORCH_LIBRARY_IMPL(chord, CUDA, m) {
  m.impl("launch_kernel", COMMON_TORCH_BOX(&launch_kernel));
}

COMMON_TORCH_LIBRARY_IMPL(chord, Undefined, m) {
  m.impl("register_kernel", COMMON_TORCH_BOX(&register_kernel));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
