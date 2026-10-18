// SM90 grouped W4A16 GEMM on the generic C++ JIT; see grouped.h.  Generated
// source: grouped/kernel.py CODE_TEMPLATE and GroupedNVRTCCompiler
// (device_prelude, get_flags).  Registration and launch:
// register_grouped_w4a16_kernel / launch_grouped_w4a16 (csrc/launcher/launcher.cpp).
// Keep each in step with its Python / launcher counterpart while both paths exist.

#include "grouped.h"

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../jit/jit.h"

namespace chord::grouped {

namespace {

using jit::fail;

[[noreturn]] void invalid(const std::string& message) { throw std::invalid_argument(message); }

std::string str(int64_t v) { return std::to_string(v); }

const char* gemm_type_cpp(int64_t t) { return t == 2 ? "MGroupedMasked" : "MGroupedContiguous"; }

void check_config(const GemmConfig& c) {
  if (c.gemm_type != 1 && c.gemm_type != 2) invalid("grouped gemm_type must be 1 or 2");
  for (int64_t v : {c.shape_n, c.shape_k, c.num_groups, c.num_sms, c.block_m, c.block_n,
                    c.block_k, c.num_stages, c.smem_size, c.num_tma_threads, c.num_math_threads,
                    c.num_threads, c.cluster_size, c.num_ctas_per_sm}) {
    if (v <= 0) invalid("grouped config fields must be positive");
  }
  if (c.block_k != 64 && c.block_k != 128) invalid("grouped block-K must be 64 or 128");
  const int arch = jit::current_arch();
  if (arch != 90) fail("the grouped W4A16 backend requires SM90, got sm_" + str(arch));
}

// GroupedNVRTCCompiler.device_prelude (prepended to the rendered template).
// The whitespace-only line after the includes is part of the Python text
// (kept as kPreludeGap so editors that strip trailing spaces cannot alter it).
const char* kPreludeGap = "            ";
const char* kPreludeHead = R"(
            #include <cuda/std/type_traits>
            #include <cuda/std/utility>
            #include <cuda/std/cstdint>
            #include <cuda/std/__algorithm_>
)";
const char* kPreludeTail = R"(
    namespace std {
    using cuda::std::is_same;
    using cuda::std::conditional_t;
    using cuda::std::conditional;
    using cuda::std::enable_if;
    using cuda::std::enable_if_t;
    }

            namespace std { using cuda::std::min; }

            __device__ __forceinline__ void cudaGridDependencySynchronize() {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            __device__ __forceinline__ void cudaTriggerProgrammaticLaunchCompletion() {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            )";

// grouped/kernel.py kernel_expr, whitespace collapsed as NVRTCCompiler._compile does.
std::string name_expression(const GemmConfig& c) {
  return "deep_gemm::sm90_w4a16_gemm_impl< cute::SM90::GMMA::Major::K, 0, " + str(c.shape_n) +
         ", " + str(c.shape_k) + ", " + str(c.num_groups) + ", " + str(c.block_m) + ", " +
         str(c.block_n) + ", " + str(c.block_k) + ", " + str(c.swizzle_a) + ", " +
         str(c.swizzle_b) + ", " + str(c.swizzle_cd) + ", " + str(c.num_stages) + ", " +
         str(c.num_tma_threads) + ", " + str(c.num_math_threads) + ", " + str(c.cluster_size) +
         ", " + (c.multicast_on_a ? "true" : "false") + ", " + str(c.num_sms) +
         ", deep_gemm::GemmType::" + gemm_type_cpp(c.gemm_type) +
         ", deep_gemm::epilogue::transform::EpilogueIdentity, true, 32, " +
         str(c.num_ctas_per_sm) + ">";
}

// grouped/kernel.py CODE_TEMPLATE rendered (jinja2 drops the final newline).
std::string kernel_source(const GemmConfig& c) {
  const std::string multicast = c.multicast_on_a ? "true" : "false";
  std::string s = std::string(kPreludeHead) + kPreludeGap + kPreludeTail;
  s += "\n#include <deep_gemm/impls/sm90_w4a16_gemm.cuh>\n\n";
  s += "static void __instantiate_kernel() {\n";
  s += "    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_w4a16_gemm_impl<\n";
  s += "        cute::SM90::GMMA::Major::K,                     // kMajorSFB (dead slot under W4A16)\n";
  s += "        0, " + str(c.shape_n) + ", " + str(c.shape_k) +
       ",        // SHAPE_M is 0: runtime m\n";
  s += "        " + str(c.num_groups) + ",\n";
  s += "        " + str(c.block_m) + ", " + str(c.block_n) + ", " + str(c.block_k) + ",\n";
  s += "        " + str(c.swizzle_a) + ", " + str(c.swizzle_b) + ", " + str(c.swizzle_cd) + ",\n";
  s += "        " + str(c.num_stages) + ",\n";
  s += "        " + str(c.num_tma_threads) + ", " + str(c.num_math_threads) + ",\n";
  s += "        " + str(c.cluster_size) + ", " + multicast + ",\n";
  s += "        " + str(c.num_sms) + ", deep_gemm::GemmType::" + gemm_type_cpp(c.gemm_type) + ",\n";
  s += "        deep_gemm::epilogue::transform::EpilogueIdentity,\n";
  s += "        true, 32, " + str(c.num_ctas_per_sm) + "\n";
  s += "    >);\n";
  s += "}\n\n";
  auto constant = [&](const char* name, int64_t value) {
    s += std::string("extern \"C\" __constant__ uint32_t ") + name + " = " + str(value) + ";\n";
  };
  constant("SMEM_SIZE", c.smem_size);
  constant("NUM_THREADS", c.num_threads);
  constant("GRID_DIM", c.num_sms * c.num_ctas_per_sm);
  constant("CLUSTER_DIM", c.cluster_size);
  constant("GRP_BLOCK_M", c.block_m);
  constant("GRP_BLOCK_N", c.block_n);
  constant("GRP_BLOCK_K", c.block_k);
  constant("GRP_SWIZZLE_A", c.swizzle_a);
  constant("GRP_SWIZZLE_B", c.swizzle_b);
  constant("GRP_SWIZZLE_CD", c.swizzle_cd);
  constant("GRP_NUM_GROUPS", c.num_groups);
  constant("GRP_SCALE_GROUP", 32);
  constant("GRP_GEMM_TYPE", c.gemm_type);
  s.pop_back();
  return s;
}

// GroupedNVRTCCompiler.get_flags: the upstream DeepGEMM NVRTC set (C++20, no
// fast-math, int128) instead of the indexed kernel's.
std::vector<std::string> grouped_flags() {
  std::vector<std::string> flags;
  for (const auto& flag : jit::default_flags()) {
    if (flag == "--use_fast_math" || flag == "-extra-device-vectorization" || flag == "--dopt=on") {
      continue;
    }
    if (flag == "-std=c++17") {
      flags.push_back("-std=c++20");
    } else if (flag == "--diag-suppress=39,161,174,177,940") {
      flags.push_back("--diag-suppress=39,161,174,177,186,940");
    } else {
      flags.push_back(flag);
    }
  }
  flags.push_back("--device-int128");
  return flags;
}

jit::Program make_program(const GemmConfig& c) {
  check_config(c);
  jit::Program program;
  program.name_expression = name_expression(c);
  program.source = kernel_source(c);
  program.flags = grouped_flags();
  program.headers = jit::std_header_shims();
  program.cache_tag = "grouped_gemm";
  return program;
}

// --------------------------------------------------------------------------
// Registration (launcher.cpp: DGKernelData, register_grouped_w4a16_kernel)

struct Registered {
  int device;
  CUfunction func;
  uint32_t smem_size, num_threads, grid_dim, cluster_dim;
  uint32_t block_m, block_n, block_k;
  uint32_t swizzle_a, swizzle_b, swizzle_cd;
  uint32_t num_groups, gemm_type;
};

std::mutex g_mutex;                   // guards g_registered and g_ids
std::deque<Registered> g_registered;  // id -> kernel
std::map<std::tuple<int, std::string>, int64_t> g_ids;

Registered registered_kernel(int64_t id) {
  std::lock_guard<std::mutex> hold(g_mutex);
  if (id < 0 || id >= static_cast<int64_t>(g_registered.size())) {
    fail("grouped kernel is not registered: " + str(id));
  }
  return g_registered[id];
}

// launcher.cpp make_dg_tma_2d_desc, through the generic jit helper.
CUtensorMap tma_2d(void* ptr, CUtensorMapDataType dtype, int64_t elem_size, int64_t inner,
                   int64_t outer, int64_t box_inner, int64_t box_outer, int64_t stride_elems,
                   uint32_t swizzle) {
  return jit::make_tma_2d(ptr, dtype, elem_size, inner, outer, box_inner, box_outer, stride_elems,
                          swizzle);
}

}  // namespace

int64_t register_gemm(const GemmConfig& config) {
  const jit::Program program = make_program(config);
  const int device = jit::current_device();
  std::lock_guard<std::mutex> hold(g_mutex);
  const auto key = std::make_tuple(device, program.name_expression);
  if (auto it = g_ids.find(key); it != g_ids.end()) return it->second;

  const jit::Kernel& kernel = jit::get_kernel(program);
  Registered r;
  r.device = device;
  r.func = kernel.func;
  r.smem_size = kernel.constant("SMEM_SIZE");
  r.num_threads = kernel.constant("NUM_THREADS");
  r.grid_dim = kernel.constant("GRID_DIM");
  r.cluster_dim = kernel.constant("CLUSTER_DIM");
  r.block_m = kernel.constant("GRP_BLOCK_M");
  r.block_n = kernel.constant("GRP_BLOCK_N");
  r.block_k = kernel.constant("GRP_BLOCK_K");
  r.swizzle_a = kernel.constant("GRP_SWIZZLE_A");
  r.swizzle_b = kernel.constant("GRP_SWIZZLE_B");
  r.swizzle_cd = kernel.constant("GRP_SWIZZLE_CD");
  r.num_groups = kernel.constant("GRP_NUM_GROUPS");
  r.gemm_type = kernel.constant("GRP_GEMM_TYPE");
  if (r.num_threads == 0) fail("missing DG NUM_THREADS metadata");
  if (r.grid_dim == 0) fail("missing DG GRID_DIM metadata");
  if (kernel.constant("GRP_SCALE_GROUP") != 32) fail("vendored kernel requires scale group 32");
  jit::check(cuFuncSetAttribute(r.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                r.smem_size),
             "cuFuncSetAttribute (grouped w4a16 smem)");
  const int64_t id = static_cast<int64_t>(g_registered.size());
  g_registered.push_back(r);
  g_ids.emplace(key, id);
  return id;
}

int64_t gemm_type(int64_t id) { return registered_kernel(id).gemm_type; }

void gemm(int64_t id, const GemmArgs& a, CUstream stream) {
  const Registered kd = registered_kernel(id);
  const int device = jit::current_device();
  if (device != kd.device) {
    fail("grouped kernel " + str(id) + " was registered on device " + str(kd.device) +
         ", launched on device " + str(device));
  }
  if (static_cast<int64_t>(kd.num_groups) != a.num_groups) {
    fail("kernel was compiled for " + str(kd.num_groups) + " groups, got " + str(a.num_groups));
  }
  if (kd.block_k != 64 && kd.block_k != 128) fail("bad GRP_BLOCK_K");

  // launch_grouped_w4a16 (launcher.cpp): four TMA descriptors + launch.
  const bool a_has_groups = kd.gemm_type == 2;  // masked inputs are [G, max_m, K]
  const int64_t rows = a.m * (a_has_groups ? a.num_groups : 1);
  // A: (K, rows) BF16, K-major; box (swizzle A, BLOCK_M).
  CUtensorMap tm_a = tma_2d(a.a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.k, rows, kd.block_k,
                            kd.block_m, a.k, kd.swizzle_a);
  // B: (K/2, N * groups) packed UINT8, K-major; box (swizzle B, BLOCK_N).
  CUtensorMap tm_b = tma_2d(a.weight, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, a.k / 2,
                            a.n * a.num_groups, kd.block_k / 2, kd.block_n, a.k / 2, kd.swizzle_b);
  // D: (N, m [, groups]) BF16; box (swizzle CD, BLOCK_M).  Masked adds the
  // group as a third dimension so a partial last M tile's store clips at max_m.
  CUtensorMap tm_d =
      a_has_groups
          ? jit::make_tma_3d(a.d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.n, a.m, a.num_groups,
                             kd.block_n, kd.block_m, a.n, kd.swizzle_cd)
          : tma_2d(a.d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.n, a.m, kd.block_n, kd.block_m,
                   a.n, kd.swizzle_cd);
  // SFA: (N, (K/32) * groups), MN-major with N contiguous; box (BLOCK_N, 1).
  CUtensorMap tm_sfa = tma_2d(a.scale, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.n,
                              (a.k / 32) * a.num_groups, kd.block_n, 1, a.n, 0);

  void* sfb_ptr = nullptr;  // W4A16 scales ride the SFA slot; no global SFB.
  int* layout_ptr = a.layout;
  uint32_t m_u32 = static_cast<uint32_t>(a.m);
  uint32_t n_u32 = static_cast<uint32_t>(a.n);
  uint32_t k_u32 = static_cast<uint32_t>(a.k);
  void* kernel_args[] = {&sfb_ptr, &layout_ptr, &m_u32, &n_u32, &k_u32,
                         &tm_a,    &tm_b,       &tm_d,  &tm_sfa};

  CUlaunchAttribute attrs[2];
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
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim.x = kd.cluster_dim;
    attr.value.clusterDim.y = 1;
    attr.value.clusterDim.z = 1;
  }
  if (a.enable_pdl) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr.value.programmaticStreamSerializationAllowed = 1;
  }
  jit::check(cuLaunchKernelEx(&config, kd.func, kernel_args, nullptr),
             "cuLaunchKernelEx (grouped w4a16)");
}

}  // namespace chord::grouped
