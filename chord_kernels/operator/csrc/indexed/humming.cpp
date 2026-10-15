// Indexed W4A16 GEMM on the generic C++ JIT; see humming.h.  Checks: api.py
// _get_indexed_kernel and IndexedKernelConfig, kernel/humming.py
// HummingKernel._validate_config / _select_mma_op.  Generated source:
// kernel/humming.py CODE_TEMPLATE and config/mma.py _MmaOp.to_cpp_str.
// Registration and launch: register_kernel / launch_kernel
// (csrc/launcher/launcher.cpp).  Keep each in step with its Python / launcher
// counterpart while both paths exist.

#include "humming.h"

#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../jit/jit.h"

namespace chord::indexed {

namespace {

using jit::fail;

[[noreturn]] void invalid(const std::string& message) { throw std::invalid_argument(message); }

std::string str(int64_t v) { return std::to_string(v); }

bool is_power_of_two(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

// --------------------------------------------------------------------------
// Checks, in the order the Python path raises them

struct MmaShape {
  int64_t m, n, k;
};

int64_t num_threads(const GemmConfig& c) {
  return c.block_m * c.block_n * c.block_k / (c.warp_m * c.warp_n * c.warp_k) * 32;
}

// MMA layout only: the C++ path never sees a WGMMA-packed weight.
MmaShape check_config(const GemmConfig& c) {
  const std::initializer_list<std::pair<const char*, int64_t>> fields = {
      {"shape_n", c.shape_n}, {"shape_k", c.shape_k}, {"num_experts", c.num_experts},
      {"block_m", c.block_m}, {"block_n", c.block_n}, {"block_k", c.block_k},
      {"warp_m", c.warp_m},   {"warp_n", c.warp_n},   {"warp_k", c.warp_k},
      {"num_stages", c.num_stages}, {"num_ctas_per_sm", c.num_ctas_per_sm}};
  for (const auto& [name, v] : fields) {
    if (v <= 0) invalid(std::string(name) + " must be a positive integer, got " + str(v));
  }
  // IndexedKernelConfig.__post_init__
  if (c.block_m % 8) invalid("indexed block-M must be a multiple of 8");
  if (c.block_m % c.warp_m || c.block_n % c.warp_n || c.block_k % c.warp_k) {
    invalid("indexed block shape must be divisible by warp shape");
  }
  if (c.warp_n != 32 && c.warp_n != 64) invalid("indexed warp-N must be 32 (WGMMA) or 64 (MMA)");
  if (c.num_stages < 2) invalid("num_stages must be an integer of at least 2");
  if (c.use_stream_k && c.num_ctas_per_sm > 4) {
    invalid("use_stream_k supports at most 4 CTAs/SM (stream-K lock capacity), got "
            "num_ctas_per_sm=" + str(c.num_ctas_per_sm));
  }
  // api.py _get_indexed_kernel
  if (!c.swap_ab && c.block_m % 16 == 8) {
    invalid(
        "MMA indexed configs with block_m=8 (or another 16q+8 tile) must set swap_ab=True");
  }
  if (c.block_k % 64) {
    invalid("indexed block-K must be a positive multiple of 64, got " + str(c.block_k));
  }
  // HummingKernel._validate_config
  if (c.shape_n % c.block_n || c.shape_k % c.block_k) {
    invalid("problem N/K must be divisible by the block N/K tile");
  }
  if (c.warp_n != 64) invalid("WGMMA requires warp-N 32 and MMA requires warp-N 64");
  if (c.warp_k < 32) invalid("warp-K must be at least 32");
  for (int64_t v : {c.block_n, c.block_k, c.warp_n, c.warp_k, c.block_m / c.warp_m,
                    c.block_n / c.warp_n, c.block_k / c.warp_k}) {
    if (!is_power_of_two(v)) invalid("indexed kernel tile ratios must be powers of two");
  }
  // HummingKernel.init_kernel / _select_mma_op
  const int arch = jit::current_arch();
  if (arch != 100 && arch != 103) {
    fail("the C++ JIT indexed W4A16 path requires SM100 or SM103, got sm_" + str(arch));
  }
  MmaShape mma = {16, 8, 16};
  if (!c.swap_ab && c.warp_m % 16 == 8) mma = {8, 8, 8};
  if (c.swap_ab) {
    if (c.warp_n % mma.m || c.warp_m % mma.n) invalid("warp shape is incompatible with swap-AB MMA");
  } else if (c.warp_m % mma.m || c.warp_n % mma.n) {
    invalid("warp shape is incompatible with the MMA tile");
  }
  if (c.warp_k % mma.k) invalid("warp-K is incompatible with the MMA instruction");
  return mma;
}

// --------------------------------------------------------------------------
// Kernel source

int64_t register_count(int64_t rows, int64_t cols, int64_t bits) {
  const int64_t total = rows * cols * bits;
  if (total % (32 * 32)) {
    invalid("MMA fragment " + str(rows) + "x" + str(cols) + "x" + str(bits) +
            " does not fill whole registers");
  }
  return total / (32 * 32);
}

std::string join(int64_t count, const std::string& prefix, const std::string& suffix) {
  std::string out;
  for (int64_t i = 0; i < count; ++i) {
    if (i) out += ", ";
    out += prefix + str(i) + suffix;
  }
  return out;
}

// config/mma.py _MmaOp.to_cpp_str: every line indented by 2, the asm block by 2 more.
std::string mma_op_class(const MmaShape& mma) {
  const int64_t ra = register_count(mma.m, mma.k, 16);
  const int64_t rb = register_count(mma.k, mma.n, 16);
  const int64_t rc = register_count(mma.m, mma.n, 32);
  // {%0, ..}, {..}, {..}, {..}: d, a, b, c operands numbered in order.
  std::string placeholders;
  int64_t start = 0;
  for (int64_t count : {rc, ra, rb, rc}) {
    if (start) placeholders += ", ";
    placeholders += "{";
    for (int64_t i = 0; i < count; ++i) {
      if (i) placeholders += ", ";
      placeholders += "%" + str(start + i);
    }
    placeholders += "}";
    start += count;
  }
  const std::string a_params = join(ra, " \"r\"(a[", "])");
  const std::string b_params = join(rb, " \"r\"(b[", "])");
  const std::string c_params = join(rc, " \"f\"(c[", "])");
  const std::string d_params = join(rc, "\"+f\"(d[", "])");
  const std::string asm_op = "mma.sync.aligned.m" + str(mma.m) + "n" + str(mma.n) + "k" +
                             str(mma.k) + ".row.col.f32.bf16.bf16.f32";
  const std::vector<std::string> lines = {
      "static constexpr bool kUseWgmma = false;",
      "using MmaShape = Shape<" + str(mma.m) + ", " + str(mma.n) + ", " + str(mma.k) + ">;",
      "using ARegisters = uint32_t[" + str(ra) + "];",
      "using BRegisters = uint32_t[" + str(rb) + "];",
      "using CRegisters = float[" + str(rc) + "];",
      "",
      "CUDA_INLINE",
      "static void fma(uint32_t *a, uint32_t *b, float *c, float *d) {",
      "  asm volatile(",
      "    \"" + asm_op + " \"",
      "    \"" + placeholders + ";\\n\"",
      "    : " + d_params,
      "    : " + a_params + ",",
      "      " + b_params + ",",
      "      " + c_params,
      "  );",
      "};",
  };
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (i) out += "\n";
    if (!lines[i].empty()) out += "  " + lines[i];
  }
  return out;
}

const char* bool_str(bool v) { return v ? "true" : "false"; }

// kernel/humming.py kernel_expr, whitespace collapsed as NVRTCCompiler._compile does.
std::string name_expression(const GemmConfig& c) {
  return "humming< MmaOpClass, Shape<0, " + str(c.shape_n) + ", " + str(c.shape_k) + ">, Shape<" +
         str(c.block_m) + ", " + str(c.block_n) + ", " + str(c.block_k) + ">, Shape<" +
         str(c.warp_m) + ", " + str(c.warp_n) + ", " + str(c.warp_k) + ">, TuningConfig>";
}

// kernel/humming.py CODE_TEMPLATE rendered (jinja2 drops the final newline).
std::string kernel_source(const GemmConfig& c, const MmaShape& mma) {
  const std::string threads = str(num_threads(c));
  std::string s;
  s += "\n#include <humming/kernel/humming.cuh>\n\n";
  s += "class MmaOpClass {\npublic:\n" + mma_op_class(mma) + "\n};\n\n";
  s += "class TuningConfig {\npublic:\n";
  s += "  static constexpr uint32_t kNumStages = " + str(c.num_stages) + ";\n";
  s += "  static constexpr uint32_t kNumCtasPerSm = " + str(c.num_ctas_per_sm) + ";\n";
  s += "  static constexpr uint32_t kNumThreads = " + threads + ";\n";
  s += std::string("  static constexpr bool kSwapAb = ") + bool_str(c.swap_ab) + ";\n";
  s += std::string("  static constexpr bool kUseStreamK = ") + bool_str(c.use_stream_k) + ";\n";
  s += "};\n\n";
  s += "using SharedStorageType = SharedStorage<\n";
  s += "    Shape<" + str(c.block_m) + ", " + str(c.block_n) + ", " + str(c.block_k) + ">,\n";
  s += "    Shape<" + str(c.warp_m) + ", " + str(c.warp_n) + ", " + str(c.warp_k) + ">,\n";
  s += "    TuningConfig>;\n\n";
  auto constant = [&](const char* name, const std::string& value) {
    s += std::string("extern \"C\" __constant__ uint32_t ") + name + " = " + value + ";\n";
  };
  constant("SMEM_SIZE", "sizeof(SharedStorageType)");
  constant("PROBLEM_SHAPE_N", str(c.shape_n));
  constant("PROBLEM_SHAPE_K", str(c.shape_k));
  constant("NUM_EXPERTS", str(c.num_experts));
  constant("NUM_THREADS", threads);
  constant("NUM_CTAS_PER_SM", str(c.num_ctas_per_sm));
  constant("USE_STREAM_K", c.use_stream_k ? "1" : "0");
  s.pop_back();
  return s;
}

jit::Program make_program(const GemmConfig& c) {
  const MmaShape mma = check_config(c);
  jit::Program program;
  program.name_expression = name_expression(c);
  program.source = kernel_source(c, mma);
  program.flags = jit::default_flags();
  program.headers = jit::std_header_shims();
  program.cache_tag = "indexed_gemm";
  return program;
}

// --------------------------------------------------------------------------
// Registration (launcher.cpp: KernelData, register_kernel)

struct Registered {
  GemmConfig config;
  int device;
  CUfunction func;
  uint32_t smem_size;
  uint32_t num_threads;
  uint32_t num_ctas_per_sm;
  bool use_stream_k;
};

using ConfigKey = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int64_t, int64_t, int64_t, bool, bool>;

ConfigKey config_key(int device, const GemmConfig& c) {
  return {device,      c.shape_n,  c.shape_k,    c.num_experts,     c.block_m,
          c.block_n,   c.block_k,  c.warp_m,     c.warp_n,          c.warp_k,
          c.num_stages, c.num_ctas_per_sm, c.swap_ab, c.use_stream_k};
}

std::mutex g_mutex;                  // guards g_registered and g_ids
std::deque<Registered> g_registered;  // id -> kernel; deque keeps references stable
std::map<ConfigKey, int64_t> g_ids;

int64_t register_locked(const GemmConfig& c) {
  const int device = jit::current_device();
  const ConfigKey key = config_key(device, c);
  if (auto it = g_ids.find(key); it != g_ids.end()) return it->second;

  const jit::Kernel& kernel = jit::get_kernel(make_program(c));
  Registered r;
  r.config = c;
  r.device = device;
  r.func = kernel.func;
  r.smem_size = kernel.constant("SMEM_SIZE");
  r.num_threads = kernel.constant("NUM_THREADS");
  r.num_ctas_per_sm = kernel.constant("NUM_CTAS_PER_SM");
  r.use_stream_k = kernel.constant("USE_STREAM_K") != 0;
  if (kernel.constant("NUM_EXPERTS") == 0) fail("indexed kernel requires num_experts > 0");
  if (r.num_ctas_per_sm == 0) fail("indexed kernel requires num_ctas_per_sm > 0");
  if (kernel.constant("PROBLEM_SHAPE_N") != uint32_t(c.shape_n) ||
      kernel.constant("PROBLEM_SHAPE_K") != uint32_t(c.shape_k) ||
      kernel.constant("NUM_EXPERTS") != uint32_t(c.num_experts)) {
    fail("indexed kernel metadata disagrees with its config");
  }
  // launch_kernel sets this before every launch; once per function is enough.
  jit::check(cuFuncSetAttribute(r.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                r.smem_size),
             "cuFuncSetAttribute");
  const int64_t id = static_cast<int64_t>(g_registered.size());
  g_registered.push_back(r);
  g_ids.emplace(key, id);
  return id;
}

// --------------------------------------------------------------------------
// Stream-K locks (launcher.cpp: kNumLocks, get_locks)

constexpr int64_t kNumLocks = 1024;

CUdeviceptr locks_for(int device, CUstream stream) {
  static std::mutex mutex;
  static std::unordered_map<int, std::unordered_map<CUstream, CUdeviceptr>> locks;
  std::lock_guard<std::mutex> hold(mutex);
  auto& device_locks = locks[device];
  if (auto it = device_locks.find(stream); it != device_locks.end()) return it->second;
  CUdeviceptr ptr = 0;
  jit::check(cuMemAlloc(&ptr, kNumLocks * sizeof(int32_t)), "cuMemAlloc for stream-K locks");
  // Zero on the launch stream so the first stream-K kernel on it is ordered after the fill.
  jit::check(cuMemsetD32Async(ptr, 0, kNumLocks, stream), "cuMemsetD32Async for stream-K locks");
  device_locks[stream] = ptr;
  return ptr;
}

}  // namespace

int64_t register_gemm(const GemmConfig& config) {
  std::lock_guard<std::mutex> hold(g_mutex);
  return register_locked(config);
}

std::vector<int64_t> precompile_gemm(const std::vector<GemmConfig>& configs) {
  std::lock_guard<std::mutex> hold(g_mutex);
  std::vector<jit::Program> programs;
  for (const auto& c : configs) programs.push_back(make_program(c));
  jit::precompile(programs);
  std::vector<int64_t> ids;
  for (const auto& c : configs) ids.push_back(register_locked(c));
  return ids;
}

const GemmConfig& gemm_config(int64_t id) {
  std::lock_guard<std::mutex> hold(g_mutex);
  if (id < 0 || id >= static_cast<int64_t>(g_registered.size())) {
    fail("indexed kernel is not registered: " + str(id));
  }
  return g_registered[id].config;
}

void gemm(int64_t id, const GemmArgs& a, CUstream stream) {
  Registered kd;
  {
    std::lock_guard<std::mutex> hold(g_mutex);
    if (id < 0 || id >= static_cast<int64_t>(g_registered.size())) {
      fail("indexed kernel is not registered: " + str(id));
    }
    kd = g_registered[id];
  }
  const int device = jit::current_device();
  if (device != kd.device) {
    fail("indexed kernel " + str(id) + " was registered on device " + str(kd.device) +
         ", launched on device " + str(device));
  }
  if (a.top_k <= 0) fail("top_k must be positive");
  uint32_t shape_m = static_cast<uint32_t>(a.shape_m);
  uint32_t top_k = static_cast<uint32_t>(a.top_k);
  if (static_cast<int64_t>(shape_m) != a.shape_m) fail("shape_m overflows uint32_t: " + str(a.shape_m));
  if (static_cast<int64_t>(top_k) != a.top_k) fail("top_k overflows uint32_t: " + str(a.top_k));

  const uint32_t num_ctas = kd.num_ctas_per_sm * static_cast<uint32_t>(jit::num_sms());
  // Only stream-K touches the locks; data-parallel launches pass a null pointer.  The
  // scheduler needs at most gridDim.x + gridDim.x / 10 of them (launcher.cpp).
  CUdeviceptr locks = 0;
  if (kd.use_stream_k) {
    const int64_t max_streamk_tiles = static_cast<int64_t>(num_ctas) + num_ctas / 10;
    if (max_streamk_tiles > kNumLocks) {
      fail("stream-K launch may need " + str(max_streamk_tiles) + " lock words but only " +
           str(kNumLocks) + " are allocated");
    }
    locks = locks_for(device, stream);
  }
  const void* input = a.input;
  const void* weight = a.weight;
  void* output = a.output;
  const void* weight_scale = a.weight_scale;
  const int32_t* sorted_ids = a.sorted_ids;
  const int32_t* expert_ids = a.expert_ids;
  const int32_t* num_tokens_padded = a.num_tokens_padded;
  void* locks_ptr = reinterpret_cast<void*>(locks);
  void* kernel_args[] = {&input,      &weight,     &output,            &weight_scale, &sorted_ids,
                         &expert_ids, &num_tokens_padded, &locks_ptr, &shape_m,      &top_k};

  CUlaunchConfig config = {};
  config.gridDimX = num_ctas;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = kd.num_threads;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = kd.smem_size;
  config.hStream = stream;
  jit::check(cuLaunchKernelEx(&config, kd.func, kernel_args, nullptr),
             "cuLaunchKernelEx (indexed)");
}

}  // namespace chord::indexed
