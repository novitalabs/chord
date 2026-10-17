// Generic C++ JIT; see jit.h.  The compile flags, std header shims and cache
// layout follow chord_kernels/operator/jit/compiler.py so that both paths build
// the same code; the cache directories are kept apart.

#include "jit.h"

#include <fcntl.h>
#include <nvrtc.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

// Optional build-time defaults (operator include dir, CUDA include dirs, cache
// dir), written by chord_kernels/operator/cpp_jit.py into its build dir.
// The CHORD_OPERATOR_INCLUDE_DIR / CHORD_CUDA_INCLUDE_DIRS / CHORD_CACHE_DIR
// environment variables override them at run time.
#if __has_include("chord_jit_defaults.h")
#include "chord_jit_defaults.h"
#endif
#ifndef CHORD_JIT_DEFAULT_OPERATOR_INCLUDE_DIR
#define CHORD_JIT_DEFAULT_OPERATOR_INCLUDE_DIR ""
#endif
#ifndef CHORD_JIT_DEFAULT_CUDA_INCLUDE_DIRS
#define CHORD_JIT_DEFAULT_CUDA_INCLUDE_DIRS ""
#endif
#ifndef CHORD_JIT_DEFAULT_CACHE_DIR
#define CHORD_JIT_DEFAULT_CACHE_DIR ""
#endif

namespace fs = std::filesystem;

namespace chord::jit {

void fail(const std::string& message) { throw std::runtime_error(message); }

void check(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  fail(std::string(what) + " failed with error: " + (name ? name : "?") + " (" +
       (text ? text : "?") + ")");
}

namespace {

void check_nvrtc(nvrtcResult result, const char* what) {
  if (result != NVRTC_SUCCESS) fail(std::string(what) + " failed: " + nvrtcGetErrorString(result));
}

std::string env_or(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string(fallback);
}

std::vector<std::string> split_paths(const std::string& value) {
  std::vector<std::string> out;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ':')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

std::string operator_include_dir() {
  std::string dir = env_or("CHORD_OPERATOR_INCLUDE_DIR", CHORD_JIT_DEFAULT_OPERATOR_INCLUDE_DIR);
  if (dir.empty()) fail("CHORD_OPERATOR_INCLUDE_DIR is not set and has no build-time default");
  return dir;
}

std::vector<std::string> cuda_include_dirs() {
  return split_paths(env_or("CHORD_CUDA_INCLUDE_DIRS", CHORD_JIT_DEFAULT_CUDA_INCLUDE_DIRS));
}

std::string cache_dir() {
  std::string dir = env_or("CHORD_CACHE_DIR", CHORD_JIT_DEFAULT_CACHE_DIR);
  if (dir.empty()) fail("CHORD_CACHE_DIR is not set and has no build-time default");
  return dir;
}

// Like Compiler.cuh_last_update_time: editing any .cuh invalidates the cache.
std::string cuh_last_update_time(const std::string& include_dir) {
  std::vector<std::pair<std::string, int64_t>> entries;
  for (const auto& entry : fs::recursive_directory_iterator(include_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".cuh") continue;
    const auto stamp = entry.last_write_time().time_since_epoch().count();
    entries.emplace_back(entry.path().generic_string(), static_cast<int64_t>(stamp));
  }
  std::sort(entries.begin(), entries.end());
  std::string out;
  for (const auto& [path, stamp] : entries) out += path + "=" + std::to_string(stamp) + "\n";
  return out;
}

std::string fnv1a_hex(const std::string& data) {
  uint64_t hash = 0xcbf29ce484222325ull;
  for (unsigned char byte : data) {
    hash ^= byte;
    hash *= 0x100000001b3ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
  return buf;
}

std::string read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot read " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_file(const fs::path& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) fail("cannot write " + path.string());
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  out.close();  // flushes; a failure here (e.g. disk full) must not publish a truncated file
  if (!out) fail("cannot write " + path.string());
}

// Exclusive flock on <dir>/lock for the life of the object.
class FileLock {
 public:
  explicit FileLock(const fs::path& path) {
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd_ < 0) fail("cannot open lock file " + path.string());
    if (::flock(fd_, LOCK_EX) != 0) {
      ::close(fd_);
      fail("cannot lock " + path.string());
    }
  }
  ~FileLock() {
    ::flock(fd_, LOCK_UN);
    ::close(fd_);
  }
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

 private:
  int fd_ = -1;
};

struct Compiled {
  std::string cubin;
  std::string name;
};

Compiled nvrtc_compile(
    const Program& program, const std::vector<std::string>& options, const fs::path& dir) {
  std::vector<const char*> header_sources, header_names;
  for (const auto& [name, content] : program.headers) {
    header_names.push_back(name.c_str());
    header_sources.push_back(content.c_str());
  }
  nvrtcProgram prog;
  check_nvrtc(
      nvrtcCreateProgram(
          &prog, program.source.c_str(), "kernel.cu", static_cast<int>(header_names.size()),
          header_sources.empty() ? nullptr : header_sources.data(),
          header_names.empty() ? nullptr : header_names.data()),
      "nvrtcCreateProgram");
  // Destroys the program on every exit path.
  std::unique_ptr<nvrtcProgram, void (*)(nvrtcProgram*)> guard(
      &prog, [](nvrtcProgram* p) { nvrtcDestroyProgram(p); });
  check_nvrtc(nvrtcAddNameExpression(prog, program.name_expression.c_str()),
              "nvrtcAddNameExpression");

  std::vector<const char*> opts;
  for (const auto& option : options) opts.push_back(option.c_str());
  const nvrtcResult compiled =
      nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());

  size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  std::string log(log_size, '\0');
  if (log_size > 0) nvrtcGetProgramLog(prog, log.data());
  while (!log.empty() && log.back() == '\0') log.pop_back();
  write_file(dir / "nvrtc.log", log);
  if (compiled != NVRTC_SUCCESS) {
    fail(std::string("NVRTC compile failed (") + nvrtcGetErrorString(compiled) + ") in " +
         dir.string() + ":\n" + log);
  }

  const char* lowered = nullptr;
  check_nvrtc(nvrtcGetLoweredName(prog, program.name_expression.c_str(), &lowered),
              "nvrtcGetLoweredName");
  Compiled out;
  out.name = lowered;
  size_t cubin_size = 0;
  check_nvrtc(nvrtcGetCUBINSize(prog, &cubin_size), "nvrtcGetCUBINSize");
  if (cubin_size == 0) fail("NVRTC produced an empty cubin (no --gpu-architecture=sm_XX?)");
  out.cubin.resize(cubin_size);
  check_nvrtc(nvrtcGetCUBIN(prog, out.cubin.data()), "nvrtcGetCUBIN");
  return out;
}

// ELF64 symbol table -> file offsets, as the launcher's CubinReader (elf.h).
#pragma pack(push, 1)
struct ElfHeader {
  unsigned char ident[16];
  uint16_t type, machine;
  uint32_t version;
  uint64_t entry, phoff, shoff;
  uint32_t flags;
  uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};
struct ElfSection {
  uint32_t name, type;
  uint64_t flags, addr, offset, size;
  uint32_t link, info;
  uint64_t addralign, entsize;
};
struct ElfSymbol {
  uint32_t name;
  unsigned char info, other;
  uint16_t shndx;
  uint64_t value, size;
};
#pragma pack(pop)

constexpr uint32_t kShtSymtab = 2;
constexpr uint32_t kShtNobits = 8;

std::map<std::string, uint64_t> read_symbol_offsets(const std::vector<char>& elf) {
  auto at = [&](uint64_t offset, uint64_t size) {
    if (offset > elf.size() || size > elf.size() - offset) fail("truncated cubin");
    return elf.data() + offset;
  };
  ElfHeader header;
  std::memcpy(&header, at(0, sizeof(header)), sizeof(header));
  std::vector<ElfSection> sections(header.shnum);
  std::memcpy(sections.data(), at(header.shoff, sizeof(ElfSection) * header.shnum),
              sizeof(ElfSection) * header.shnum);

  std::map<std::string, uint64_t> offsets;
  for (const auto& symtab : sections) {
    if (symtab.type != kShtSymtab || symtab.entsize == 0) continue;
    if (symtab.link >= sections.size()) fail("bad cubin symbol table");
    const ElfSection& strtab = sections[symtab.link];
    const char* names = at(strtab.offset, strtab.size);
    const uint64_t count = symtab.size / symtab.entsize;
    for (uint64_t i = 0; i < count; ++i) {
      ElfSymbol sym;
      std::memcpy(&sym, at(symtab.offset + i * symtab.entsize, sizeof(sym)), sizeof(sym));
      if (sym.name == 0 || sym.name >= strtab.size || sym.shndx >= sections.size()) continue;
      const std::string name(names + sym.name, strnlen(names + sym.name, strtab.size - sym.name));
      const ElfSection& owner = sections[sym.shndx];
      offsets[name] = owner.type == kShtNobits ? 0 : owner.offset + sym.value;
    }
  }
  return offsets;
}

}  // namespace

uint32_t Kernel::constant(const std::string& symbol) const {
  auto it = symbol_offsets_.find(symbol);
  if (it == symbol_offsets_.end()) fail("cubin has no symbol " + symbol);
  if (it->second == 0) return 0;  // zero-initialized (.bss)
  if (it->second + sizeof(uint32_t) > cubin_.size()) fail("symbol " + symbol + " is out of range");
  uint32_t value = 0;
  std::memcpy(&value, cubin_.data() + it->second, sizeof(value));
  return value;
}

int current_device() {
  CUdevice device = 0;
  const CUresult result = cuCtxGetDevice(&device);
  if (result == CUDA_ERROR_INVALID_CONTEXT) fail("no current CUDA context on this thread");
  check(result, "cuCtxGetDevice");
  return static_cast<int>(device);
}

int device_attribute(CUdevice_attribute attribute) {
  int value = 0;
  check(cuDeviceGetAttribute(&value, attribute, current_device()), "cuDeviceGetAttribute");
  return value;
}

int current_arch() {
  return device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10 +
         device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
}

int num_sms() { return device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT); }

std::vector<std::string> default_flags() {
  std::vector<std::string> flags = {
      "-std=c++17",
      "--use_fast_math",
      "--dopt=on",
      "-extra-device-vectorization",
      "--ptxas-options=-O3",
      "--ptxas-options=--register-usage-level=10",
      "--diag-suppress=39,161,174,177,940",
      "-default-device",
  };
  if (env_or("CHORD_LINEINFO", "0") == "1") flags.push_back("-lineinfo");
  return flags;
}

std::vector<std::pair<std::string, std::string>> std_header_shims() {
  static const std::string type_trait_imports = R"(
    namespace std {
    using cuda::std::is_same;
    using cuda::std::conditional_t;
    using cuda::std::conditional;
    using cuda::std::enable_if;
    using cuda::std::enable_if_t;
    }
)";
  return {
      {"climits", "#include <cuda/std/climits>"},
      {"cfloat", "#include <cuda/std/cfloat>"},
      {"cstddef", R"(
            #include <cuda/std/cstddef>
            using namespace cuda::std;
        )"},
      {"cstdint", R"(
            #include <cuda/std/cstdint>
            #include <cuda/std/type_traits>
            using namespace cuda::std;
        )" + type_trait_imports},
      {"type_traits", R"(
            #include <cuda/std/type_traits>
        )" + type_trait_imports},
  };
}

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Progress bar on stderr over `total` compiles; prints nothing when total is 0.
// On a terminal the bar is redrawn in place, otherwise (logs) one line per step.
class Progress {
 public:
  Progress(const std::string& label, int total)
      : label_(label), total_(total), tty_(::isatty(STDERR_FILENO) != 0), start_(Clock::now()) {
    draw();
  }
  ~Progress() {
    if (total_ > 0 && tty_) std::fputc('\n', stderr);
  }
  Progress(const Progress&) = delete;
  Progress& operator=(const Progress&) = delete;
  void step() {
    ++done_;
    draw();
  }

 private:
  void draw() {
    if (total_ == 0 || (!tty_ && done_ == 0)) return;
    constexpr int kWidth = 30;
    const int filled = done_ * kWidth / total_;
    std::fprintf(stderr, "%s[chord-jit] compiling %s [%s%s] %d/%d %.1fs%s", tty_ ? "\r" : "",
                 label_.c_str(), std::string(filled, '#').c_str(),
                 std::string(kWidth - filled, '.').c_str(), done_, total_,
                 ms_since(start_) / 1000.0, tty_ ? "" : "\n");
    std::fflush(stderr);
  }
  std::string label_;
  int total_;
  int done_ = 0;
  bool tty_;
  Clock::time_point start_;
};

// Everything the cache needs to know about one program on the current device.
struct Resolved {
  const Program* program;
  std::vector<std::string> options;
  std::string key;         // cache key: compiler, options, headers, source, .cuh mtimes
  std::string memo_key;    // device + key
  std::string cuh_stamp;   // the .cuh mtimes part of the key
  fs::path dir;
};

// Automatic precompiled headers (NVRTC -pch, CUDA 12.8+): every chord kernel
// source starts with one #include of its kernel header, so the compiles in one
// process after the first skip parsing it (B300: ~0.5 s -> ~0.04 s each, same
// cubin).  A PCH lives in this process's NVRTC instance only (NVRTC keeps the
// files in a per-process temp dir and deletes them on unload), and does not
// change the cubin, so it is not part of the cache key.  NVRTC does not check
// header mtimes when it reuses a PCH, so once a .cuh changes within the
// process, PCH stays off for the rest of it.  CHORD_JIT_PCH=0 turns it off.
std::vector<std::string> pch_options(const std::string& cuh_stamp) {
  static std::string first_stamp;  // guarded by g_mutex (every compile holds it)
  static bool stale = false;
  if (env_or("CHORD_JIT_PCH", "1") == "0") return {};
  if (first_stamp.empty()) first_stamp = cuh_stamp;
  if (cuh_stamp != first_stamp) stale = true;
  if (stale) return {};
  return {"-pch"};
}

Resolved resolve(const Program& program) {
  Resolved r;
  r.program = &program;
  const int arch = current_arch();
  const std::string include_dir = operator_include_dir();
  r.options = program.flags;
  r.options.insert(r.options.begin(),
                   "--gpu-architecture=sm_" + std::to_string(arch) + (arch >= 90 ? "a" : ""));
  r.options.push_back("-I" + include_dir);
  for (const auto& dir : cuda_include_dirs()) r.options.push_back("-I" + dir);

  int nvrtc_major = 0, nvrtc_minor = 0;
  check_nvrtc(nvrtcVersion(&nvrtc_major, &nvrtc_minor), "nvrtcVersion");
  r.key = "chord-cpp-jit-1\nnvrtc " + std::to_string(nvrtc_major) + "." +
          std::to_string(nvrtc_minor) + "\n";
  for (const auto& option : r.options) r.key += option + "\n";
  r.key += "name " + program.name_expression + "\n";
  for (const auto& [name, content] : program.headers) {
    r.key += "header " + name + "\n" + content + "\n";
  }
  r.cuh_stamp = cuh_last_update_time(include_dir);
  r.key += "source\n" + program.source + "\n" + r.cuh_stamp;
  r.memo_key = std::to_string(current_device()) + "\n" + r.key;
  r.dir = fs::path(cache_dir()) / "cpp" / program.cache_tag / fnv1a_hex(r.key);
  return r;
}

bool cached(const Resolved& r) { return fs::exists(r.dir / "kernel.cubin"); }

// The cubin and lowered name of `r`: from the cache, or compiled under the
// per-entry file lock (another process may be compiling the same entry).
Compiled compile_or_read(const Resolved& r) {
  const fs::path cubin_path = r.dir / "kernel.cubin";
  fs::create_directories(r.dir);

  Compiled compiled;
  auto read_cached = [&] {
    // kernel.cubin is published last, so key.txt and name.txt are complete.
    if (read_file(r.dir / "key.txt") != r.key) {
      fail("JIT cache hash collision in " + r.dir.string());
    }
    compiled.name = read_file(r.dir / "name.txt");
    compiled.cubin = read_file(cubin_path);
  };
  if (fs::exists(cubin_path)) {
    read_cached();
  } else {
    FileLock lock(r.dir / "lock");
    if (fs::exists(cubin_path)) {
      read_cached();
    } else {
      write_file(r.dir / "key.txt", r.key);
      write_file(r.dir / "kernel.cu", r.program->source);
      std::vector<std::string> options = r.options;
      for (auto& option : pch_options(r.cuh_stamp)) options.push_back(std::move(option));
      compiled = nvrtc_compile(*r.program, options, r.dir);
      write_file(r.dir / "name.txt", compiled.name);
      const fs::path temporary = r.dir / "kernel_tmp.cubin";
      write_file(temporary, compiled.cubin);
      fs::rename(temporary, cubin_path);
    }
  }
  return compiled;
}

std::mutex g_mutex;  // guards g_loaded and every compile in this process
std::map<std::string, std::unique_ptr<Kernel>> g_loaded;  // memo_key -> kernel

}  // namespace

// Fills the private parts of a Kernel (friend of Kernel).
class KernelLoader {
 public:
  static const Kernel& load(const Resolved& r, Progress* progress) {
    if (auto it = g_loaded.find(r.memo_key); it != g_loaded.end()) return *it->second;
    const Compiled compiled = compile_or_read(r);
    auto kernel = std::make_unique<Kernel>();
    kernel->name = compiled.name;
    kernel->cubin_.assign(compiled.cubin.begin(), compiled.cubin.end());
    kernel->symbol_offsets_ = read_symbol_offsets(kernel->cubin_);
    // From the cache file, as the Python JIT path's launcher (csrc/launcher) does.
    const std::string cubin_file = (r.dir / "kernel.cubin").string();
    check(cuModuleLoad(&kernel->module, cubin_file.c_str()), "cuModuleLoad");
    check(cuModuleGetFunction(&kernel->func, kernel->module, kernel->name.c_str()),
          "cuModuleGetFunction");
    if (progress != nullptr) progress->step();
    return *g_loaded.emplace(r.memo_key, std::move(kernel)).first->second;
  }
};

const Kernel& get_kernel(const Program& program) {
  std::lock_guard<std::mutex> hold(g_mutex);
  const Resolved r = resolve(program);
  if (auto it = g_loaded.find(r.memo_key); it != g_loaded.end()) return *it->second;
  if (cached(r)) return KernelLoader::load(r, nullptr);
  Progress progress(program.cache_tag + " kernel", 1);
  return KernelLoader::load(r, &progress);
}

void precompile(const std::vector<Program>& programs) {
  std::lock_guard<std::mutex> hold(g_mutex);
  std::vector<Resolved> todo;
  std::vector<bool> miss;
  std::map<std::string, bool> seen;
  for (const auto& program : programs) {
    Resolved r = resolve(program);
    if (g_loaded.count(r.memo_key) || !seen.emplace(r.memo_key, true).second) continue;
    miss.push_back(!cached(r));
    todo.push_back(std::move(r));
  }
  const int misses = static_cast<int>(std::count(miss.begin(), miss.end(), true));
  const std::string label =
      (programs.empty() ? std::string() : programs.front().cache_tag) + " kernels";
  Progress progress(label, misses);
  for (size_t i = 0; i < todo.size(); ++i) {
    KernelLoader::load(todo[i], miss[i] ? &progress : nullptr);
  }
}

CUtensorMap make_tma_2d(
    void* ptr, CUtensorMapDataType dtype, int64_t elem_size, int64_t gmem_inner_dim,
    int64_t gmem_outer_dim, int64_t smem_inner_dim, int64_t smem_outer_dim,
    int64_t gmem_outer_stride_elems, uint32_t swizzle_mode) {
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  switch (swizzle_mode) {
    case 0:
    case 16: swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
    case 32: swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
    case 64: swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
    case 128: swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    default: fail("unsupported TMA swizzle mode: " + std::to_string(swizzle_mode));
  }
  if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;

  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                   static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride_elems * elem_size)};
  const cuuint32_t elem_strides[2] = {1, 1};
  check(cuTensorMapEncodeTiled(
            &tensor_map, dtype, 2, ptr, gmem_dims, gmem_strides, smem_dims, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");
  return tensor_map;
}

CUtensorMap make_tma_3d(
    void* ptr, CUtensorMapDataType dtype, int64_t elem_size, int64_t gmem_inner_dim,
    int64_t gmem_outer_dim, int64_t gmem_batch_dim, int64_t smem_inner_dim,
    int64_t smem_outer_dim, int64_t gmem_outer_stride_elems, uint32_t swizzle_mode) {
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  switch (swizzle_mode) {
    case 0:
    case 16: swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
    case 32: swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
    case 64: swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
    case 128: swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    default: fail("unsupported TMA swizzle mode: " + std::to_string(swizzle_mode));
  }
  if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;

  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[3] = {
      static_cast<cuuint64_t>(gmem_inner_dim),
      static_cast<cuuint64_t>(gmem_outer_dim),
      static_cast<cuuint64_t>(gmem_batch_dim)};
  const cuuint32_t smem_dims[3] = {
      static_cast<cuuint32_t>(smem_inner_dim),
      static_cast<cuuint32_t>(smem_outer_dim), 1};
  const cuuint64_t gmem_strides[2] = {
      static_cast<cuuint64_t>(gmem_outer_stride_elems * elem_size),
      static_cast<cuuint64_t>(gmem_outer_stride_elems * gmem_outer_dim * elem_size)};
  const cuuint32_t elem_strides[3] = {1, 1, 1};
  check(cuTensorMapEncodeTiled(
            &tensor_map, dtype, 3, ptr, gmem_dims, gmem_strides, smem_dims, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");
  return tensor_map;
}

}  // namespace chord::jit
