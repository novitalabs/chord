// Generic C++ JIT: NVRTC in process, an on-disk cubin cache shared across
// processes, and a per-device loaded-kernel cache.  Knows no kernel; depends on
// the CUDA driver API and NVRTC only (no torch, no Python).
//
// Errors throw std::runtime_error.  Every call uses the calling thread's
// current CUDA context.

#pragma once

#include <cuda.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace chord::jit {

struct Program {
  std::string source;           // full translation unit (any prelude already prepended)
  std::string name_expression;  // the kernel instantiation whose lowered name is wanted
  std::vector<std::string> flags;  // NVRTC options without --gpu-architecture and -I
  std::vector<std::pair<std::string, std::string>> headers;  // in-memory headers: name, content
  std::string cache_tag;        // cache subdirectory, one per operator
};

class Kernel {
 public:
  CUmodule module = nullptr;
  CUfunction func = nullptr;
  std::string name;  // lowered (mangled) kernel name

  // Value of an `extern "C" __constant__ uint32_t` in the cubin.
  uint32_t constant(const std::string& symbol) const;

 private:
  friend class KernelLoader;  // jit.cpp
  std::vector<char> cubin_;
  std::map<std::string, uint64_t> symbol_offsets_;  // file offset; 0 for .bss symbols
};

// Compile (or read the cache) and load into the current context.  Within a
// process each (device, program) is compiled and loaded once; the returned
// reference stays valid for the life of the process.  A cache miss prints a
// one-kernel progress bar on stderr; a cache hit prints nothing.
const Kernel& get_kernel(const Program& program);

// get_kernel for every program ahead of time (duplicates are fine), with one
// progress bar on stderr over the cache misses; prints nothing when every
// program is already cached.
void precompile(const std::vector<Program>& programs);

// The flags chord's Python NVRTCCompiler.get_flags passes besides
// --gpu-architecture and -I (CHORD_LINEINFO=1 adds -lineinfo).  get_kernel /
// precompile also pass NVRTC -pch (automatic precompiled headers, not in the
// cache key: it does not change the cubin; CHORD_JIT_PCH=0 turns it off).
std::vector<std::string> default_flags();
// The std headers NVRTC lacks, mapped onto libcu++ (NVRTCCompiler._STD_HEADER_SHIMS).
std::vector<std::pair<std::string, std::string>> std_header_shims();

void check(CUresult result, const char* what);
[[noreturn]] void fail(const std::string& message);

int current_device();
int current_arch();  // major * 10 + minor: 90, 100, 103, ...
int num_sms();
int device_attribute(CUdevice_attribute attribute);

// Same arguments and behaviour as the launcher's make_dg_tma_2d_desc: a
// non-zero swizzle overrides the inner box extent with swizzle / elem_size.
CUtensorMap make_tma_2d(
    void* ptr, CUtensorMapDataType dtype, int64_t elem_size, int64_t gmem_inner_dim,
    int64_t gmem_outer_dim, int64_t smem_inner_dim, int64_t smem_outer_dim,
    int64_t gmem_outer_stride_elems, uint32_t swizzle_mode);

}  // namespace chord::jit
