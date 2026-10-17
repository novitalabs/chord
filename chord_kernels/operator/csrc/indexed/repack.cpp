// repack_w4a16 on the generic C++ JIT; see repack.h.  Generated source:
// kernel/repack_weight.py CODE_TEMPLATE / RepackWeightKernel.init_kernel.
// Launch: RepackWeightKernel.__call__.  Keep each in step with its Python
// counterpart while both paths exist.

#include "repack.h"

#include <string>

#include "../jit/jit.h"

namespace chord::indexed {

namespace {

jit::Program make_program(bool packed_input) {
  jit::Program program;
  // kernel_expr, whitespace collapsed as NVRTCCompiler._compile does.
  program.name_expression = std::string("repack_w4a16<") + (packed_input ? "1" : "0") + ">";
  // CODE_TEMPLATE rendered (jinja2 drops the final newline).
  program.source = "\n#include <humming/kernel/repack.cuh>\n";
  program.flags = jit::default_flags();
  program.headers = jit::std_header_shims();
  program.cache_tag = "repack";
  return program;
}

}  // namespace

void repack_w4a16(const RepackArgs& a, CUstream stream) {
  if (a.input == nullptr || a.output == nullptr) jit::fail("repack_w4a16: null tensor pointer");
  if (a.num_experts <= 0 || a.shape_n <= 0 || a.shape_k <= 0) {
    jit::fail("repack_w4a16: dimensions must be positive");
  }
  const jit::Kernel& kernel = jit::get_kernel(make_program(a.packed_input));

  const void* input = a.input;
  void* output = a.output;
  uint32_t shape_n = static_cast<uint32_t>(a.shape_n);
  uint32_t shape_k = static_cast<uint32_t>(a.shape_k);
  uint32_t padded_shape_n = shape_n;
  uint32_t padded_shape_k = shape_k;
  uint32_t interleave_mode = 3;
  void* kernel_args[] = {&input,          &output,         &shape_n,        &shape_k,
                         &padded_shape_n, &padded_shape_k, &interleave_mode};

  CUlaunchConfig config = {};
  config.gridDimX = static_cast<uint32_t>((a.shape_n + 63) / 64);
  config.gridDimY = static_cast<uint32_t>((a.shape_k + 63) / 64);
  config.gridDimZ = static_cast<uint32_t>(a.num_experts);
  config.blockDimX = 32;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.hStream = stream;
  jit::check(cuLaunchKernelEx(&config, kernel.func, kernel_args, nullptr),
             "cuLaunchKernelEx (repack_w4a16)");
}

}  // namespace chord::indexed
