# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import ctypes
import dataclasses
import math
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from chord_kernels.operator.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/repack.cuh>

""")


@dataclasses.dataclass(kw_only=True)
class RepackWeightKernel(KernelRuntime):
    name: ClassVar[str] = "repack_w4a16"
    is_weight_packed: bool
    use_wgmma: bool = False

    def init_kernel(self):
        self.code = CODE_TEMPLATE.render()
        self.kernel_expr = (
            f"repack_w4a16<\n"
            f"    {int(self.is_weight_packed)},\n"
            f"    {int(self.use_wgmma)}>"
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        padded_shape_n: int | None = None,
        padded_shape_k: int | None = None,
        interleave_mode: int = 3,
    ):
        self.check_context()
        num_experts = 1 if inputs.ndim == 2 else inputs.size(0)
        shape_n = inputs.size(-2)
        shape_k = inputs.size(-1)
        if self.is_weight_packed:
            shape_k *= 8

        device = inputs.device

        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(shape_n / 64)
        config.gridDimY = math.ceil(shape_k / 64)
        config.gridDimZ = num_experts
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            shape_n,
            shape_k,
            padded_shape_n or shape_n,
            padded_shape_k or shape_k,
            interleave_mode,
        )

        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
