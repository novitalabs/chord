# Derived from deepseek-ai/DeepGEMM; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.
"""NVRTC codegen + cubin registration for the vendored DeepGEMM W4A16 kernel.

Mirrors ``SM90FP8Gemm1D2DRuntime::generate_impl``: a fixed
instantiation of ``deep_gemm::sm90_w4a16_gemm_impl`` (upstream's
``sm90_fp8_gemm_1d2d_impl``, renamed here for the one instantiation this
distribution ships -- see SOURCE.md) compiled with
``SHAPE_M == 0`` (dynamic M at launch) and ``compiled_dims == "nk"`` (N/K
baked), the identity epilogue, and the weight scales on the SFA slot
(``kMajorSFB = MN``).  ``--use_fast_math`` and friends are NOT applied: the
upstream flags are reproduced via :class:`GroupedNVRTCCompiler`.

Launch metadata the C++ launcher cannot derive from the tensors (tile sizes,
swizzle modes, stage count folded into the smem size, grid extent, cluster)
is exported as ``extern "C" __constant__ uint32_t`` symbols and read out of
the cubin by the ELF symbol reader, exactly like the phase-1 humming kernel.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import jinja2

from chord_kernels.operator.grouped.heuristics import (
    W4A16_SCALE_GROUP,
    W4A16GemmConfig,
    W4A16GemmDesc,
)
from chord_kernels.operator.jit.compiler import (
    _STD_TYPE_TRAIT_IMPORTS,
    NVRTCCompiler,
)
from chord_kernels.operator.jit.runtime import KernelRuntime
from chord_kernels.operator.utils import jit as jit_utils


class GroupedNVRTCCompiler(NVRTCCompiler):
    """NVRTC flags matching the upstream W4A16 JIT (C++20, no fast-math)."""

    @classmethod
    def signature(cls):
        return "dg-" + super().signature()

    @classmethod
    def device_prelude(cls):
        """Bridge the two NVRTC gaps in the vendored DeepGEMM include closure.

        1. CUTLASS routes to ``cuda/std`` (or ``cccl/cuda/std`` from CUDA 13) under
           ``__CUDACC_RTC__`` and never includes bare ``<type_traits>``, so the
           ``--header`` shims never fire for this closure and the ``std::`` names
           the kernel spells must be imported here instead.
        2. Programmatic dependent launch is a CUDA runtime builtin that NVRTC does
           not declare; the kernel's two call sites map onto the PTX directly.
        """
        return (
            """
            #include <cuda/std/type_traits>
            #include <cuda/std/utility>
            #include <cuda/std/cstdint>
            #include <cuda/std/__algorithm_>
            """
            + _STD_TYPE_TRAIT_IMPORTS
            + """
            namespace std { using cuda::std::min; }

            __device__ __forceinline__ void cudaGridDependencySynchronize() {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            __device__ __forceinline__ void cudaTriggerProgrammaticLaunchCompletion() {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            """
        )

    @classmethod
    def get_flags(cls, sm_version):
        flags = super().get_flags(sm_version)
        # Replace the phase-1 C++17 baseline with the upstream W4A16 baseline
        # (-std=c++20, --device-int128) and drop the non-upstream extras
        # (fast-math/device-vectorization/--dopt) so the codegen matches
        # DeepGEMM's own NVRTC builds of this kernel.
        flags[flags.index("-std=c++17")] = "-std=c++20"
        for flag in ("--use_fast_math", "-extra-device-vectorization", "--dopt=on"):
            flags.remove(flag)
        flags.append("--device-int128")
        suppress = "--diag-suppress=39,161,174,177,940"
        flags[flags.index(suppress)] = "--diag-suppress=39,161,174,177,186,940"
        return flags


CODE_TEMPLATE = jinja2.Template("""
#include <deep_gemm/impls/sm90_w4a16_gemm.cuh>

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_w4a16_gemm_impl<
        cute::UMMA::Major::K,                           // kMajorSFB (upstream spelling; dead slot under W4A16)
        {{shape_m}}, {{shape_n}}, {{shape_k}},        // SHAPE_M is 0: runtime m
        {{num_groups}},
        {{config.layout.block_m}}, {{config.layout.block_n}}, {{config.layout.block_k}},
        {{config.storage.swizzle_a_mode}}, {{config.storage.swizzle_b_mode}}, {{config.storage.swizzle_cd_mode}},
        {{config.pipeline.num_stages}},
        {{config.launch.num_tma_threads}}, {{config.launch.num_math_threads}},
        {{config.layout.cluster_size}}, {{multicast_on_a}},
        {{desc.num_sms}}, deep_gemm::GemmType::{{gemm_type_cpp}},
        deep_gemm::epilogue::transform::EpilogueIdentity,
        true, {{scale_group}}
    >);
}

extern "C" __constant__ uint32_t SMEM_SIZE = {{config.pipeline.smem_size}};
extern "C" __constant__ uint32_t NUM_THREADS = {{config.launch.num_threads}};
extern "C" __constant__ uint32_t GRID_DIM = {{desc.num_sms}};
extern "C" __constant__ uint32_t CLUSTER_DIM = {{config.layout.cluster_size}};
extern "C" __constant__ uint32_t GRP_BLOCK_M = {{config.layout.block_m}};
extern "C" __constant__ uint32_t GRP_BLOCK_N = {{config.layout.block_n}};
extern "C" __constant__ uint32_t GRP_BLOCK_K = {{config.layout.block_k}};
extern "C" __constant__ uint32_t GRP_SWIZZLE_A = {{config.storage.swizzle_a_mode}};
extern "C" __constant__ uint32_t GRP_SWIZZLE_B = {{config.storage.swizzle_b_mode}};
extern "C" __constant__ uint32_t GRP_SWIZZLE_CD = {{config.storage.swizzle_cd_mode}};
extern "C" __constant__ uint32_t GRP_NUM_GROUPS = {{num_groups}};
extern "C" __constant__ uint32_t GRP_SCALE_GROUP = {{scale_group}};
extern "C" __constant__ uint32_t GRP_GEMM_TYPE = {{gemm_type_int}};
""")


_KERNEL_SYMBOL = "sm90_w4a16_gemm_impl"
_GEMM_TYPE_CPP = {"masked": "MGroupedMasked", "contiguous": "MGroupedContiguous"}
_GEMM_TYPE_INT = {"masked": 2, "contiguous": 1}


@dataclasses.dataclass(kw_only=True)
class GroupedW4A16Kernel(KernelRuntime):
    """Compile-once/launch-many handle for one W4A16 shape+regime.

    ``desc`` and ``config`` fully determine the cubin; the
    :class:`~chord_kernels.operator.jit.runtime.KernelRuntime` instance cache
    guarantees one JIT compile per distinct instantiation.
    """

    name: ClassVar[str] = _KERNEL_SYMBOL

    desc: W4A16GemmDesc
    config: W4A16GemmConfig

    def __post_init__(self) -> None:
        if not isinstance(self.desc, W4A16GemmDesc):
            raise TypeError("desc must be a W4A16GemmDesc")
        if not isinstance(self.config, W4A16GemmConfig):
            raise TypeError("config must be a W4A16GemmConfig")
        KernelRuntime.__post_init__(self)

    def init_kernel(self) -> None:
        if self.sm_version != 90:
            raise RuntimeError(
                "the grouped W4A16 backend requires SM90, got "
                f"sm_{self.sm_version}"
            )
        gemm_type = self.desc.gemm_type
        self.code = CODE_TEMPLATE.render(
            shape_m=0,  # dynamic per launch (upstream compiled_dims="nk" keeps N/K baked)
            shape_n=self.desc.n,
            shape_k=self.desc.k,
            num_groups=self.desc.num_groups,
            config=self.config,
            desc=self.desc,
            gemm_type_cpp=_GEMM_TYPE_CPP[gemm_type],
            gemm_type_int=_GEMM_TYPE_INT[gemm_type],
            multicast_on_a="true" if self.config.layout.cluster_n > 1 else "false",
            scale_group=W4A16_SCALE_GROUP,
        )
        expr = (
            "deep_gemm::sm90_w4a16_gemm_impl<\n"
            "cute::UMMA::Major::K,\n"
            f"0, {self.desc.n}, {self.desc.k},\n"
            f"{self.desc.num_groups},\n"
            f"{self.config.layout.block_m}, {self.config.layout.block_n}, {self.config.layout.block_k},\n"
            f"{self.config.storage.swizzle_a_mode}, {self.config.storage.swizzle_b_mode}, {self.config.storage.swizzle_cd_mode},\n"
            f"{self.config.pipeline.num_stages},\n"
            f"{self.config.launch.num_tma_threads}, {self.config.launch.num_math_threads},\n"
            f"{self.config.layout.cluster_size}, {'true' if self.config.layout.cluster_n > 1 else 'false'},\n"
            f"{self.desc.num_sms}, deep_gemm::GemmType::{_GEMM_TYPE_CPP[gemm_type]},\n"
            "deep_gemm::epilogue::transform::EpilogueIdentity,\n"
            f"true, {W4A16_SCALE_GROUP}>"
        )
        self.kernel_expr = expr
        self.prepare()

    def prepare(self) -> None:
        # Same flow as KernelRuntime.prepare, but through the grouped flag set.
        self._ensure_cuda_context()
        self.kernel_filename = GroupedNVRTCCompiler.compile(
            self.code,
            sm_version=self.sm_version_str,
            kernel_expr=self.kernel_expr,
        )
        self.kernel_name = jit_utils.find_kernel_name_in_cubin(
            self.kernel_filename, self.name, nested=True
        )
        import threading

        if threading.current_thread() is threading.main_thread():
            self.load_cubin()

    def load_cubin(self) -> None:
        from chord_kernels.operator import ops

        if self.cubin_loaded:
            return
        self.kernel_id = ops.register_grouped_w4a16_kernel(
            self.kernel_filename, self.kernel_name
        )
        self.cubin_loaded = True

    def __call__(self):
        raise NotImplementedError(
            "launch GroupedW4A16Kernel through chord_kernels.operator.ops"
        )


__all__ = ["GroupedNVRTCCompiler", "GroupedW4A16Kernel"]
