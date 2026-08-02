# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import dataclasses
import math
from typing import ClassVar

import jinja2

from chord_kernels.operator.config import MmaOpClass
from chord_kernels.operator.jit.runtime import KernelRuntime


CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/humming.cuh>

class MmaOpClass {
public:
{{mma_op_class}}
};

class TuningConfig {
public:
  static constexpr uint32_t kNumStages = {{num_stages}};
  static constexpr uint32_t kNumCtasPerSm = {{num_ctas_per_sm}};
  static constexpr uint32_t kNumThreads = {{num_threads}};
  static constexpr bool kSwapAb = {{swap_ab}};
  static constexpr bool kUseStreamK = {{use_stream_k}};
};

using SharedStorageType = SharedStorage<
    Shape<{{block_shape[0]}}, {{block_shape[1]}}, {{block_shape[2]}}>,
    Shape<{{warp_shape[0]}}, {{warp_shape[1]}}, {{warp_shape[2]}}>,
    TuningConfig>;

extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(SharedStorageType);
extern "C" __constant__ uint32_t PROBLEM_SHAPE_N = {{shape_n}};
extern "C" __constant__ uint32_t PROBLEM_SHAPE_K = {{shape_k}};
extern "C" __constant__ uint32_t NUM_EXPERTS = {{num_experts}};
extern "C" __constant__ uint32_t NUM_THREADS = {{num_threads}};
extern "C" __constant__ uint32_t NUM_CTAS_PER_SM = {{num_ctas_per_sm}};
extern "C" __constant__ uint32_t USE_STREAM_K = {{use_stream_k_int}};
""")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclasses.dataclass(kw_only=True)
class HummingKernel(KernelRuntime):
    name: ClassVar[str] = "humming"

    shape_n: int
    shape_k: int
    num_experts: int
    block_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]
    num_stages: int = 4
    num_ctas_per_sm: int = 1
    swap_ab: bool = False
    use_stream_k: bool = False
    mma_type: str = "mma"

    def __post_init__(self) -> None:
        self.block_shape = tuple(self.block_shape)
        self.warp_shape = tuple(self.warp_shape)
        if not isinstance(self.mma_type, str):
            raise TypeError("mma_type must be 'mma' or 'wgmma'")
        self.mma_type = self.mma_type.lower()
        self._validate_config()
        KernelRuntime.__post_init__(self)

    def _validate_config(self) -> None:
        for name, value in (
            ("shape_n", self.shape_n),
            ("shape_k", self.shape_k),
            ("num_experts", self.num_experts),
            ("num_stages", self.num_stages),
            ("num_ctas_per_sm", self.num_ctas_per_sm),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        for name, shape in (
            ("block_shape", self.block_shape),
            ("warp_shape", self.warp_shape),
        ):
            if len(shape) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            ):
                raise ValueError(f"{name} must contain three positive integers")

        if not isinstance(self.swap_ab, bool):
            raise TypeError("swap_ab must be a bool")
        if not isinstance(self.use_stream_k, bool):
            raise TypeError("use_stream_k must be a bool")
        if self.mma_type not in ("mma", "wgmma"):
            raise ValueError(
                f"mma_type must be 'mma' or 'wgmma', got {self.mma_type!r}"
            )
        if self.swap_ab and self.mma_type != "mma":
            raise ValueError("swap_ab is only available for MMA kernels")
        if self.num_stages < 2:
            raise ValueError("num_stages must be at least 2")

        block_m, block_n, block_k = self.block_shape
        warp_m, warp_n, warp_k = self.warp_shape
        if self.shape_n % block_n or self.shape_k % block_k:
            raise ValueError("problem N/K must be divisible by the block N/K tile")
        if any(self.block_shape[i] % self.warp_shape[i] for i in range(3)):
            raise ValueError("block shape must be divisible by warp shape")
        if warp_n != (32 if self.mma_type == "wgmma" else 64):
            raise ValueError("WGMMA requires warp-N 32 and MMA requires warp-N 64")
        if warp_k < 32:
            raise ValueError("warp-K must be at least 32")

        power_of_two_values = (
            block_n,
            block_k,
            warp_n,
            warp_k,
            block_m // warp_m,
            block_n // warp_n,
            block_k // warp_k,
        )
        if not all(_is_power_of_two(value) for value in power_of_two_values):
            raise ValueError("indexed kernel tile ratios must be powers of two")

        num_threads = math.prod(self.block_shape) // math.prod(self.warp_shape) * 32
        if self.mma_type == "wgmma" and num_threads % 128:
            raise ValueError("WGMMA requires complete four-warp groups")

    @property
    def num_threads(self) -> int:
        return math.prod(self.block_shape) // math.prod(self.warp_shape) * 32

    def init_kernel(self) -> None:
        if self.sm_version < 90:
            raise RuntimeError("indexed W4A16 requires SM90 or newer")

        mma_op = self._select_mma_op()
        self.code = CODE_TEMPLATE.render(
            mma_op_class=mma_op.to_cpp_str(),
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            block_shape=self.block_shape,
            warp_shape=self.warp_shape,
            num_experts=self.num_experts,
            num_stages=self.num_stages,
            num_threads=self.num_threads,
            num_ctas_per_sm=self.num_ctas_per_sm,
            swap_ab=str(self.swap_ab).lower(),
            use_stream_k=str(self.use_stream_k).lower(),
            use_stream_k_int=int(self.use_stream_k),
        )
        self.kernel_expr = (
            "humming<\n"
            "    MmaOpClass,\n"
            f"    Shape<0, {self.shape_n}, {self.shape_k}>,\n"
            f"    Shape<{self.block_shape[0]}, {self.block_shape[1]}, {self.block_shape[2]}>,\n"
            f"    Shape<{self.warp_shape[0]}, {self.warp_shape[1]}, {self.warp_shape[2]}>,\n"
            "    TuningConfig>"
        )
        self.prepare()

    def load_cubin(self) -> None:
        from chord_kernels.operator import ops

        if self.cubin_loaded:
            return
        self.kernel_id = ops.register_kernel(self.kernel_filename, self.kernel_name)
        self.cubin_loaded = True

    def _select_mma_op(self):
        mma_m = self.warp_shape[0] if self.mma_type == "wgmma" else 16
        mma_n = 64 if self.mma_type == "wgmma" else 8
        mma_k = 16

        if (
            self.mma_type == "mma"
            and not self.swap_ab
            and self.warp_shape[0] % 16 == 8
        ):
            mma_m = 8
            mma_k = 8

        if self.mma_type == "wgmma":
            if self.warp_shape[0] % mma_m or self.warp_shape[1] % (mma_n // 4):
                raise ValueError("warp shape is incompatible with the WGMMA tile")
        elif self.swap_ab:
            if self.warp_shape[1] % mma_m or self.warp_shape[0] % mma_n:
                raise ValueError("warp shape is incompatible with swap-AB MMA")
        elif self.warp_shape[0] % mma_m or self.warp_shape[1] % mma_n:
            raise ValueError("warp shape is incompatible with the MMA tile")
        if self.warp_shape[2] % mma_k:
            raise ValueError("warp-K is incompatible with the MMA instruction")

        return MmaOpClass.from_config(self.mma_type, mma_m, mma_n, mma_k)

    def __call__(self):
        raise NotImplementedError(
            "launch HummingKernel through chord_kernels.operator.ops.launch_kernel()"
        )
