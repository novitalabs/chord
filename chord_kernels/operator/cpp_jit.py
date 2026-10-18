"""C++ JIT for every chord kernel (``CHORD_CPP_JIT=1``, SM90/SM100/SM103).

Codegen, cubin cache, NVRTC, load and launch run in ``csrc/``; configs are
still picked in Python.  This module builds the extension once per process and
forwards calls.  See docs/cpp_jit.md.  :func:`precompile_indexed` at startup
keeps forward passes from compiling.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.utils.cpp_extension
from filelock import FileLock

from chord_kernels.operator.jit.compiler import NVRTCCompiler
from chord_kernels.operator.ops import _resolve_use_torch_stable_api
from chord_kernels.operator.utils import jit as jit_utils
from chord_kernels.operator.utils.cuda import filter_cuda_paths
from chord_kernels.operator.utils.nvrtc import _find_nvrtc_lib_dir

if TYPE_CHECKING:
    from chord_kernels.operator.api import IndexedKernelConfig

_CSRC = Path(__file__).resolve().parent / "csrc"
_CPP_JIT_CAPABILITIES = frozenset({(9, 0), (10, 0), (10, 3)})
_module = None


def enabled(device: torch.device | int) -> bool:
    """Whether ``CHORD_CPP_JIT=1`` and ``device`` is SM90 / SM100 / SM103."""
    if os.environ.get("CHORD_CPP_JIT", "0") != "1":
        return False
    if isinstance(device, torch.device) and device.type != "cuda":
        return False
    index = device if isinstance(device, int) else device.index
    if index is None:
        index = torch.cuda.current_device()
    return _capability(index) in _CPP_JIT_CAPABILITIES


@functools.lru_cache(maxsize=None)
def _capability(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def _defaults_header() -> str:
    """Build-time defaults of the C++ JIT: the -I list and cache dir the Python
    NVRTCCompiler uses, so both paths compile the same code."""
    include_dirs = NVRTCCompiler._get_include_dirs()
    operator_dir, cuda_dirs = include_dirs[0], include_dirs[1:]
    assert operator_dir == NVRTCCompiler.operator_include_dir()
    values = {
        "CHORD_JIT_DEFAULT_OPERATOR_INCLUDE_DIR": operator_dir,
        "CHORD_JIT_DEFAULT_CUDA_INCLUDE_DIRS": ":".join(cuda_dirs),
        "CHORD_JIT_DEFAULT_CACHE_DIR": jit_utils.get_chord_cache_dir(),
    }
    return "".join(f"#define {k} {json.dumps(v)}\n" for k, v in values.items())


def _load():
    global _module
    if _module is not None:
        return _module
    use_stable_api = _resolve_use_torch_stable_api()
    defaults = _defaults_header()
    lib_dir, lib_path, nvrtc_env = _find_nvrtc_lib_dir()
    if lib_dir is None:
        raise RuntimeError("Could not locate libnvrtc.so in CUDA path")
    cuda_env = filter_cuda_paths(
        required_headers=["cuda.h", "crt/host_defines.h", "cuda/std/cstdint"],
    )
    include_paths = list(
        dict.fromkeys([*cuda_env["include_paths"], *nvrtc_env["include_paths"]])
    )

    source_hash = jit_utils.hash_to_hex(
        "$$".join(
            [
                jit_utils.hash_path_content((_CSRC / "jit").as_posix(), relative=True),
                jit_utils.hash_path_content((_CSRC / "indexed").as_posix(), relative=True),
                jit_utils.hash_path_content((_CSRC / "grouped").as_posix(), relative=True),
                *(
                    jit_utils.hash_path_content(
                        (_CSRC / "launcher" / name).as_posix(), relative=True
                    )
                    for name in ("torch_api.h", "tensor.h", "utils.h")
                ),
                defaults,
                lib_path,
                *include_paths,
            ]
        )
    )
    torch_major, torch_minor = torch.__version__.split(".")[:2]
    build_dir = (
        Path(jit_utils.get_chord_cache_dir())
        / "chord_cpp_jit"
        / (
            f"py{sys.version_info.major}{sys.version_info.minor}"
            f"_torch{torch_major}{torch_minor}"
            f"_{'stable' if use_stable_api else 'nostable'}"
        )
        / source_hash
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(jit_utils.get_chord_lock_filename("chord_cpp_jit")):
        if _module is not None:
            return _module
        # As ops.init_launcher: a process killed mid-build leaves torch's own
        # build lock behind, which would block every later load.
        (build_dir / "lock").unlink(missing_ok=True)
        # Rewriting an unchanged header would bump its mtime and make ninja
        # rebuild the extension in every new process.
        defaults_path = build_dir / "chord_jit_defaults.h"
        if not defaults_path.is_file() or defaults_path.read_text() != defaults:
            defaults_path.write_text(defaults)
        _module = torch.utils.cpp_extension.load(
            name="chord_cpp_jit",
            sources=[
                (_CSRC / "jit" / "jit.cpp").as_posix(),
                (_CSRC / "indexed" / "humming.cpp").as_posix(),
                (_CSRC / "indexed" / "repack.cpp").as_posix(),
                (_CSRC / "grouped" / "grouped.cpp").as_posix(),
                (_CSRC / "indexed" / "torch_ops.cpp").as_posix(),
            ],
            extra_include_paths=[build_dir.as_posix(), *include_paths],
            extra_cflags=["-O3", f"-DUSE_TORCH_STABLE_API={int(use_stable_api)}"],
            extra_ldflags=[
                "-lcuda",
                "-lc10_cuda",
                "-ltorch_cuda",
                lib_path,
                f"-Wl,-rpath,{lib_dir}",
            ],
            build_directory=build_dir.as_posix(),
        )
        return _module


def _config_fields(
    shape_n: int, shape_k: int, num_experts: int, config: IndexedKernelConfig
) -> list[int]:
    return [
        shape_n,
        shape_k,
        num_experts,
        *config.block_shape,
        *config.warp_shape,
        config.num_stages,
        config.num_ctas_per_sm,
        int(config.swap_ab),
        int(config.use_stream_k),
        int(config.stream_k_fp32),
    ]


@functools.lru_cache(maxsize=128)
def get_indexed_kernel(
    device_index: int,
    shape_n: int,
    shape_k: int,
    num_experts: int,
    config: IndexedKernelConfig,
) -> int:
    """Compile (or read from the cache), load and register one indexed kernel
    on ``device_index``; returns its id.  A bad config raises ValueError."""
    module = _load()
    with torch.cuda.device(device_index):
        return module.register_gemm(
            device_index, _config_fields(shape_n, shape_k, num_experts, config)
        )


def indexed_gemm(
    kernel_id: int,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    top_k: int,
    outputs: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch kernel ``kernel_id`` (from :func:`get_indexed_kernel`), as
    ``ops.launch_kernel`` does for the Python JIT path; ``locks`` is the
    caller-owned stream-K lock buffer (``w4a16_indexed``)."""
    _load()
    return torch.ops.chord_cpp.indexed_gemm(
        kernel_id,
        inputs,
        weight,
        outputs,
        weight_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        top_k,
        locks,
    )


def repack_w4a16(inputs: torch.Tensor, outputs: torch.Tensor, *, packed: bool) -> torch.Tensor:
    """``RepackWeightKernel(is_weight_packed=packed)`` applied to ``inputs``
    into ``outputs``."""
    _load()
    return torch.ops.chord_cpp.repack_w4a16(inputs, outputs, packed)


def precompile_indexed(
    kernels: Sequence[tuple[int, int, int, IndexedKernelConfig]],
    *,
    device: torch.device | int | str | None = None,
) -> None:
    """Compile or load every ``(shape_n, shape_k, num_experts, config)`` on
    ``device`` so later calls never compile; a progress bar over cache misses."""
    index = torch.device(device if device is not None else "cuda").index
    if index is None:
        index = torch.cuda.current_device()
    module = _load()
    with torch.cuda.device(index):
        module.precompile_gemm(
            index, [_config_fields(n, k, e, c) for n, k, e, c in kernels]
        )
    for n, k, e, c in kernels:
        get_indexed_kernel(index, n, k, e, c)  # registered above: no compile, no load


@functools.lru_cache(maxsize=128)
def get_grouped_kernel(device_index: int, desc, config) -> int:
    """Compile (or read from the cache), load and register one grouped kernel
    (``W4A16GemmDesc`` + ``W4A16GemmConfig``) on ``device_index``; returns its id."""
    fields = [
        2 if desc.gemm_type == "masked" else 1,
        desc.n,
        desc.k,
        desc.num_groups,
        desc.num_sms,
        config.layout.block_m,
        config.layout.block_n,
        config.layout.block_k,
        config.storage.swizzle_a_mode,
        config.storage.swizzle_b_mode,
        config.storage.swizzle_cd_mode,
        config.pipeline.num_stages,
        config.pipeline.smem_size,
        config.launch.num_tma_threads,
        config.launch.num_math_threads,
        config.launch.num_threads,
        config.layout.cluster_size,
        int(config.layout.cluster_n > 1),
        config.launch.num_ctas_per_sm,
    ]
    module = _load()
    with torch.cuda.device(device_index):
        return module.register_grouped(device_index, fields)


def grouped_masked(kernel_id, inputs, weight, scale, masked_m, outputs, enable_pdl):
    """``ops.launch_grouped_w4a16_masked`` on the C++ JIT path."""
    _load()
    return torch.ops.chord_cpp.grouped_masked(
        kernel_id, inputs, weight, scale, masked_m, outputs, enable_pdl
    )


def grouped_contiguous(kernel_id, inputs, weight, scale, m_indices, outputs, enable_pdl):
    """``ops.launch_grouped_w4a16_contiguous`` on the C++ JIT path."""
    _load()
    return torch.ops.chord_cpp.grouped_contiguous(
        kernel_id, inputs, weight, scale, m_indices, outputs, enable_pdl
    )


def indexed_kernels(
    shape_n: int,
    shape_k: int,
    num_experts: int,
    *,
    device: torch.device | int | str | None = None,
    capability: tuple[int, int] | None = None,
) -> list[tuple[int, int, int, IndexedKernelConfig]]:
    """Every distinct ``(shape_n, shape_k, num_experts, config)`` that
    ``select_indexed_config`` picks for one projection on this GPU, for
    :func:`precompile_indexed`."""
    from chord_kernels.operator.tuning import select_indexed_config

    configs: list[IndexedKernelConfig] = []
    # The tuned schedules settle by routed_m = 368 * experts (prefill's last
    # block-M step); the sentinel covers anything past it.
    for routed_m in [*range(1, max(4096, 368 * num_experts) + 1), 1 << 29]:
        config = select_indexed_config(
            shape_n, shape_k, num_experts, routed_m, device=device, capability=capability
        )
        if config not in configs:
            configs.append(config)
    return [(shape_n, shape_k, num_experts, c) for c in configs]


__all__ = [
    "enabled",
    "get_grouped_kernel",
    "get_indexed_kernel",
    "grouped_contiguous",
    "grouped_masked",
    "indexed_gemm",
    "indexed_kernels",
    "precompile_indexed",
    "repack_w4a16",
]
