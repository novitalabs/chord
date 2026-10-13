"""C++ JIT for the indexed W4A16 GEMM and repack_w4a16 (``CHORD_CPP_JIT=1``).

On SM100 / SM103 with ``CHORD_CPP_JIT=1``, code generation, the cubin cache,
NVRTC, loading and launch of both kernels run in C++ (``csrc/jit`` +
``csrc/indexed``, no torch, no Python); ``csrc/indexed/torch_ops.cpp`` is a
thin torch op over it.  The configuration is still chosen in Python
(``tuning.py``) and passed in.  This module only builds that extension once
per process, like ``ops.init_launcher``, and forwards the calls.  Elsewhere
(SM90, or the switch off) the Python JIT path is used unchanged.

Call :func:`precompile_indexed` at startup so that no forward pass compiles: a
cache miss prints a progress bar on stderr, a cache hit prints nothing.
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
_CPP_JIT_CAPABILITIES = frozenset({(10, 0), (10, 3)})
_module = None


def enabled(device: torch.device | int) -> bool:
    """Whether ``CHORD_CPP_JIT=1`` and ``device`` is SM100 / SM103."""
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
                (_CSRC / "indexed" / "select.cpp").as_posix(),
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
) -> torch.Tensor:
    """Launch kernel ``kernel_id`` (from :func:`get_indexed_kernel`), as
    ``ops.launch_kernel`` does for the Python JIT path."""
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
    )


def repack_w4a16(
    inputs: torch.Tensor, outputs: torch.Tensor, *, packed: bool, wgmma_layout: bool
) -> torch.Tensor:
    """``RepackWeightKernel(is_weight_packed=packed, use_wgmma=wgmma_layout)``
    applied to ``inputs`` into ``outputs``."""
    _load()
    return torch.ops.chord_cpp.repack_w4a16(inputs, outputs, packed, wgmma_layout)


def precompile_indexed(
    kernels: Sequence[tuple[int, int, int, IndexedKernelConfig]],
    *,
    device: torch.device | int | str | None = None,
) -> None:
    """Compile (or read from the cache), load and register every
    ``(shape_n, shape_k, num_experts, config)`` on ``device`` (default: the
    current device), so that later calls never compile.  One progress bar on
    stderr over the cache misses; nothing when all are cached."""
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


def _to_config(fields: Sequence[int]) -> IndexedKernelConfig:
    from chord_kernels.operator.api import IndexedKernelConfig

    return IndexedKernelConfig(
        block_shape=tuple(fields[0:3]),
        warp_shape=tuple(fields[3:6]),
        num_stages=fields[6],
        num_ctas_per_sm=fields[7],
        swap_ab=bool(fields[8]),
        use_stream_k=bool(fields[9]),
        stream_k_fp32=bool(fields[10]),
    )


def default_config(
    shape_n: int, shape_k: int, num_experts: int, *, block_m: int, swap_ab: bool
) -> IndexedKernelConfig:
    """``IndexedKernelConfig.default`` (MMA layout, ``_compatible_block_n``),
    chosen in C++."""
    return _to_config(
        _load().default_config(shape_n, shape_k, num_experts, block_m, swap_ab)
    )


@functools.lru_cache(maxsize=None)
def _python_only_profiles() -> frozenset[str]:
    # Published non-Blackwell profiles: their tuning rows are not ported to C++
    # (validate_device keeps them off SM100 / SM103 anyway).
    from chord_kernels.operator.profiles import INDEXED_PROFILES

    return frozenset(INDEXED_PROFILES) - {"blackwell_decode_ep8"}


@functools.lru_cache(maxsize=256)
def _table_id(tuning_config: str, swap_ab: bool, layout: str) -> int | None:
    """Register a tuning_config JSON string with C++ once; ``None`` when it is
    not a list of rows (a single config, or an invalid type), which the Python
    path keeps."""
    from chord_kernels.operator.api import IndexedKernelConfig

    try:
        value = json.loads(tuning_config)
    except json.JSONDecodeError as exc:
        raise ValueError("tuning_config must be valid JSON") from exc
    if not isinstance(value, list):
        return None
    rows = []
    # As tuning._resolve_tuning_config parses a matched row, for every row.
    for lower, upper, row in value:
        row = dict(row)
        row.setdefault("swap_ab", swap_ab)
        if row.get("layout", layout) != layout:
            raise ValueError("tuning_config layout does not match the packed weight")
        c = IndexedKernelConfig.from_dict(row)
        rows.append(
            [
                int(lower),
                int(upper),
                *c.block_shape,
                *c.warp_shape,
                c.num_stages,
                c.num_ctas_per_sm,
                int(c.swap_ab),
                int(c.use_stream_k),
                int(c.stream_k_fp32),
            ]
        )
    return _load().register_table(rows)


def select_layer_config(
    meta,
    device: torch.device,
    selected_shape_m: object,
    tuning_config: object | None,
    block_m: object | None,
) -> IndexedKernelConfig | None:
    """The config ``dispatch.forward_w4a16`` picks for one indexed call
    (profile table / fallback, or the tuning_config rows, then the routing
    block-M), chosen in C++.  ``None`` when the call is not on the C++ path
    (switch off, not SM100 / SM103, WGMMA layout, a non-Blackwell published
    profile, or an argument form the Python path keeps); the caller then runs
    the Python resolution unchanged."""
    if not enabled(device) or meta.layout != "mma":
        return None
    if meta.profile.name in _python_only_profiles():
        return None
    if isinstance(selected_shape_m, bool) or not isinstance(selected_shape_m, int):
        return None
    if block_m is not None and (isinstance(block_m, bool) or not isinstance(block_m, int)):
        return None
    table = -1
    if tuning_config is not None:
        if not isinstance(tuning_config, str):
            return None
        table_id = _table_id(tuning_config, meta.swap_ab, meta.layout)
        if table_id is None:
            return None
        table = table_id
    layer = [
        meta.shape_n,
        meta.shape_k,
        meta.num_experts,
        int(meta.profile.name == "blackwell_decode_ep8"),
        meta.profile.block_m,
        int(meta.swap_ab),
    ]
    return _to_config(
        _load().select_config(
            layer, selected_shape_m, table, block_m is not None, block_m or 0
        )
    )


def layer_kernels(meta) -> list[tuple[int, int, int, IndexedKernelConfig]]:
    """Every ``(shape_n, shape_k, num_experts, config)`` the published tuning
    table of an indexed layer (``IndexedLayerMeta``) can pick, for
    :func:`precompile_indexed`."""
    from chord_kernels.operator.api import IndexedKernelConfig
    from chord_kernels.operator.tuning import _indexed_tuning_rows

    configs = []
    for _, _, row in _indexed_tuning_rows(meta):
        row = {k: v for k, v in row.items() if k not in ("block_m", "layout")}
        config = IndexedKernelConfig.from_dict(row)
        if config not in configs:
            configs.append(config)
    return [(meta.shape_n, meta.shape_k, meta.num_experts, c) for c in configs]


__all__ = [
    "default_config",
    "enabled",
    "get_indexed_kernel",
    "indexed_gemm",
    "layer_kernels",
    "precompile_indexed",
    "repack_w4a16",
    "select_layer_config",
]
