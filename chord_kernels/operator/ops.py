# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import os
import sys
from pathlib import Path

import torch
import torch.utils.cpp_extension
from filelock import FileLock

from chord_kernels.operator.utils import jit as jit_utils
from chord_kernels.operator.utils.cuda import filter_cuda_paths

_launcher_inited = False
_launcher_module = None


def _resolve_use_torch_stable_api() -> bool:
    from packaging.version import Version

    override = os.environ.get("CHORD_USE_TORCH_STABLE_API")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    return Version(torch.__version__.split("+")[0]) >= Version("2.11")


def _launcher_build_dir(use_torch_stable_api: bool) -> str:
    package_dir = Path(__file__).resolve().parent
    source_hash = jit_utils.hash_path_content(
        (package_dir / "csrc" / "launcher").as_posix(),
        relative=True,
    )
    py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_major, torch_minor = torch.__version__.split(".")[:2]
    abi_tag = "stable" if use_torch_stable_api else "nostable"
    dirname = Path(jit_utils.get_chord_cache_dir()) / "chord_launcher"
    dirname = (
        dirname
        / f"{py_version}_torch{torch_major}{torch_minor}_{abi_tag}"
        / source_hash
    )
    dirname.mkdir(exist_ok=True, parents=True)
    return dirname.as_posix()


def init_launcher() -> None:
    global _launcher_inited, _launcher_module
    if _launcher_inited:
        return

    use_stable_api = _resolve_use_torch_stable_api()
    lock_filename = jit_utils.get_chord_lock_filename("chord_launcher")
    with FileLock(lock_filename):
        if _launcher_inited:
            return

        package_dir = Path(__file__).resolve().parent
        build_dir = _launcher_build_dir(use_stable_api)
        stale_torch_lock = Path(build_dir) / "lock"
        stale_torch_lock.unlink(missing_ok=True)
        cuda_env = filter_cuda_paths(
            required_headers=["cuda.h", "crt/host_defines.h", "cuda/std/cstdint"],
        )
        _launcher_module = torch.utils.cpp_extension.load(
            name="chord_launcher",
            sources=[(package_dir / "csrc" / "launcher" / "launcher.cpp").as_posix()],
            extra_include_paths=list(cuda_env["include_paths"]),
            extra_ldflags=["-lcuda", "-lc10_cuda", "-ltorch_cuda"],
            extra_cflags=["-O3", f"-DUSE_TORCH_STABLE_API={int(use_stable_api)}"],
            build_directory=build_dir,
        )
        _launcher_inited = True


def register_kernel(cubin_path: str, func_name: str) -> int:
    init_launcher()
    return torch.ops.chord.register_kernel(cubin_path, func_name)


def launch_kernel(
    *,
    kernel_id: int,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    top_k: int,
    outputs: torch.Tensor | None = None,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    init_launcher()
    # Kernel selection is already performed by the indexed Python API.  Keep
    # valid_shape_m in this wrapper's signature for layer compatibility; the
    # narrowed launcher receives one concrete kernel id.
    del valid_shape_m
    return torch.ops.chord.launch_kernel(
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


__all__ = ["init_launcher", "launch_kernel", "register_kernel"]
