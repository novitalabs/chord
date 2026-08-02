# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import glob
import json
import os
import subprocess
from pathlib import Path

from filelock import FileLock

from chord_kernels.operator.utils import jit as jit_utils
from chord_kernels.operator.utils.cuda import filter_cuda_paths


def _select_nvrtc_lib(lib_dir):
    unversioned = os.path.join(lib_dir, "libnvrtc.so")
    if os.path.exists(unversioned):
        return unversioned
    versioned = sorted(glob.glob(os.path.join(lib_dir, "libnvrtc.so.*")))
    if versioned:
        return versioned[-1]
    return None


def _find_nvrtc_lib_dir():
    env = filter_cuda_paths(required_headers=["nvrtc.h"])
    root = env["path"]
    candidates = [
        os.path.join(root, "lib64"),
        os.path.join(root, "lib"),
        *sorted(glob.glob(os.path.join(root, "*", "lib64"))),
        *sorted(glob.glob(os.path.join(root, "*", "lib"))),
    ]
    for d in candidates:
        lib_path = _select_nvrtc_lib(d)
        if lib_path is not None:
            return d, lib_path, env
    return None, None, env


_cached_binary_path = None


def may_build_nvrtc_compile_binary():
    global _cached_binary_path
    if _cached_binary_path is not None:
        return _cached_binary_path

    src_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "nvrtc_compile.cpp")
    src_path = os.path.abspath(src_path)
    src_hash = jit_utils.hash_path_content(src_path, relative=True)

    lib_dir, lib_path, cuda_env = _find_nvrtc_lib_dir()
    if lib_dir is None:
        raise RuntimeError("Could not locate libnvrtc.so in CUDA path")
    include_paths = list(cuda_env["include_paths"])
    env_signature = json.dumps(
        {
            "lib_dir": lib_dir,
            "lib_path": lib_path,
            "include_paths": include_paths,
            "path": cuda_env["path"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    full_hash = jit_utils.hash_to_hex(src_hash + "$$" + env_signature)

    build_dir = Path(jit_utils.get_chord_cache_dir()) / "nvrtc_compile" / full_hash
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / "nvrtc_compile"

    if binary_path.exists():
        _cached_binary_path = binary_path.as_posix()
        return _cached_binary_path

    lock_filename = jit_utils.get_chord_lock_filename("nvrtc_compile_" + full_hash)
    with FileLock(lock_filename):
        if binary_path.exists():
            _cached_binary_path = binary_path.as_posix()
            return _cached_binary_path

        tmp_binary = binary_path.with_suffix(".tmp")
        cmd = [
            "g++",
            "-O2",
            "-std=c++17",
            src_path,
            *[f"-I{d}" for d in include_paths],
            lib_path,
            f"-Wl,-rpath,{lib_dir}",
            "-o",
            tmp_binary.as_posix(),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build nvrtc_compile:\nCMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        os.replace(tmp_binary, binary_path)
        _cached_binary_path = binary_path.as_posix()
        return _cached_binary_path
