# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import functools
import glob
import json
import os
import re
import sys


def _parse_major_minor(value: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)\.(\d+)", value)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d+)", value)
    if match:
        return int(match.group(1)), 0
    return None, None


def _read_cuda_system_version(cuda_home: str) -> tuple[int | None, int | None]:
    version_json = os.path.join(cuda_home, "version.json")
    if os.path.isfile(version_json):
        try:
            with open(version_json) as version_file:
                version = json.load(version_file).get("cuda", {}).get("version")
            if version:
                return _parse_major_minor(version)
        except (OSError, ValueError):
            pass

    version_txt = os.path.join(cuda_home, "version.txt")
    if os.path.isfile(version_txt):
        try:
            with open(version_txt) as version_file:
                return _parse_major_minor(version_file.read())
        except OSError:
            pass

    basename = os.path.basename(os.path.realpath(cuda_home))
    match = re.match(r"cuda-(\d+(?:\.\d+)?)$", basename)
    if match:
        return _parse_major_minor(match.group(1))
    return None, None


def _add_include_path(paths: list[str], include_dir: str) -> None:
    if not os.path.isdir(include_dir):
        return
    paths.append(include_dir)
    cccl = os.path.join(include_dir, "cccl")
    if os.path.isdir(cccl):
        paths.append(cccl)


def _collect_include_paths(root: str, recurse_components: bool = False) -> list[str]:
    paths: list[str] = []
    _add_include_path(paths, os.path.join(root, "include"))
    if recurse_components:
        for name in sorted(os.listdir(root)):
            if name.startswith("cu") and name[2:].isdigit():
                continue
            _add_include_path(paths, os.path.join(root, name, "include"))
    return paths


def _find_nvidia_pypi_cuda_paths() -> list[dict]:
    results = []
    seen = set()
    for entry in sys.path:
        nvidia_root = os.path.join(entry, "nvidia")
        if not os.path.isdir(nvidia_root) or nvidia_root in seen:
            continue
        seen.add(nvidia_root)

        if any(
            os.path.isdir(os.path.join(nvidia_root, component))
            for component in ("cuda_runtime", "cuda_nvrtc")
        ):
            results.append(
                {
                    "source": "pypi",
                    "path": nvidia_root,
                    "major": 12,
                    "minor": None,
                    "include_paths": _collect_include_paths(
                        nvidia_root, recurse_components=True
                    ),
                }
            )

        cu13_root = os.path.join(nvidia_root, "cu13")
        if os.path.isdir(os.path.join(cu13_root, "include")):
            results.append(
                {
                    "source": "pypi",
                    "path": cu13_root,
                    "major": 13,
                    "minor": None,
                    "include_paths": _collect_include_paths(cu13_root),
                }
            )
    return results


def filter_cuda_paths(required_headers: list[str] | None = None) -> dict:
    import torch

    target_major, _ = _parse_major_minor(str(torch.version.cuda))
    if target_major is None:
        raise RuntimeError("PyTorch does not report a CUDA Toolkit version")
    headers = required_headers or []

    def has_headers(environment: dict) -> bool:
        return all(
            any(
                os.path.exists(os.path.join(include_path, header))
                for include_path in environment["include_paths"]
            )
            for header in headers
        )

    matches = [
        environment
        for environment in find_all_cuda_paths()
        if environment["major"] == target_major and has_headers(environment)
    ]
    matches.sort(key=lambda item: (item["minor"] is None, -(item["minor"] or 0)))
    if matches:
        return matches[0]

    header_text = ", ".join(headers) if headers else "CUDA headers"
    raise RuntimeError(
        f"No CUDA {target_major} Toolkit environment provides {header_text}. "
        "Install a matching system Toolkit or NVIDIA CUDA Python packages."
    )


@functools.lru_cache(maxsize=1)
def find_all_cuda_paths() -> list[dict]:
    results = []
    seen_real = set()
    candidates = ["/usr/local/cuda", *sorted(glob.glob("/usr/local/cuda-*"))]
    configured = os.environ.get("CUDA_HOME")
    if configured:
        candidates.append(configured)

    for path in candidates:
        if not os.path.isdir(path):
            continue
        real_path = os.path.realpath(path)
        if real_path in seen_real:
            continue
        seen_real.add(real_path)
        major, minor = _read_cuda_system_version(path)
        results.append(
            {
                "source": "system",
                "path": path,
                "real_path": real_path,
                "major": major,
                "minor": minor,
                "include_paths": _collect_include_paths(path),
            }
        )

    results.extend(_find_nvidia_pypi_cuda_paths())
    return results
