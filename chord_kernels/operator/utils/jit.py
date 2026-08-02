# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import functools
import glob
import hashlib
import os
import re
from pathlib import Path

from elftools.elf.elffile import ELFFile


def find_kernel_name_in_cubin(filename: str, func_keyword: str) -> str:
    with open(filename, "rb") as cubin:
        symbol_table = ELFFile(cubin).get_section_by_name(".symtab")
        symbol_names = [
            symbol.name
            for symbol in symbol_table.iter_symbols()
            if symbol["st_info"]["type"] == "STT_FUNC"
            and re.findall(f"^_Z\\d+{func_keyword}", symbol.name)
        ]
    if len(symbol_names) != 1:
        raise RuntimeError(
            f"expected one {func_keyword!r} kernel in {filename}, got {symbol_names}"
        )
    return symbol_names[0]


def hash_to_hex(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def get_chord_tmp_dir() -> str:
    """Return the package-local temporary directory used by this operator.

    The public Humming package uses ``~/.humming``.  This extracted operator
    deliberately keeps its build artifacts in a separate namespace so that it
    can coexist with an upstream Humming installation in the same process.
    ``CHORD_TMP_DIR`` is useful for read-only site-packages and multi-process
    deployments.
    """
    configured = os.getenv("CHORD_TMP_DIR")
    if configured is not None:
        return configured
    # ``jit.py`` lives at <root>/chord_kernels/operator/utils/jit.py in a source
    # checkout.  Keeping the default next to the package makes generated
    # artifacts visible and easy to clean without sharing Humming's cache.
    path = Path(__file__).resolve().parents[3] / ".chord_tmp"
    path.mkdir(exist_ok=True, parents=True)
    return path.as_posix()


@functools.lru_cache(maxsize=1)
def get_chord_cache_dir() -> str:
    """Return the package-local JIT cache directory for indexed W4A16."""
    configured = os.getenv("CHORD_CACHE_DIR")
    if configured is not None:
        return configured
    return (Path(__file__).resolve().parents[3] / ".chord_cache").as_posix()


@functools.lru_cache(maxsize=1)
def get_chord_lock_dir() -> str:
    path = Path(get_chord_tmp_dir()) / "lock"
    path.mkdir(exist_ok=True, parents=True)
    return path.as_posix()


def hash_path_content(path: str, relative: bool = False) -> str:
    data = {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if os.path.isfile(path):
        filename = os.path.basename(path) if relative else path
        with open(path, "rb") as source:
            data[filename] = str(source.read())
    else:
        pattern = os.path.join(path, "**/*")
        for filename in sorted(glob.glob(pattern, recursive=True)):
            if not os.path.isfile(filename):
                continue
            key = os.path.relpath(filename, path) if relative else filename
            try:
                with open(filename) as source:
                    data[key] = source.read()
            except UnicodeDecodeError:
                continue
    return hash_to_hex(str(data))


@functools.lru_cache(maxsize=128)
def get_chord_lock_filename(name: str) -> str:
    filename = name if name.endswith(".lock") else name + ".lock"
    return (Path(get_chord_lock_dir()) / filename).as_posix()
