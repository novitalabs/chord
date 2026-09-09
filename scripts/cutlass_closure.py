"""Compute the CUTLASS/CuTe include closure needed by the vendored kernels.

The CUTLASS submodule under ``third_party/cutlass`` is the source of truth for
these headers. Only the transitive ``cute/`` + ``cutlass/`` closure of the
kernel sources is packaged, so the wheel carries the arch-level headers the
NVRTC path actually compiles rather than the full 800-file upstream tree.
"""

from __future__ import annotations

import re
from pathlib import Path

# ``#include <cute/...>`` / ``#include "cutlass/..."`` for the two upstream roots.
_INCLUDE_RE = re.compile(
    r"^[ \t]*#[ \t]*include[ \t]*[<\"]((?:cute|cutlass)/[^>\"]+)[>\"]",
    re.MULTILINE,
)

ROOTS = ("cute", "cutlass")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def compute_closure(seed_dirs: list[Path], cutlass_include: Path) -> set[str]:
    """Return include-relative paths of every CUTLASS/CuTe header reachable from ``seed_dirs``."""
    pending: list[str] = []
    for seed_dir in seed_dirs:
        if not seed_dir.is_dir():
            continue
        for source in sorted(seed_dir.rglob("*")):
            if source.is_file() and source.suffix in {".cuh", ".cu", ".hpp", ".h", ".cpp"}:
                pending.extend(_INCLUDE_RE.findall(_read(source)))

    closure: set[str] = set()
    missing: set[str] = set()
    while pending:
        rel = pending.pop()
        if rel in closure or rel in missing:
            continue
        header = cutlass_include / rel
        if not header.is_file():
            # Upstream guards some includes behind arch macros that never resolve
            # for the SM90/SM100 targets this package compiles.
            missing.add(rel)
            continue
        closure.add(rel)
        pending.extend(_INCLUDE_RE.findall(_read(header)))
    return closure
