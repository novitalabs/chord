# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

"""Process-level integration switches for the chord operator.

Serving frameworks (vLLM/SGLang) launch prefill and decode instances from the
same code path, so per-process kernel/layout decisions come from environment
variables read at **weight-loading (pack) time**.  Each variable is
process-level: read once when weights are packed, applied to every chord layer
in the process.  Set them before model load — changing one afterwards does not
repack existing weights.

==========================  =======================================
chord variable              effect
==========================  =======================================
``CHORD_SM90_DECODE``       SM90 instance role: 1 = decode (D),
                            0/unset = prefill (P).
``CHORD_USE_GROUPED``       Master switch for the grouped SM90
                            W4A16 backend (DeepGEMM-derived).
==========================  =======================================

The two switches combine into one per-layer backend:

====================  ====================  =================================
``CHORD_USE_GROUPED``  ``CHORD_SM90_DECODE``  backend
====================  ====================  =================================
``0`` (unset)          any                    ``indexed``
``1``                  ``1``                  ``grouped_masked``     (BK128)
``1``                  ``0``/unset            ``grouped_contiguous`` (BK64)
====================  ====================  =================================

Two variables rather than one because the weight layout is frozen at pack time
while the two grouped kernels need DIFFERENT reorder widths (BLOCK_K 128 vs
64), and the gemm type is not known until forward: the role bit is what maps
an instance to its kernel, and the master switch is the on/off gate.

Neither variable applies to a TP8 layer.  TP8 is a single-instance ``mix`` role
serving both phases from one packed weight, so there is no P/D bit to read, and
each grouped kernel is tied to one phase, so no grouped layout can serve it.
"""

from __future__ import annotations

import os
from typing import Literal

# A profile's serving role.  ``prefill``/``decode`` are the disaggregated roles,
# each packing its own weight layout; ``mix`` is a single instance serving both
# phases from one packed weight, so its schedule must hold across the whole
# routed-M range instead of one phase's bracket.
IndexedMode = Literal["prefill", "decode", "mix"]

# P/D role bit for SM90 profile selection (default OFF -> the prefill/WGMMA
# path).  The variable only fills in the role when neither an explicit profile
# name nor a mode/role argument decides it, and only on SM90 — Blackwell
# publishes a decode profile only, so there is no role to select there.
_SM90_DECODE_ENV = "CHORD_SM90_DECODE"

# Master switch for the grouped SM90 W4A16 backend.  Only when it is set does a
# W4A16 layer pack the grouped weight layout and dispatch the grouped kernels;
# default OFF keeps the indexed path.
_USE_GROUPED_ENV = "CHORD_USE_GROUPED"


def _read_switch(name: str) -> tuple[str, str] | None:
    """Return ``(variable_name, stripped_value)`` or ``None`` if unset/empty."""
    value = os.getenv(name)
    if value is not None and value.strip():
        return name, value.strip()
    return None


def indexed_mode_from_env() -> IndexedMode | None:
    """Read the SM90 P/D role bit from ``CHORD_SM90_DECODE``.

    Returns ``"decode"`` for ``1``, ``"prefill"`` for ``0``, and ``None`` when
    the variable is unset or empty (callers then apply the default of
    prefill).  Any other value is rejected instead of being silently treated
    as one of the roles.
    """
    read = _read_switch(_SM90_DECODE_ENV)
    if read is None:
        return None
    source, value = read
    if value == "1":
        return "decode"
    if value == "0":
        return "prefill"
    raise ValueError(
        f"{source} must be '1' (decode) or '0' (prefill), got {value!r}"
    )


def use_grouped_from_env() -> bool:
    """Read the grouped-backend master switch from ``CHORD_USE_GROUPED``.

    Unset/empty means off; any value other than ``0``/``1`` is rejected.
    """
    read = _read_switch(_USE_GROUPED_ENV)
    if read is None:
        return False
    source, value = read
    if value in ("0", "1"):
        return value == "1"
    raise ValueError(f"{source} must be '0' or '1', got {value!r}")


def resolve_backend_name(
    role: IndexedMode,
    *,
    use_grouped: bool | None = None,
) -> str:
    """Map an SM90 instance role to its per-layer backend name.

    This is chord's single backend-selection policy: the master switch picks the
    family and the role picks the kernel/layout within it.  ``use_grouped=None``
    reads the environment; passing a bool makes the policy testable without
    touching the process environment.

    Upstream Humming has no equivalent function.  It derives the MMA type from
    dtype and SM version and takes the GEMM family as a per-call ``GemmType``
    argument, so it never needs to map a process to a backend; chord does,
    because its weight layout is frozen when the profile is chosen at pack time.

    ``mix`` (the single-instance TP8 role) always resolves to ``indexed``: the
    two grouped kernels are each tied to one phase, so there is no grouped
    weight layout that serves both from one packed buffer, and
    ``CHORD_USE_GROUPED`` therefore does not apply to a mix layer.
    """
    if role not in ("prefill", "decode", "mix"):
        raise ValueError(
            f"role must be 'prefill', 'decode', or 'mix', got {role!r}"
        )
    if role == "mix":
        return "indexed"
    if use_grouped is None:
        use_grouped = use_grouped_from_env()
    if use_grouped:
        return "grouped_masked" if role == "decode" else "grouped_contiguous"
    return "indexed"


__all__ = [
    "IndexedMode",
    "indexed_mode_from_env",
    "resolve_backend_name",
    "use_grouped_from_env",
]
