"""Humming-compatible facade over ``chord_kernels``.

``chord`` mirrors the module layout and leaf names of the ``humming`` package
(`inclusionAI/humming`) for the **W4A16 group-scale MoE**
surface, so a serving framework whose Humming integration spells

.. code-block:: python

    from humming.layer import HummingLayerMethod, HummingLayerMeta
    import humming.ops

can switch kernel providers by import root alone::

    import chord as humming              # or: pip-level alias / sys.modules shim
    from chord.layer import HummingLayerMethod, HummingLayerMeta

Scope is deliberately exact: chord ships the extracted W4A16 MoE operator
(BF16 activation, INT4 weight, group-32 scale) with the indexed and — once
registered — grouped SM90 backends.  Humming APIs outside that scope (other
dtypes, dense GEMM schemas, quantization ops, Hadamard) are either absent or
raise with actionable messages rather than approximating behaviour.

Runtime policy is selected by ``CHORD_SM90_DECODE`` / ``CHORD_USE_GROUPED``
(read at weight-pack time); see ``chord.config`` for the backend mapping.

Where chord adds rather than mirrors: upstream computes a launch schedule per
call from device heuristics, while chord selects a named *profile* at pack time
from a published tuning table.  A profile therefore has to name things upstream
never does — the serving role (including ``mix``, for a single instance serving
both phases) and the 8-way shard axis (EP8 vs TP8).  Adapters need not supply
either: both are recovered from the ``shape_n``/``shape_k`` and environment an
upstream-shaped adapter already provides.  ``chord.config`` is likewise chord's
own policy surface, not an upstream API; its docstring says so.
"""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("chord_kernels")
except Exception:  # pragma: no cover - source tree without installed dist
    __version__ = "0.0.0+unknown"

import chord.dtypes  # noqa: F401,E402
import chord.ops  # noqa: F401,E402

__all__ = ["__version__", "config", "dtypes", "layer", "ops"]


def __getattr__(name: str):
    if name in ("layer", "config"):
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
