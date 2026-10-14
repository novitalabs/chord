"""``humming`` import root for frameworks that hardcode the upstream name.

vLLM's lazy facade (``vllm/utils/humming.py``) resolves fixed module paths and
gates on ``find_spec("humming")``.  Installing ``chord_kernels`` ships this
package so those paths resolve with no framework change: the functional
interface vLLM 0.29+ uses (``humming.transform``, ``humming.tune``,
``humming.forward`` with ``humming.config.LayerConfig``) and the earlier
layer-delegation one (``humming.layer``), plus ``humming.dtypes``,
``humming.schema`` and ``humming.utils.weight``.  The implementation defers to
``chord_kernels.operator``.  Do not install upstream ``vllm-project/humming``
alongside it — the two provide the same import name by design.

Scope is the indexed W4A16 MoE contract only; schemas and ops outside it raise
``NotImplementedError`` rather than approximating behaviour.  Importing this
package has no CUDA side effects; JIT compilation starts at first use.
"""

__all__ = [
    "config",
    "dtypes",
    "forward",
    "layer",
    "schema",
    "transform",
    "tune",
    "utils",
]
