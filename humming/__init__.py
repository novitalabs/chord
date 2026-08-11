"""``humming`` import root for frameworks that hardcode the upstream name.

vLLM's lazy facade (``vllm/utils/humming.py``) resolves fixed module paths —
``humming.dtypes``, ``humming.config``, ``humming.layer``, ``humming.schema``,
``humming.utils.weight`` — and gates on ``find_spec("humming")``.  Installing
``chord_kernels`` ships this package so those paths resolve with no framework
change; the implementation defers to the ``chord`` facade over
``chord_kernels.operator``.  Do not install upstream ``inclusionAI/humming``
alongside it — the two provide the same import name by design.

Scope is the indexed W4A16 MoE contract only; schemas and ops outside it raise
``NotImplementedError`` rather than approximating behaviour.  Importing this
package has no CUDA side effects; JIT compilation starts at first use.
"""

__all__ = [
    "config",
    "dtypes",
    "layer",
    "schema",
    "utils",
]
