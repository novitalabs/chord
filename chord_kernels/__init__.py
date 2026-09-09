"""Chord indexed W4A16 operator and runtime backend."""

from importlib import import_module
from typing import Any


__all__ = ["contiguous", "indexed", "masked", "operator"]

_OPERATOR_NAMES = frozenset(("indexed", "masked", "contiguous"))


def __getattr__(name: str) -> Any:
    if name in _OPERATOR_NAMES:
        value = getattr(import_module(f"{__name__}.ops"), name)
        globals()[name] = value
        return value
    if name == "operator":
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
