"""Humming-name aliases for the chord layer/method surface.

The classes below are the same objects as their ``chord_kernels.operator``
counterparts, re-exported under the names vLLM's Humming integration imports
(``from humming.layer import HummingLayer, HummingLayerMethod, ...``).  The
delegation contract — ``humming_metas`` on the module, classmethod
``prepare_layer_meta`` / ``transform_humming_layer`` / ``forward_layer`` /
``get_default_tuning_configs``, the ``may_*`` input helpers — matches the
upstream call shapes, so no adapter code changes when the import root flips
from ``humming`` to ``chord``.
"""

from __future__ import annotations

import torch

from chord_kernels.operator.layer import (
    IndexedW4A16Layer,
    IndexedW4A16Method,
    unpack_packed_uint4,
)
from chord_kernels.operator.profiles import (
    IndexedLayerMeta,
    IndexedLayerProfile,
    select_indexed_profile,
)

# vLLM's Humming adapter imports these exact names.
HummingLayerMeta = IndexedLayerMeta
HummingLayerMethod = IndexedW4A16Method
HummingMethod = IndexedW4A16Method
HummingLayer = IndexedW4A16Layer
HummingModule = torch.nn.Module


def get_default_f16_torch_dtype() -> torch.dtype:
    """The half-precision dtype this operator packs against: always BF16.

    Convenience for adapters that ask the backend rather than hardcoding a dtype.
    Upstream Humming has no function of this name — it resolves the f16 dtype
    inline in ``prepare_layer_config`` from ``torch.get_default_dtype()`` and the
    device capability, since it supports FP16 and BF16; chord's extracted W4A16
    path is BF16-only, so the answer is constant.
    """
    return torch.bfloat16


__all__ = [
    "HummingLayer",
    "HummingLayerMeta",
    "HummingLayerMethod",
    "HummingMethod",
    "HummingModule",
    "IndexedLayerMeta",
    "IndexedLayerProfile",
    "IndexedW4A16Layer",
    "IndexedW4A16Method",
    "get_default_f16_torch_dtype",
    "select_indexed_profile",
    "unpack_packed_uint4",
]
