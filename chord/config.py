"""chord's backend-selection policy, in humming-shaped spellings.

Note on provenance: unlike the rest of this facade, the names here are **not**
mirrors of an upstream API.  Upstream Humming (inclusionAI/humming at ``main``)
has no backend-selection policy function and no such constants: its
``humming.config`` package carries only ``LayerConfig``/``MmaType``/``GemmType``,
its MMA type is derived from dtype and SM version, and the GEMM family
(``dense``/``indexed``/``grouped_contiguous``/``grouped_masked``) is a per-call
``GemmType`` argument rather than a load-time property.

This module exists because chord replaced upstream's per-call device heuristics
with a fixed profile table chosen at pack time, which needs a policy to pick a
profile.  It is offered in the naming style of the rest of the facade for
consistency, and the constants below are chord's own.  Framework code should not
expect an upstream ``humming`` install to provide them.

Constant mapping (constant -> chord backend name):

* ``SM90_W4A16_DEEPGEMM_MASKED`` (``"deepgemm"``) -> ``grouped_masked``
* ``SM90_W4A16_DEEPGEMM_CONTIGUOUS`` -> ``grouped_contiguous``
* ``SM90_W4A16_WGMMA`` / ``SM90_W4A16_SWAP_AB`` -> ``indexed`` (the extracted
  operator folds humming-native WGMMA prefill and swap-AB decode into its
  profile table, so both map onto the indexed backend here).
"""

from __future__ import annotations

import os

from chord_kernels.operator.env import (
    indexed_mode_from_env,
    resolve_backend_name,
    use_grouped_from_env,
)

SM90_W4A16_DEEPGEMM_MASKED = "deepgemm"
SM90_W4A16_DEEPGEMM_CONTIGUOUS = "deepgemm_contiguous"
SM90_W4A16_SWAP_AB = "swap_ab"
SM90_W4A16_WGMMA = "wgmma"

_BACKEND_TO_CONSTANT = {
    "grouped_masked": SM90_W4A16_DEEPGEMM_MASKED,
    "grouped_contiguous": SM90_W4A16_DEEPGEMM_CONTIGUOUS,
}


def _sm90_ep8_min_experts() -> int:
    """EP8-class expert-count threshold for the swap-AB report.

    Read live from ``CHORD_SM90_EP8_MIN_EXPERTS`` (default 48: EP8 packs 48
    experts/GPU).
    """
    value = os.getenv("CHORD_SM90_EP8_MIN_EXPERTS")
    if value is not None and value.strip():
        return int(value)
    return 48


def is_sm90_decode() -> bool:
    """Whether this process is an SM90 decode instance (``CHORD_SM90_DECODE``)."""
    return indexed_mode_from_env() == "decode"


def use_deepgemm() -> bool:
    """Whether the grouped-backend master switch is on (``CHORD_USE_GROUPED``)."""
    return use_grouped_from_env()


def sm90_w4a16_decode_backend(
    num_experts: int = 0,
    a_num_bits: int = 16,
    b_num_bits: int = 4,
    weight_scale_group_size: int = 32,
    use_fused_e8m0_scale: bool = False,
) -> str:
    """Report the W4A16 backend for this process as one of the constants above.

    chord ships only the W4A16 group-scale MoE operator, so a signature outside
    that scope reports ``wgmma`` rather than guessing.  Within scope the result
    comes from :func:`resolve_backend_name`; the WGMMA/swap-AB split is a profile
    detail inside the indexed backend, so both indexed roles report a constant
    chosen by role.

    This function has no upstream counterpart (see the module docstring); it is
    chord's own policy, exposed in the facade's naming style.
    """
    is_w4a16_group = (
        a_num_bits == 16
        and b_num_bits == 4
        and weight_scale_group_size > 0
        and not use_fused_e8m0_scale
    )
    if not (is_w4a16_group and num_experts):
        return SM90_W4A16_WGMMA
    role = indexed_mode_from_env() or "prefill"
    backend = resolve_backend_name(role)
    if backend in _BACKEND_TO_CONSTANT:
        return _BACKEND_TO_CONSTANT[backend]
    # Indexed backend: report the WGMMA/swap-AB split.  The swap-AB decode path
    # applies only to EP8-class expert counts; below the threshold the WGMMA
    # profile is used even for decode.
    if role == "decode" and num_experts >= _sm90_ep8_min_experts():
        return SM90_W4A16_SWAP_AB
    return SM90_W4A16_WGMMA


__all__ = [
    "SM90_W4A16_DEEPGEMM_CONTIGUOUS",
    "SM90_W4A16_DEEPGEMM_MASKED",
    "SM90_W4A16_SWAP_AB",
    "SM90_W4A16_WGMMA",
    "is_sm90_decode",
    "sm90_w4a16_decode_backend",
    "use_deepgemm",
]
