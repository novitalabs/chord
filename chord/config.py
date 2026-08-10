"""Humming-name aliases for the backend-selection policy.

Upstream Humming exposes the SM90 W4A16 backend policy as
``humming.config.sm90_w4a16_decode_backend`` with the string constants below.
chord's policy lives in :mod:`chord_kernels.operator.env`; this module maps
the names so framework code keyed on the upstream constants keeps working.

Constant mapping (upstream -> chord backend name):

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

_BACKEND_TO_UPSTREAM = {
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
    """Upstream spelling of the SM90 P/D role bit."""
    return indexed_mode_from_env() == "decode"


def use_deepgemm() -> bool:
    """Upstream spelling of the grouped-backend master switch."""
    return use_grouped_from_env()


def sm90_w4a16_decode_backend(
    num_experts: int = 0,
    a_num_bits: int = 16,
    b_num_bits: int = 4,
    weight_scale_group_size: int = 32,
    use_fused_e8m0_scale: bool = False,
) -> str:
    """Upstream-shaped policy function returning the upstream constants.

    chord ships only the W4A16 group-scale MoE operator, so a non-W4A16
    signature falls back to ``wgmma`` exactly like upstream.  Within scope the
    result comes from :func:`resolve_backend_name`; the humming-native
    swap-AB/WGMMA split is a profile detail inside the indexed backend, so
    both indexed roles report their upstream constant by role.
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
    if backend in _BACKEND_TO_UPSTREAM:
        return _BACKEND_TO_UPSTREAM[backend]
    # Indexed backend: report upstream's own split.  Its swap-AB decode path
    # applies only to EP8-class expert counts; below the threshold upstream
    # stays on WGMMA even for decode.
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
