# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

"""Flat W4A16 kernel dispatch keyed on a backend tag.

chord hosts three W4A16 MoE kernel families:

* ``indexed`` — the humming-native indexed kernel (MMA/WGMMA int32 layout),
  routed by ``sorted_ids`` / ``expert_ids`` / ``num_tokens_padded``;
* ``grouped_contiguous`` — the DeepGEMM-derived SM90 prefill kernel, routed by
  a per-row ``m_indices`` map;
* ``grouped_masked`` — the DeepGEMM-derived SM90 decode kernel, routed by a
  per-expert ``expert_layout`` (valid token counts).

Following upstream humming, the three do not hide behind a registry or an
abstract base class: the family is a plain string tag carried on the profile
(:attr:`IndexedLayerProfile.backend`), and each entry point below switches on
that tag with a flat ``if``/``elif``.  The routing tensors differ per family,
so each branch consumes only its own and rejects the others loudly — a packed
weight serves exactly one family, so a mismatched call is a config error, never
a silent fallback.

The grouped branches are entry points only: the DeepGEMM kernel package is not
vendored here yet, so they raise :class:`NotImplementedError` with the tag that
must be wired in.  The indexed branch is complete and is the single dispatch
shared by both the layer forward and the framework-method forward.
"""

from __future__ import annotations

import dataclasses

import torch

from chord_kernels.operator.api import IndexedKernelConfig, w4a16_indexed
from chord_kernels.operator.env import IndexedMode, resolve_backend_name
from chord_kernels.operator.packing import PreparedWeight, WeightLayout, pack_w4a16
from chord_kernels.operator.profiles import IndexedLayerMeta
from chord_kernels.operator.tuning import (
    _resolve_tuning_config,
    _validate_indexed_compute_config,
)

_GROUPED_BACKENDS = ("grouped_contiguous", "grouped_masked")


def _grouped_not_available(backend: str) -> NotImplementedError:
    """A grouped tag selected on a build that does not vendor the kernel."""
    return NotImplementedError(
        f"W4A16 backend {backend!r} is not available in this build: the "
        "DeepGEMM-derived grouped kernel package is not vendored yet. Unset "
        "CHORD_USE_GROUPED to stay on the indexed backend."
    )


def profile_name_for_role(backend: str, role: IndexedMode) -> str:
    """Auto-profile name a backend publishes for an SM90 role.

    Only the EP8 names are reachable from a role: ``mix`` is the TP8 profile's
    role and TP8 is never auto-selected (the shard axis is not discoverable), so
    it is requested by name or via ``tensor_parallel_size=8`` instead.
    """
    if backend == "indexed":
        if role == "mix":
            return "h200_tp8"
        return f"h200_{role}_ep8"
    if backend in _GROUPED_BACKENDS:
        raise _grouped_not_available(backend)
    raise ValueError(f"unknown W4A16 backend {backend!r}")


def pack_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    meta: IndexedLayerMeta,
    packed: bool,
) -> PreparedWeight:
    """Convert checkpoint tensors into the backend's kernel layout."""
    if meta.backend == "indexed":
        return pack_w4a16(
            weight.contiguous(),
            weight_scale.contiguous(),
            layout=meta.layout,
            packed=packed,
        )
    if meta.backend in _GROUPED_BACKENDS:
        raise _grouped_not_available(meta.backend)
    raise ValueError(f"unknown W4A16 backend {meta.backend!r}")


def transformed_shapes(
    meta: IndexedLayerMeta,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(packed_shape, scale_shape)`` of the transformed buffers."""
    if meta.backend == "indexed":
        return (
            (meta.num_experts, meta.shape_k // 16, meta.shape_n * 2),
            (meta.num_experts, meta.shape_k // 32, meta.shape_n),
        )
    if meta.backend in _GROUPED_BACKENDS:
        raise _grouped_not_available(meta.backend)
    raise ValueError(f"unknown W4A16 backend {meta.backend!r}")


def forward_w4a16(
    prepared: object,
    meta: IndexedLayerMeta,
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    m_indices: torch.Tensor | None = None,
    top_k: int | None = None,
    valid_shape_m: int = 0,
    compute_config: object | None = None,
    tuning_config: object | None = None,
    block_m: int | None = None,
    validate_routing: bool = False,
) -> torch.Tensor:
    """Dispatch one W4A16 GEMM on the prepared weight's backend.

    The full Humming delegation argument set is accepted so a framework passes
    the same named arguments to every family; each branch consumes only its own
    routing tensors.
    """
    backend = meta.backend
    if backend in _GROUPED_BACKENDS:
        raise _grouped_not_available(backend)
    if backend != "indexed":
        raise ValueError(f"unknown W4A16 backend {backend!r}")

    # --- indexed family -----------------------------------------------------
    if expert_layout is not None or m_indices is not None:
        raise ValueError("indexed W4A16 uses sorted_ids/expert_ids routing only")
    if compute_config is not None:
        _validate_indexed_compute_config(compute_config)
    if (
        sorted_ids is None
        or expert_ids is None
        or num_tokens_padded is None
        or top_k is None
    ):
        raise TypeError(
            "forward requires sorted_ids, expert_ids, num_tokens_padded, and top_k"
        )
    selected_shape_m = valid_shape_m
    if selected_shape_m <= 0:
        selected_shape_m = inputs.size(0) * top_k
    kernel_config = _resolve_tuning_config(
        meta,
        selected_shape_m,
        tuning_config,
        default=meta.kernel_config(selected_shape_m),
    )
    if block_m is not None and block_m != kernel_config.block_m:
        # An explicit routing block-M (e.g. the shared w13 table's) overrides
        # the row's M tile while keeping the projection's N/K tile.
        kernel_config = _config_with_block_m(kernel_config, block_m, layout=meta.layout)
    return w4a16_indexed(
        inputs,
        prepared,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        top_k,
        outputs=outputs,
        config=kernel_config,
        layout=meta.layout,
        swap_ab=kernel_config.swap_ab,
        valid_shape_m=valid_shape_m,
        validate_routing=validate_routing,
    )


def _config_with_block_m(
    config: IndexedKernelConfig,
    block_m: int,
    *,
    layout: WeightLayout,
) -> IndexedKernelConfig:
    """Keep a projection's tile choice while matching the shared route blocks."""
    if block_m == config.block_m:
        return config
    if (
        isinstance(block_m, bool)
        or not isinstance(block_m, int)
        or block_m <= 0
        or block_m % 8
    ):
        raise ValueError(
            f"routing block_m must be a positive multiple of 8, got {block_m!r}"
        )
    swap_ab = config.swap_ab
    if layout == "mma" and block_m % 16 == 8:
        # MMA's non-swap path cannot issue an m8n8k8 BF16 instruction.  The
        # swap layout accepts these small token tiles and is compatible with
        # the same packed weight buffer.
        swap_ab = True
    return dataclasses.replace(
        config,
        block_shape=(block_m, config.block_n, config.block_k),
        warp_shape=(block_m, config.warp_shape[1], config.warp_shape[2]),
        swap_ab=swap_ab,
    )


def backend_profile_name(role: IndexedMode) -> str:
    """Auto-profile name for an SM90 role via the env-driven backend policy.

    ``select_indexed_profile('auto')`` calls this so the whole env policy lives
    in ``env.py`` (role -> backend) plus this one name mapping (backend, role
    -> profile).
    """
    return profile_name_for_role(resolve_backend_name(role), role)


__all__ = [
    "backend_profile_name",
    "build_prepared",
    "forward_w4a16",
    "pack_weight",
    "profile_name_for_role",
    "transformed_shapes",
]


def build_prepared(
    packed: torch.Tensor,
    scale: torch.Tensor,
    meta: IndexedLayerMeta,
) -> PreparedWeight:
    """Rebuild the prepared-weight view over already-transformed tensors."""
    if meta.backend == "indexed":
        return PreparedWeight(
            packed=packed,
            scale=scale,
            layout=meta.layout,
            n=meta.shape_n,
            k=meta.shape_k,
            num_experts=meta.num_experts,
        )
    if meta.backend in _GROUPED_BACKENDS:
        raise _grouped_not_available(meta.backend)
    raise ValueError(f"unknown W4A16 backend {meta.backend!r}")
