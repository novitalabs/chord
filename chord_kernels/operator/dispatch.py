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

All three branches are live, and this module is the single dispatch shared by
both the layer forward and the framework-method forward.  The grouped branches
additionally validate the requested ``gemm_type`` against the mode baked into
the packed buffer, and they reject the indexed tuning knobs (``tuning_config``,
``block_m``) because the grouped SM90 heuristic owns tile selection per call.
"""

from __future__ import annotations

import dataclasses

import torch

from chord_kernels.operator.api import IndexedKernelConfig, w4a16_indexed
from chord_kernels.operator.env import IndexedMode, resolve_backend_name
from chord_kernels.operator.grouped.api import w4a16_contiguous, w4a16_masked
from chord_kernels.operator.grouped.packing import (
    GroupedPreparedWeight,
    pack_w4a16_grouped,
)
from chord_kernels.operator.packing import PreparedWeight, WeightLayout, pack_w4a16
from chord_kernels.operator.profiles import IndexedLayerMeta
from chord_kernels.operator.tuning import (
    _resolve_tuning_config,
    _validate_grouped_compute_config,
    _validate_indexed_compute_config,
)

_GROUPED_BACKENDS = ("grouped_contiguous", "grouped_masked")


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
        # Both grouped profiles are SM90 EP deployments, one per phase, so the
        # role alone names them.
        return f"h200_grouped_{role}"
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
        return pack_w4a16_grouped(
            weight.contiguous(),
            weight_scale.contiguous(),
            mode=meta.profile.grouped_mode,
            packed=packed,
        )
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
        # The grouped buffer keeps the checkpoint's [G, N, ...] orientation and
        # holds two 4-bit codes per byte; the scale is MN-major like indexed.
        return (
            (meta.num_experts, meta.shape_n, meta.shape_k // 2),
            (meta.num_experts, meta.shape_k // 32, meta.shape_n),
        )
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
        return _forward_grouped(
            prepared,
            meta,
            inputs,
            outputs=outputs,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            m_indices=m_indices,
            valid_shape_m=valid_shape_m,
            compute_config=compute_config,
            tuning_config=tuning_config,
            block_m=block_m,
        )
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


def _forward_grouped(
    prepared: object,
    meta: IndexedLayerMeta,
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None,
    sorted_ids: torch.Tensor | None,
    expert_ids: torch.Tensor | None,
    num_tokens_padded: torch.Tensor | None,
    expert_layout: torch.Tensor | None,
    m_indices: torch.Tensor | None,
    valid_shape_m: int,
    compute_config: object | None,
    tuning_config: object | None,
    block_m: int | None,
) -> torch.Tensor:
    """Dispatch grouped-packed buffers to their masked/contiguous kernels.

    The packed buffer's mode must agree with the requested ``gemm_type`` (the
    reorder perm width is baked into the weight, so a mismatch is a silent
    wrong answer), and the scheduling tiles are owned by the grouped SM90
    heuristic rather than the indexed tuning rows.
    """
    if not isinstance(prepared, GroupedPreparedWeight):
        raise TypeError(
            f"backend {meta.backend!r} requires a GroupedPreparedWeight, got "
            f"{type(prepared).__name__}"
        )
    if sorted_ids is not None or expert_ids is not None or num_tokens_padded is not None:
        raise ValueError(
            "the grouped backend consumes expert_layout/m_indices routing, "
            "not sorted_ids/expert_ids/num_tokens_padded"
        )
    if tuning_config is not None:
        raise ValueError(
            "the grouped backend has no indexed tuning rows; omit tuning_config"
        )
    if block_m is not None:
        raise ValueError("the grouped heuristic owns the tile shape; omit block_m")
    if (
        isinstance(valid_shape_m, bool)
        or not isinstance(valid_shape_m, int)
        or valid_shape_m < 0
    ):
        raise ValueError(
            f"valid_shape_m must be a non-negative int, got {valid_shape_m!r}"
        )
    _validate_grouped_compute_config(compute_config, prepared.mode)

    if prepared.mode == "masked":
        if expert_layout is None:
            raise ValueError(
                "grouped masked decode requires expert_layout (per-expert "
                "valid token counts, [G] int32)"
            )
        if m_indices is not None:
            raise ValueError("masked decode does not take m_indices")
        if valid_shape_m <= 0:
            raise ValueError(
                "grouped masked decode requires valid_shape_m > 0 "
                "(it feeds the launch heuristic's expected_m)"
            )
        # Follow the upstream layer: expected_m is the average tokens per
        # expert, which the masked heuristic turns into the BLOCK_M tile.
        expected_m = max(1, valid_shape_m // meta.num_experts)
        return w4a16_masked(
            inputs,
            prepared,
            expert_layout,
            expected_m,
            outputs=outputs,
        )

    if expert_layout is not None:
        raise ValueError("contiguous prefill takes m_indices, not expert_layout")
    if m_indices is None:
        raise ValueError(
            "grouped contiguous prefill requires m_indices ([m] int32; "
            "-1 marks the 128-row padding rows)"
        )
    return w4a16_contiguous(
        inputs,
        prepared,
        m_indices,
        outputs=outputs,
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
        return GroupedPreparedWeight(
            packed=packed,
            scale=scale,
            mode=meta.profile.grouped_mode,
            n=meta.shape_n,
            k=meta.shape_k,
            num_experts=meta.num_experts,
        )
    raise ValueError(f"unknown W4A16 backend {meta.backend!r}")
