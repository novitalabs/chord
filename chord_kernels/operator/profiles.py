# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

"""Published serving profiles and per-layer metadata.

A profile is a *load-time contract*: it fixes the physical weight layout, the
kernel backend that consumes it, and the launch-schedule family for one
serving regime.  Changing a profile changes the packed bytes, so profiles are
selected before the weight is packed and never switch at runtime.

A profile also fixes the *shard axis*: the EP8 profiles split the expert list
and leave each expert's N/K whole, while ``h200_tp8`` keeps every expert on
every rank and slices ``moe_intermediate``.  That changes the per-expert GEMM
shape, not just the schedule, so the axis is part of the contract and — being
undiscoverable from the device — is never guessed by ``profile='auto'``.

``backend`` names the physical kernel family behind the profile:

* ``indexed`` — the humming-native indexed kernel (MMA/WGMMA int32 layout).
* ``grouped_masked`` / ``grouped_contiguous`` — the grouped SM90 W4A16 kernels
  (DeepGEMM-derived, decode/prefill), published as ``h200_grouped_decode`` and
  ``h200_grouped_prefill``.  The selection policy
  (:func:`chord_kernels.operator.env.resolve_backend_name`) routes to these
  names when ``CHORD_USE_GROUPED`` is set.  Unlike the indexed profiles they
  publish no tuning table: tile selection belongs to the grouped heuristic at
  dispatch time, so ``block_m`` here is metadata only.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Literal

import torch

from chord_kernels.operator import dtypes
from chord_kernels.operator.api import IndexedKernelConfig, _compatible_block_n
from chord_kernels.operator.env import IndexedMode, indexed_mode_from_env
from chord_kernels.operator.packing import (
    SUPPORTED_INDEXED_COMPUTE_CAPABILITIES,
    WeightLayout,
)

# Physical weight-buffer backends behind a profile.  Like ``layout``, this is
# a load-time contract: the backends pack mutually incompatible weight buffers.
IndexedBackend = Literal["indexed", "grouped_masked", "grouped_contiguous"]

_KNOWN_BACKENDS = ("indexed", "grouped_masked", "grouped_contiguous")


@dataclasses.dataclass(frozen=True)
class IndexedLayerProfile:
    """A fixed weight layout and launch schedule for one serving regime."""

    name: str
    device_major: int
    mode: IndexedMode
    expert_parallel_size: int
    layout: WeightLayout
    swap_ab: bool
    block_m: int
    compute_capabilities: tuple[tuple[int, int], ...]
    backend: IndexedBackend = "indexed"
    # Which axis the 8-way shard runs along, since that changes the per-expert
    # GEMM shape and not just the schedule: EP8 splits the expert list and leaves
    # N/K whole, TP8 slices ``moe_intermediate``.  Exactly one size is 8.
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("profile name must be a non-empty string")
        if (
            isinstance(self.device_major, bool)
            or not isinstance(self.device_major, int)
            or self.device_major not in (9, 10)
        ):
            raise ValueError(
                "indexed W4A16 profiles must target SM90 or SM100/SM103, got "
                f"device_major={self.device_major!r}"
            )
        capabilities = self.compute_capabilities
        if not isinstance(capabilities, tuple) or not capabilities or any(
            not isinstance(capability, tuple)
            or len(capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in capability
            )
            or capability[0] != self.device_major
            or capability not in SUPPORTED_INDEXED_COMPUTE_CAPABILITIES
            for capability in capabilities
        ):
            raise ValueError(
                "profile compute_capabilities must contain supported (major, minor) "
                f"pairs matching device_major={self.device_major}, got {capabilities!r}"
            )
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("profile compute_capabilities must not contain duplicates")
        if self.mode not in ("prefill", "decode", "mix"):
            raise ValueError(
                "profile mode must be 'prefill', 'decode', or 'mix', got "
                f"{self.mode!r}"
            )
        if self.backend not in _KNOWN_BACKENDS:
            raise ValueError(
                f"profile backend must be one of {_KNOWN_BACKENDS}, got "
                f"{self.backend!r}"
            )
        allowed_ep = (1, 8) if self.backend == "indexed" else (8, 16, 32)
        if (
            isinstance(self.expert_parallel_size, bool)
            or not isinstance(self.expert_parallel_size, int)
            or self.expert_parallel_size not in allowed_ep
        ):
            raise ValueError(
                "indexed W4A16 profiles are published only for 8-way sharding "
                "(expert_parallel_size=8, or 1 on the TP8 profile); "
                "grouped-backend profiles cover expert_parallel_size 8, 16, and "
                f"32, got {self.expert_parallel_size!r}"
            )
        if (
            isinstance(self.tensor_parallel_size, bool)
            or not isinstance(self.tensor_parallel_size, int)
            or self.tensor_parallel_size not in (1, 8)
        ):
            raise ValueError(
                "profile tensor_parallel_size must be 1 or 8, got "
                f"{self.tensor_parallel_size!r}"
            )
        # Rejecting the other combinations keeps "which axis" unambiguous, so the
        # shard degree cannot silently disagree with the shape table.  The
        # grouped backends are EP-only: they take the group count per call and
        # publish no TP-sharded shape table.
        if self.backend == "indexed":
            valid_shard = (self.expert_parallel_size, self.tensor_parallel_size) in (
                (8, 1),
                (1, 8),
            )
        else:
            valid_shard = self.tensor_parallel_size == 1
        if not valid_shard:
            raise ValueError(
                "indexed W4A16 profiles are published for 8-way sharding along "
                "exactly one axis (expert_parallel_size=8 with "
                "tensor_parallel_size=1, or the reverse) and grouped-backend "
                "profiles for expert parallelism only, got backend="
                f"{self.backend!r} "
                f"expert_parallel_size={self.expert_parallel_size}, "
                f"tensor_parallel_size={self.tensor_parallel_size}"
            )
        if self.layout not in ("mma", "wgmma"):
            raise ValueError(
                f"profile layout must be 'mma' or 'wgmma', got {self.layout!r}"
            )
        if self.layout == "wgmma" and self.swap_ab:
            raise ValueError("WGMMA profiles cannot use swap_ab")
        if self.backend != "indexed" and (self.layout != "wgmma" or self.swap_ab):
            # Mirroring upstream Humming's DeepGEMM integration: the meta still
            # reports WGMMA (no humming kernel runs), and swap-AB never applies.
            raise ValueError(
                "grouped-backend profiles must keep layout='wgmma' and swap_ab=False"
            )
        if self.backend != "indexed" and self.mode != (
            "decode" if self.backend == "grouped_masked" else "prefill"
        ):
            raise ValueError(
                "grouped_masked profiles are decode-only and grouped_contiguous "
                f"profiles are prefill-only, got backend={self.backend!r} "
                f"mode={self.mode!r}"
            )
        if (
            isinstance(self.block_m, bool)
            or not isinstance(self.block_m, int)
            or self.block_m <= 0
            or self.block_m % 8
        ):
            raise ValueError(
                f"profile block_m must be a positive multiple of 8, got {self.block_m!r}"
            )

    @property
    def compute_capability(self) -> tuple[int, int]:
        return self.compute_capabilities[0]

    @property
    def role(self) -> IndexedMode:
        return self.mode

    @property
    def shard_axis(self) -> str:
        """Which axis this profile shards over 8 GPUs, ``"ep"`` or ``"tp"``."""
        return "ep" if self.expert_parallel_size == 8 else "tp"

    @property
    def is_grouped(self) -> bool:
        return self.backend != "indexed"

    @property
    def grouped_mode(self) -> str | None:
        """The grouped kernel/layout name, or ``None`` on the indexed backend.

        The packing and dispatch layers key on the short spelling
        (``masked``/``contiguous``), since the reorder width is what the packed
        buffer actually carries.
        """
        return {
            "grouped_masked": "masked",
            "grouped_contiguous": "contiguous",
        }.get(self.backend)

    def validate_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError(
                f"profile {self.name!r} requires a CUDA device, got {device}"
            )
        capability = torch.cuda.get_device_capability(device)
        if capability not in self.compute_capabilities:
            gpu = (
                "Hopper SM90"
                if self.device_major == 9
                else "Blackwell SM100/SM103"
            )
            supported = ", ".join(
                f"{major}.{minor}"
                for major, minor in self.compute_capabilities
            )
            raise RuntimeError(
                f"profile {self.name!r} requires {gpu} compute capability "
                f"{supported}, got {capability[0]}.{capability[1]} on {device}"
            )


@dataclasses.dataclass(frozen=True)
class IndexedLayerMeta:
    """Small, serializable metadata object used by framework adapters.

    A framework keeps this object on ``layer.humming_metas`` and reads its
    logical dimensions and tensor-name properties while preparing a layer.  The
    operator has one fixed BF16/INT4/group-32 schema, so the fields below carry
    only what indexed MoE integration needs.
    """

    shape_n: int
    shape_k: int
    num_experts: int
    profile: IndexedLayerProfile
    sublayer_name: str = ""
    pad_shape_n: int = 0
    pad_shape_k: int = 0
    a_dtype: Any = dtypes.bfloat16
    b_dtype: Any = dtypes.uint4
    c_dtype: Any = dtypes.bfloat16
    bs_dtype: Any = dtypes.bfloat16
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 32
    weight_scale_group_size_n: int = 1
    weight_scale_type: str = "group"
    has_bias: bool = False
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    use_int_weight_scale: bool = False
    use_fused_e8m0_scale: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("shape_n", self.shape_n),
            ("shape_k", self.shape_k),
            ("num_experts", self.num_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        if self.shape_n <= 0 or self.shape_k <= 0 or self.num_experts <= 0:
            raise ValueError("shape_n, shape_k, and num_experts must be positive")
        if self.pad_shape_n < 0 or self.pad_shape_k < 0:
            raise ValueError("pad_shape_n and pad_shape_k must be non-negative")

    @property
    def name_prefix(self) -> str:
        return f"{self.sublayer_name}_" if self.sublayer_name else ""

    @property
    def weight_name(self) -> str:
        return self.name_prefix + "weight"

    @property
    def weight_scale_name(self) -> str:
        return self.name_prefix + "weight_scale"

    @property
    def zero_point_name(self) -> str:
        return self.name_prefix + "zero_point"

    @property
    def global_scale_name(self) -> str:
        return self.name_prefix + "global_scale"

    @property
    def bias_name(self) -> str:
        return self.name_prefix + "bias"

    @property
    def param_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def layout(self) -> WeightLayout:
        return self.profile.layout

    @property
    def backend(self) -> IndexedBackend:
        return self.profile.backend

    @property
    def swap_ab(self) -> bool:
        return self.profile.swap_ab

    @property
    def block_m(self) -> int:
        return self.profile.block_m

    @property
    def block_shape(self) -> tuple[int, int, int]:
        """The profile's fallback ``(M, N, K)`` tile.

        Consistent with :attr:`block_m` and :attr:`swap_ab`: this is the
        conservative profile tile, not a routed-M tuning row.  Use
        :meth:`kernel_config` with a real routed-M for the launch schedule.
        """
        return (
            self.profile.block_m,
            _compatible_block_n(self.shape_n),
            64,
        )

    def kernel_config(self, valid_shape_m: int = 0) -> IndexedKernelConfig:
        if self.profile.is_grouped:
            # The grouped dispatch runs its own SM90 heuristic per call; there
            # is no per-M tuning table to resolve on this layer contract.
            raise ValueError(
                "grouped-backend metas do not publish indexed kernel configs"
            )
        from chord_kernels.operator.tuning import _select_indexed_kernel_config

        return _select_indexed_kernel_config(self, valid_shape_m)

    def get_tensors_attrs(self) -> dict[str, dict[str, object]]:
        """Checkpoint tensor shapes/dtypes an adapter must allocate.

        Shaped like upstream Humming's ``weight_schema.get_tensors_attrs()`` so
        framework code can create parameters and set the loader hints
        (``input_dim``/``output_dim``/``packed_factor``) without knowing chord's
        internals.  These are the **checkpoint** shapes, pre-``transform``: the
        expert dim leads, weights are INT32-packed along K, and scales are
        group-32 along K.  Only the two tensors the W4A16 operator consumes are
        reported; bias and zero-point are out of scope for this operator and are
        rejected at :meth:`prepare_layer_meta` time instead of appearing here.
        """
        pack_factor = 32 // self.b_dtype.num_bits
        group_size = self.weight_scale_group_size or self.shape_k
        return {
            "weight": {
                "shape": (self.num_experts, self.shape_n, self.shape_k // pack_factor),
                "dtype": torch.int32,
                "extra_attrs": {
                    "input_dim": 2,
                    "output_dim": 1,
                    "packed_factor": pack_factor,
                    "packed_dim": 2,
                },
            },
            "weight_scale": {
                "shape": (self.num_experts, self.shape_n, self.shape_k // group_size),
                "dtype": self.param_dtype,
                "extra_attrs": {
                    "input_dim": 2,
                    "output_dim": 1,
                    "scale_type": "group",
                },
            },
        }

    @property
    def weight_shape(self) -> tuple[int, ...]:
        """Checkpoint shape of the packed weight (convenience over attrs)."""
        return self.get_tensors_attrs()["weight"]["shape"]  # type: ignore[return-value]

    @property
    def weight_scale_shape(self) -> tuple[int, ...]:
        """Checkpoint shape of the group scale (convenience over attrs)."""
        return self.get_tensors_attrs()["weight_scale"]["shape"]  # type: ignore[return-value]

    @property
    def mma_type(self) -> str:
        return self.profile.layout

    @property
    def compute_capability(self) -> tuple[int, int]:
        return self.profile.compute_capability

    @property
    def compute_capabilities(self) -> tuple[tuple[int, int], ...]:
        return self.profile.compute_capabilities

    @property
    def role(self) -> IndexedMode:
        return self.profile.mode

    @property
    def weight_nbytes(self) -> int:
        # INT4 weights plus BF16 group-32 scales, across all local experts.
        # This follows the upstream adapter's weight_nbytes contract and is used
        # by framework memory estimators when sizing an expert parameter shard.
        per_expert = self.shape_n * self.shape_k // 2
        per_expert += self.shape_n * (self.shape_k // 32) * 2
        return per_expert * self.num_experts

    def estimate_bound_min_shape_m(self, use_f16_accum: bool = False) -> int:
        del use_f16_accum
        return 0

    def to_str(self) -> str:
        """Return the JSON form expected by Humming-style kernel adapters."""
        return json.dumps(
            {
                "shape_n": self.shape_n,
                "shape_k": self.shape_k,
                "pad_shape_n": self.pad_shape_n,
                "pad_shape_k": self.pad_shape_k,
                "num_experts": self.num_experts,
                "a_dtype": str(self.a_dtype),
                "b_dtype": str(self.b_dtype),
                "c_dtype": str(self.c_dtype),
                "bs_dtype": str(self.bs_dtype),
                "input_scale_group_size": self.input_scale_group_size,
                "weight_scale_group_size": self.weight_scale_group_size,
                "weight_scale_group_size_n": self.weight_scale_group_size_n,
                "weight_scale_type": self.weight_scale_type,
                "has_bias": self.has_bias,
                "has_zero_point": self.has_zero_point,
                "is_fp_zero_point": self.is_fp_zero_point,
                "sublayer_name": self.sublayer_name,
            }
        )


# Names are intentionally explicit: changing a profile changes the physical
# weight layout, so an implicit architecture guess would be unsafe at load time.
_PROFILES: dict[str, IndexedLayerProfile] = {
    "h200_prefill_ep8": IndexedLayerProfile(
        name="h200_prefill_ep8",
        device_major=9,
        mode="prefill",
        expert_parallel_size=8,
        layout="wgmma",
        swap_ab=False,
        block_m=16,
        compute_capabilities=((9, 0),),
    ),
    # TP8 targets single-instance ("mix") deployment, so it carries no P/D role.
    "h200_tp8": IndexedLayerProfile(
        name="h200_tp8",
        device_major=9,
        mode="mix",
        expert_parallel_size=1,
        tensor_parallel_size=8,
        layout="wgmma",
        swap_ab=False,
        block_m=16,
        compute_capabilities=((9, 0),),
    ),
    "h200_decode_ep8": IndexedLayerProfile(
        name="h200_decode_ep8",
        device_major=9,
        mode="decode",
        expert_parallel_size=8,
        layout="mma",
        swap_ab=True,
        block_m=8,
        compute_capabilities=((9, 0),),
    ),
    # grouped paths (EP-width agnostic up to the pack-side checks):
    # masked decode (BLOCK_K=128 buffer) and contiguous prefill (BLOCK_K=64
    # buffer), selected explicitly or through CHORD_USE_GROUPED.
    # ``expert_parallel_size`` records the canonical value; EP16/32 shards are
    # accepted by ``select_indexed_profile``.
    "h200_grouped_prefill": IndexedLayerProfile(
        name="h200_grouped_prefill",
        device_major=9,
        mode="prefill",
        expert_parallel_size=8,
        layout="wgmma",
        swap_ab=False,
        # Metadata only: the grouped heuristic (not the profile) picks the
        # real tile from (m, n, k) at dispatch time; 128 anchors the table's
        # prefill BM=128 default documented by the upstream tuning notes.
        block_m=128,
        compute_capabilities=((9, 0),),
        backend="grouped_contiguous",
    ),
    "h200_grouped_decode": IndexedLayerProfile(
        name="h200_grouped_decode",
        device_major=9,
        mode="decode",
        expert_parallel_size=8,
        layout="wgmma",
        swap_ab=False,
        block_m=8,
        compute_capabilities=((9, 0),),
        backend="grouped_masked",
    ),
    "blackwell_decode_ep8": IndexedLayerProfile(
        name="blackwell_decode_ep8",
        device_major=10,
        mode="decode",
        expert_parallel_size=8,
        layout="mma",
        swap_ab=True,
        block_m=8,
        compute_capabilities=((10, 0), (10, 3)),
    ),
}

# Public constants make profile selection convenient in framework registries while
# keeping the actual values immutable.  The dictionary is intentionally a copy so
# callers cannot mutate the module's internal lookup table.
H200_PREFILL_EP8 = _PROFILES["h200_prefill_ep8"]
H200_TP8 = _PROFILES["h200_tp8"]
H200_DECODE_EP8 = _PROFILES["h200_decode_ep8"]
BLACKWELL_DECODE_EP8 = _PROFILES["blackwell_decode_ep8"]
H200_GROUPED_PREFILL = _PROFILES["h200_grouped_prefill"]
H200_GROUPED_DECODE = _PROFILES["h200_grouped_decode"]
INDEXED_PROFILES = dict(_PROFILES)

# Per-expert ``(shape_n, shape_k)`` projection pairs each shard axis publishes a
# tuned schedule for.  The two sets are disjoint, which is what makes the axis
# recoverable from a layer's shapes: EP8 splits the expert list and leaves each
# expert's projections whole, while TP8 slices ``moe_intermediate`` and so
# narrows gate/up's N and down's K by 8.
#
# These pairs must stay in step with the shape guards in
# :mod:`chord_kernels.operator.tuning`; ``test_shard_axis_shapes_match_tuning``
# asserts they do, so a new tuned shape cannot be added in one place only.
_EP8_SHAPES = ((4096, 7168), (7168, 2048))
_TP8_SHAPES = ((512, 7168), (7168, 256))


def shard_axis_from_shapes(shape_n: int, shape_k: int) -> str | None:
    """Recover the shard axis from one projection's per-expert shape.

    Returns ``"ep"``, ``"tp"``, or ``None`` when the pair is not one this
    operator publishes a tuned schedule for.  ``None`` means "do not guess": an
    unrecognised shape falls back to the EP8 default and the conservative
    profile tile, rather than claiming a tuning result for another model.

    This is how a framework adapter reaches TP8 without a chord-specific
    argument.  Upstream Humming has no shard-axis concept at all — it derives a
    schedule from N/K/num_experts on every call, so the axis never needs naming.
    chord replaced that with a fixed profile table, so the axis has to be
    recovered from the same shapes upstream would have keyed on.
    """
    pair = (shape_n, shape_k)
    if pair in _TP8_SHAPES:
        return "tp"
    if pair in _EP8_SHAPES:
        return "ep"
    return None


def _normalise_mode(mode: str | None) -> IndexedMode | None:
    if mode is None:
        return None
    if mode not in ("prefill", "decode", "mix"):
        raise ValueError(
            f"mode must be 'prefill', 'decode', or 'mix', got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def resolve_shard_axis(
    tensor_parallel_size: int | None,
    shape_n: int | None = None,
    shape_k: int | None = None,
) -> str:
    """Decide the shard axis from an explicit size and/or the layer's shapes.

    An explicit ``tensor_parallel_size`` states the deployment and wins, so a
    model whose shapes this operator has no tuned schedule for can still select
    TP8 (it gets the conservative profile tile).  When the shapes *are* published
    and the stated size contradicts them, that is a configuration error and is
    rejected rather than resolved: the two disagree about which table applies, so
    either answer would pack the weight against the wrong schedule.

    With no explicit size the axis is recovered from the shapes, defaulting to
    ``"ep"`` when they are unrecognised.
    """
    inferred = (
        shard_axis_from_shapes(shape_n, shape_k)
        if isinstance(shape_n, int) and isinstance(shape_k, int)
        else None
    )
    if tensor_parallel_size is None:
        return inferred or "ep"
    stated = "tp" if tensor_parallel_size == 8 else "ep"
    if inferred is not None and inferred != stated:
        published = _TP8_SHAPES if inferred == "tp" else _EP8_SHAPES
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} states the "
            f"{stated.upper()}8 axis, but (shape_n={shape_n}, shape_k={shape_k}) "
            f"is a published {inferred.upper()}8 projection "
            f"{tuple(published)}; the shard axis and the projection shapes must "
            "agree, since each axis has its own tuned schedule"
        )
    return stated


def _auto_profile_name(
    capability: tuple[int, int],
    selected_mode: IndexedMode | None,
    tensor_parallel_size: int | None = None,
    shard_axis: str | None = None,
) -> str:
    """Resolve ``profile='auto'`` for one CUDA capability.

    Kept separate from :func:`select_indexed_profile` so the SM90 policy reads
    as one block: an explicit mode wins; otherwise the ``CHORD_SM90_DECODE``
    role bit decides, defaulting to prefill when unset/0.  With
    ``CHORD_USE_GROUPED=1`` both roles reroute to the grouped backend's
    profiles.

    The shard axis is not a property of the *device*, so it is never inferred
    from ``capability``; the caller resolves it with :func:`resolve_shard_axis`
    and passes the result as ``shard_axis``.  TP8 is a single-instance ``mix``
    profile, so neither the P/D role bit nor ``CHORD_USE_GROUPED`` applies to it.
    """
    from chord_kernels.operator.dispatch import backend_profile_name

    if shard_axis == "tp":
        if capability != (9, 0):
            raise RuntimeError(
                "the TP8 W4A16 profile is published for SM90 only, got "
                f"compute capability {capability[0]}.{capability[1]}"
            )
        return "h200_tp8"
    if capability in ((10, 0), (10, 3)):
        # Blackwell publishes a decode profile only, so the SM90 role bit does
        # not apply and CHORD_SM90_DECODE is ignored here; an explicit
        # mode='prefill' is still rejected by the mode check in the caller.
        return "blackwell_decode_ep8"
    if capability == (9, 0):
        sm90_role: IndexedMode = selected_mode or indexed_mode_from_env() or "prefill"
        return backend_profile_name(sm90_role)
    raise RuntimeError(
        "no indexed W4A16 profile for compute capability "
        f"{capability[0]}.{capability[1]}"
    )


def select_indexed_profile(
    profile: str | IndexedLayerProfile | None = "auto",
    *,
    mode: IndexedMode | None = None,
    role: IndexedMode | None = None,
    device: torch.device | None = None,
    expert_parallel_size: int | None = None,
    tensor_parallel_size: int | None = None,
    shape_n: int | None = None,
    shape_k: int | None = None,
) -> IndexedLayerProfile:
    """Resolve an explicit profile or select one from device capability.

    ``auto`` is resolved only when this function is called, so layer
    construction remains safe on CPU/meta devices; a serving framework can
    construct a layer first and resolve it after moving weights to CUDA.  On
    SM90 the P/D role comes from an explicit ``mode``/``role`` argument, else
    ``CHORD_SM90_DECODE``, else the prefill default; Blackwell resolves to its
    only published profile (decode) regardless of the variable.

    The shard axis is decided separately from the P/D role, because it is a
    property of the deployment rather than the device.  ``auto`` takes it from an
    explicit ``tensor_parallel_size``, else from ``shape_n``/``shape_k`` when they
    name a published TP8 or EP8 projection (see :func:`shard_axis_from_shapes`),
    else defaults to EP8.  Passing the shapes is what lets a framework adapter
    land on TP8 without a chord-specific argument; omitting them keeps the EP8
    default.  TP8 is a single-instance ``mix`` profile, so it also ignores the
    P/D role bit.

    Both parallel sizes default to ``None`` rather than 8, so naming a profile is
    enough; a stated value is checked against the one the profile carries.
    """
    selected_mode = _normalise_mode(mode)
    if role is not None:
        role_mode = _normalise_mode(role)
        if selected_mode is not None and selected_mode != role_mode:
            raise ValueError("mode and role must agree when both are provided")
        selected_mode = role_mode

    if isinstance(profile, IndexedLayerProfile):
        resolved = profile
    else:
        if profile is not None and not isinstance(profile, str):
            raise TypeError(
                "profile must be a profile name, IndexedLayerProfile, or None; "
                f"got {type(profile).__name__}"
            )
        name = "auto" if profile is None else profile

        if name == "auto":
            if device is not None and torch.device(device).type != "cuda":
                raise RuntimeError(
                    f"profile='auto' requires a CUDA device for capability detection, got {device}"
                )
            capability = torch.cuda.get_device_capability(
                None if device is None else torch.device(device)
            )
            name = _auto_profile_name(
                capability,
                selected_mode,
                tensor_parallel_size,
                resolve_shard_axis(tensor_parallel_size, shape_n, shape_k),
            )
        try:
            resolved = _PROFILES[name]
        except KeyError as exc:
            valid = ", ".join(sorted(_PROFILES))
            raise ValueError(
                f"unknown indexed profile {profile!r}; choose one of {valid}"
            ) from exc

    if selected_mode is not None and resolved.mode != selected_mode:
        raise ValueError(
            f"profile {resolved.name!r} is for {resolved.mode}, not {selected_mode}"
        )

    allowed_ep = (8, 16, 32) if resolved.is_grouped else (1, 8)
    if expert_parallel_size is not None and (
        isinstance(expert_parallel_size, bool)
        or not isinstance(expert_parallel_size, int)
        or expert_parallel_size not in allowed_ep
    ):
        raise ValueError(
            f"profile {resolved.name!r} is published for expert_parallel_size "
            f"{allowed_ep}, got {expert_parallel_size!r}"
        )
    if tensor_parallel_size is not None and (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size not in (1, 8)
    ):
        raise ValueError(
            "tensor_parallel_size must be 8 (the TP8 profile) or 1 (the EP8 "
            f"profiles), got {tensor_parallel_size!r}"
        )
    # Grouped-backend profiles are EP-width agnostic: the kernel takes the
    # group count from the routing arguments per call, so the profile's own
    # ``expert_parallel_size`` is canonical metadata rather than a constraint.
    # The indexed profiles bake their shard-degree tile tables in, so there the
    # requested widths must match the profile exactly.
    if (
        not resolved.is_grouped
        and expert_parallel_size is not None
        and resolved.expert_parallel_size != expert_parallel_size
    ):
        raise ValueError(
            f"profile {resolved.name!r} expects expert_parallel_size="
            f"{resolved.expert_parallel_size}, got {expert_parallel_size}"
        )
    if (
        tensor_parallel_size is not None
        and resolved.tensor_parallel_size != tensor_parallel_size
    ):
        raise ValueError(
            f"profile {resolved.name!r} expects tensor_parallel_size="
            f"{resolved.tensor_parallel_size}, got {tensor_parallel_size}"
        )
    if device is not None and torch.device(device).type == "cuda":
        resolved.validate_device(torch.device(device))
    return resolved


__all__ = [
    "BLACKWELL_DECODE_EP8",
    "H200_DECODE_EP8",
    "H200_GROUPED_DECODE",
    "H200_GROUPED_PREFILL",
    "H200_PREFILL_EP8",
    "H200_TP8",
    "INDEXED_PROFILES",
    "IndexedBackend",
    "IndexedLayerMeta",
    "IndexedLayerProfile",
    "IndexedMode",
    "select_indexed_profile",
    "shard_axis_from_shapes",
]
