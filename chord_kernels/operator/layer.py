# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

"""W4A16 MoE layer and framework-method adapters.

The layer deliberately keeps the framework contract narrow: callers provide the
already aligned routing tensors produced by a MoE router, while this module owns
checkpoint weight loading and delegation.  Which kernel actually runs is a
*backend* decision carried on the layer's profile and dispatched flatly in
:mod:`chord_kernels.operator.dispatch`; profile and schedule policy live in
:mod:`chord_kernels.operator.profiles` / :mod:`chord_kernels.operator.tuning`,
and the process-level switches in :mod:`chord_kernels.operator.env`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from chord_kernels.operator import dispatch, dtypes
from chord_kernels.operator.api import IndexedKernelConfig, w4a16_indexed
from chord_kernels.operator.dispatch import _config_with_block_m
from chord_kernels.operator.env import (
    IndexedMode,
    indexed_mode_from_env,
    resolve_backend_name,
    use_grouped_from_env,
)
from chord_kernels.operator.packing import (
    PreparedWeight,
    WeightLayout,
    pack_w4a16,
)
from chord_kernels.operator.profiles import (
    BLACKWELL_DECODE_EP8,
    H200_DECODE_EP8,
    H200_GROUPED_DECODE,
    H200_GROUPED_PREFILL,
    H200_PREFILL_EP8,
    H200_TP8,
    INDEXED_PROFILES,
    _PROFILES,
    IndexedBackend,
    IndexedLayerMeta,
    IndexedLayerProfile,
    _normalise_mode,
    resolve_shard_axis,
    select_indexed_profile,
    shard_axis_from_shapes,
)
from chord_kernels.operator.tuning import (
    _h200_prefill_block_m,
    _h200_prefill_use_stream_k,
    _h200_tp8_block_m,
    _h200_tp8_use_stream_k,
    _indexed_tuning_rows,
    _resolve_tuning_config,
    _select_indexed_kernel_config,
    _validate_indexed_compute_config,
)

# The names imported above from profiles/tuning/backends are re-exported here
# under their historical spellings: tests and downstream adapters reach
# ``_PROFILES``, ``_h200_prefill_block_m``, ``_indexed_tuning_rows``,
# ``pack_w4a16``, ``w4a16_indexed`` etc. through ``chord_kernels.operator.layer``.


def unpack_packed_uint4(
    weight: torch.Tensor, shape_k: int | None = None
) -> torch.Tensor:
    """Unpack Humming/vLLM INT32 checkpoint words into one 4-bit code per INT32.

    Each INT32 word contains eight nibbles in little-endian bit order.  The
    returned tensor has the same leading dimensions and a final dimension eight
    times larger than the input.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if weight.dtype != torch.int32:
        raise TypeError(f"packed weight must be torch.int32, got {weight.dtype}")
    if weight.ndim < 1 or not weight.is_contiguous():
        raise ValueError(
            "packed weight must be contiguous and have at least one dimension"
        )
    if shape_k is not None:
        if isinstance(shape_k, bool) or not isinstance(shape_k, int) or shape_k <= 0:
            raise ValueError(f"shape_k must be a positive integer, got {shape_k!r}")
        if shape_k != weight.shape[-1] * 8:
            raise ValueError(
                f"packed weight has K={weight.shape[-1] * 8}, expected shape_k={shape_k}"
            )

    # Convert through int64 so right shifts of negative signed INT32 words are
    # well-defined after masking to the original unsigned 32 bits.
    words = weight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64, device=weight.device) * 4
    codes = (words.unsqueeze(-1) >> shifts) & 0xF
    return codes.reshape(*weight.shape[:-1], weight.shape[-1] * 8).to(torch.int32)


def _replace_parameter(
    module: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    parameters = getattr(module, "_parameters", None)
    old_parameter = parameters.get(name) if isinstance(parameters, dict) else None
    parameter = torch.nn.Parameter(tensor, requires_grad=False)
    # vLLM attaches loader/sharding metadata directly to Parameters. Preserve
    # that metadata when the compact checkpoint tensor is replaced by a packed
    # kernel tensor; ordinary PyTorch Parameters simply have an empty dict.
    if old_parameter is not None:
        for key, value in getattr(old_parameter, "__dict__", {}).items():
            setattr(parameter, key, value)
    if isinstance(parameters, dict) and name in parameters:
        parameters[name] = parameter
    else:
        setattr(module, name, parameter)


def _install_prepared_tensors(
    module: torch.nn.Module,
    prepared: object,
    *,
    weight_name: str = "weight",
    scale_name: str = "weight_scale",
) -> bool:
    """Replace the module's checkpoint parameters with the packed tensors.

    Backend-agnostic: any prepared-weight dataclass exposing ``packed`` and
    ``scale`` tensors is installed; anything else (a test double, a foreign
    handle) is left alone and ``False`` is returned.
    """
    packed = getattr(prepared, "packed", None)
    scale = getattr(prepared, "scale", None)
    if not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return False
    _replace_parameter(module, weight_name, packed)
    _replace_parameter(module, scale_name, scale)
    return True


class IndexedW4A16Layer(torch.nn.Module):
    """W4A16 MoE layer suitable for a vLLM weight-loader adapter.

    ``weight`` is accepted in the usual Humming checkpoint form ``[E, N, K/8]``
    (eight 4-bit codes per INT32) or as unpacked codes ``[E, N, K]``.  ``transform``
    converts it to the physical layout of the profile's backend.  Construction
    and CPU/meta checkpoint loading do not touch CUDA; the first CUDA transform
    performs the actual JIT-backed repack.
    """

    def __init__(
        self,
        shape_n: int | None = None,
        shape_k: int | None = None,
        num_experts: int = 1,
        *,
        n: int | None = None,
        k: int | None = None,
        profile: str | IndexedLayerProfile | None = "auto",
        mode: IndexedMode | None = None,
        role: IndexedMode | None = None,
        expert_parallel_size: int | None = None,
        tensor_parallel_size: int | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if shape_n is None:
            shape_n = n
        elif n is not None and n != shape_n:
            raise ValueError("shape_n and n must agree")
        if shape_k is None:
            shape_k = k
        elif k is not None and k != shape_k:
            raise ValueError("shape_k and k must agree")
        if shape_n is None or shape_k is None:
            raise TypeError("shape_n/shape_k (or n/k) are required")
        for name, value in (
            ("shape_n", shape_n),
            ("shape_k", shape_k),
            ("num_experts", num_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if shape_n % 128 != 0:
            raise ValueError(f"shape_n must be divisible by 128, got {shape_n}")
        if shape_k % 64 != 0:
            raise ValueError(f"shape_k must be divisible by 64, got {shape_k}")
        if torch_dtype != torch.bfloat16:
            raise ValueError(
                "the indexed W4A16 layer currently supports BF16 activation/scale, "
                f"got torch_dtype={torch_dtype}"
            )

        self.shape_n = shape_n
        self.shape_k = shape_k
        self.num_experts = num_experts
        self.expert_parallel_size = expert_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.torch_dtype = torch_dtype
        self._profile_spec = profile
        # Only an explicit mode/role is recorded; an auto layer leaves the role
        # open so resolution can pick it per architecture (SM90 reads
        # CHORD_SM90_DECODE with a prefill default, Blackwell is decode-only).
        self._mode = _normalise_mode(mode)
        if role is not None:
            role_mode = _normalise_mode(role)
            if self._mode is not None and self._mode != role_mode:
                raise ValueError("mode and role must agree when both are provided")
            self._mode = role_mode
        # Resolve explicit profiles now (without querying CUDA); leave auto until
        # the first real CUDA weight is transformed.
        if isinstance(profile, IndexedLayerProfile):
            self.profile = select_indexed_profile(
                profile,
                mode=self._mode,
                expert_parallel_size=expert_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
            )
        elif profile not in (None, "auto"):
            self.profile = select_indexed_profile(
                profile,
                mode=None if mode is None and role is None else self._mode,
                expert_parallel_size=expert_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
            )
        else:
            self.profile = None

        alloc_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.weight = torch.nn.Parameter(
            torch.empty(
                (num_experts, shape_n, shape_k // 8),
                dtype=torch.int32,
                device=alloc_device,
            ),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.empty(
                (num_experts, shape_n, shape_k // 32),
                # The indexed kernel currently consumes BF16 group scales.  The
                # activation dtype remains configurable independently below.
                dtype=torch.bfloat16,
                device=alloc_device,
            ),
            requires_grad=False,
        )
        self._weight_format = "checkpoint_packed"
        self._prepared_weight: object | None = None
        # Framework adapters read this mapping to discover the logical shape and
        # fixed profile.  The attribute name matches the one Humming-derived
        # adapters already look for.
        self.humming_metas: dict[str, IndexedLayerMeta] = {}
        self._set_humming_meta("")

    @property
    def _is_tensor_parallel(self) -> bool:
        """Whether this layer is TP8-sharded, by explicit size or by shape."""
        return (
            resolve_shard_axis(
                self.tensor_parallel_size, self.shape_n, self.shape_k
            )
            == "tp"
        )

    @property
    def indexed_profile(self) -> IndexedLayerProfile:
        if self.profile is None:
            if not self.weight.is_cuda:
                # A CPU/meta layer cannot query hardware.  Use the matching
                # Hopper schedule as a provisional layout — the SM90 role
                # comes from the explicit mode, else CHORD_SM90_DECODE, else
                # the prefill default, routed through the backend policy
                # (CHORD_USE_GROUPED reroutes both roles once the grouped
                # backend is registered).  An auto layer moved to Blackwell
                # must be recreated with the explicit Blackwell profile so
                # weights are never silently repacked for another layout.
                #
                # A TP8 layer keeps the TP8 provisional layout so the CPU-side
                # metadata matches the table it will resolve to; being a mix
                # profile, it takes the mix role and so neither the P/D bit nor
                # the grouped switch applies to it.  The axis comes from an
                # explicit size when given, else from this layer's own shapes.
                if self._is_tensor_parallel:
                    role: IndexedMode = "mix"
                else:
                    role = self._mode or indexed_mode_from_env() or "prefill"
                return select_indexed_profile(
                    dispatch.profile_name_for_role(
                        resolve_backend_name(role), role
                    ),
                    expert_parallel_size=self.expert_parallel_size,
                    tensor_parallel_size=self.tensor_parallel_size,
                )
            self.profile = select_indexed_profile(
                "auto",
                mode=self._mode,
                device=self.weight.device,
                expert_parallel_size=self.expert_parallel_size,
                tensor_parallel_size=self.tensor_parallel_size,
                shape_n=self.shape_n,
                shape_k=self.shape_k,
            )
        return self.profile

    @property
    def backend(self) -> IndexedBackend:
        return self.indexed_profile.backend

    @property
    def layout(self) -> WeightLayout:
        return self.indexed_profile.layout

    @property
    def block_m(self) -> int:
        return self.indexed_profile.block_m

    def _set_humming_meta(self, sublayer_name: str = "") -> IndexedLayerMeta:
        if not isinstance(sublayer_name, str):
            raise TypeError(
                f"sublayer_name must be a string, got {type(sublayer_name).__name__}"
            )
        meta = IndexedLayerMeta(
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            num_experts=self.num_experts,
            profile=self.indexed_profile,
            sublayer_name=sublayer_name,
        )
        self.humming_metas[sublayer_name] = meta
        return meta

    def _refresh_humming_metas(self) -> None:
        profile = self.indexed_profile
        if not self.humming_metas:
            self._set_humming_meta("")
            return
        for name, meta in tuple(self.humming_metas.items()):
            self.humming_metas[name] = dataclasses.replace(
                meta,
                shape_n=self.shape_n,
                shape_k=self.shape_k,
                num_experts=self.num_experts,
                profile=profile,
            )

    def load_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        packed: bool | None = None,
        transform: bool = True,
    ) -> "IndexedW4A16Layer":
        """Load checkpoint tensors and optionally prepare the kernel layout."""
        if not isinstance(weight, torch.Tensor) or not isinstance(
            weight_scale, torch.Tensor
        ):
            raise TypeError("weight and weight_scale must be torch.Tensor objects")
        if weight.device != weight_scale.device:
            raise ValueError(
                "weight and weight_scale must be on the same device, got "
                f"{weight.device} and {weight_scale.device}"
            )
        if weight.dtype != torch.int32:
            raise TypeError(f"weight must be torch.int32, got {weight.dtype}")
        if weight_scale.dtype != torch.bfloat16:
            raise TypeError(
                f"weight_scale must be torch.bfloat16, got {weight_scale.dtype}"
            )
        if weight.ndim == 2 and self.num_experts == 1:
            weight = weight.unsqueeze(0)
        if weight_scale.ndim == 2 and self.num_experts == 1:
            weight_scale = weight_scale.unsqueeze(0)
        if (
            weight.ndim != 3
            or weight.shape[0] != self.num_experts
            or weight.shape[1] != self.shape_n
        ):
            raise ValueError(
                f"weight must have leading shape ({self.num_experts}, {self.shape_n}, _), "
                f"got {tuple(weight.shape)}"
            )
        if weight_scale.shape != (self.num_experts, self.shape_n, self.shape_k // 32):
            raise ValueError(
                "weight_scale must have shape "
                f"{self.num_experts, self.shape_n, self.shape_k // 32}, "
                f"got {tuple(weight_scale.shape)}"
            )
        if packed is None:
            packed = weight.shape[-1] == self.shape_k // 8
        if packed:
            if weight.shape[-1] != self.shape_k // 8:
                raise ValueError(
                    f"packed weight must have K/8={self.shape_k // 8} words, got {weight.shape[-1]}"
                )
            weight_format = "checkpoint_packed"
        else:
            if weight.shape[-1] != self.shape_k:
                raise ValueError(
                    f"unpacked weight must have K={self.shape_k} codes, got {weight.shape[-1]}"
                )
            weight_format = "unpacked"

        weight_device = (
            weight.device
            if self.weight.is_meta
            or (self.weight.device.type == "cpu" and weight.is_cuda)
            else self.weight.device
        )
        scale_device = (
            weight_scale.device
            if self.weight_scale.is_meta
            or (self.weight_scale.device.type == "cpu" and weight_scale.is_cuda)
            else self.weight_scale.device
        )
        _replace_parameter(self, "weight", weight.contiguous().to(weight_device))
        _replace_parameter(
            self,
            "weight_scale",
            weight_scale.contiguous().to(scale_device),
        )
        self._weight_format = weight_format
        self._prepared_weight = None
        if transform:
            self.transform()
        return self

    def load_from_unquantized(
        self,
        weight_uint4: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        transform: bool = True,
    ) -> "IndexedW4A16Layer":
        return self.load_weight(
            weight_uint4, weight_scale, packed=False, transform=transform
        )

    def transform(self) -> "IndexedW4A16Layer":
        """Convert checkpoint tensors to the profile backend's kernel layout."""
        if self._prepared_weight is not None:
            return self
        if not self.weight.is_cuda:
            # Keep checkpoint-packed weights compact while a framework constructs
            # or loads the module on CPU/meta. The first CUDA forward (or an
            # explicit process_weights_after_loading call) performs both steps.
            return self

        if (
            self._weight_format == "checkpoint_packed"
            and self.weight.shape[-1] == self.shape_k
        ):
            # Framework loaders sometimes assign an already-unpacked tensor
            # directly to ``layer.weight``.  Recognize that form without making
            # the caller set a private format flag.
            source_weight = self.weight
            source_is_packed = False
        elif self._weight_format == "checkpoint_packed":
            source_weight = self.weight
            source_is_packed = True
        elif self._weight_format == "unpacked":
            source_weight = self.weight
            source_is_packed = False
        elif self._weight_format == "kernel":
            return self
        else:
            raise RuntimeError(f"unknown weight format {self._weight_format!r}")
        # Profile resolution is deliberately deferred for auto layers.  Explicit
        # profiles are checked only once a CUDA tensor is actually available.
        profile = self.indexed_profile
        self._refresh_humming_metas()
        profile.validate_device(source_weight.device)

        meta = self.humming_metas[""]
        prepared = dispatch.pack_weight(
            source_weight,
            self.weight_scale,
            meta=meta,
            packed=source_is_packed,
        )

        self._prepared_weight = prepared
        if _install_prepared_tensors(self, prepared):
            self._weight_format = "kernel"
        return self

    def _apply(self, fn):
        # Prepared weights are dataclasses rather than registered modules,
        # so refresh their tensor references when a framework moves this layer.
        result = super()._apply(fn)
        prepared = self._prepared_weight
        if dataclasses.is_dataclass(prepared) and isinstance(
            getattr(prepared, "packed", None), torch.Tensor
        ):
            self._prepared_weight = dataclasses.replace(
                prepared,
                packed=self.weight,
                scale=self.weight_scale,
            )
            if self.weight.is_cuda:
                # The physical layout is profile-specific.  A packed H200 layer
                # cannot be migrated to another architecture and silently keep
                # using the old schedule.
                self.indexed_profile.validate_device(self.weight.device)
        return result

    process_weights_after_loading = transform

    def forward(
        self,
        inputs: torch.Tensor,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        top_k: int | None = None,
        *,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        valid_shape_m: int = 0,
        # Framework routers (including vLLM) already produce aligned buffers;
        # skip GPU-to-CPU count reads and route/expert value scans by default.
        # Standalone callers can enable synchronized validation explicitly.
        validate_routing: bool = False,
        block_m: int | None = None,
        # These names are accepted by the upstream Humming layer delegation.
        # ``expert_layout`` and ``m_indices`` are the grouped backends'
        # routing forms; the indexed backend rejects both and keeps
        # sorted_ids/expert_ids routing.
        expert_layout: torch.Tensor | None = None,
        m_indices: torch.Tensor | None = None,
        compute_config: object | None = None,
        tuning_config: object | None = None,
        sublayer_name: str = "",
        **kwargs,
    ) -> torch.Tensor:
        del sublayer_name, kwargs
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        if self._prepared_weight is None:
            self.transform()
        if self._prepared_weight is None:
            raise RuntimeError(
                "W4A16 weights are not packed for CUDA; move the layer to CUDA and "
                "call process_weights_after_loading()"
            )
        if (
            isinstance(self._prepared_weight, PreparedWeight)
            and not self.weight.is_cuda
        ):
            raise RuntimeError(
                "prepared indexed W4A16 weights must remain on CUDA; move the layer "
                "to CUDA and repack it before forward"
            )
        profile = self.indexed_profile
        profile_meta = self.humming_metas.get("")
        if not isinstance(profile_meta, IndexedLayerMeta):
            profile_meta = self._set_humming_meta("")
        return dispatch.forward_w4a16(
            self._prepared_weight,
            profile_meta,
            inputs,
            outputs=outputs,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            m_indices=m_indices,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
            compute_config=compute_config,
            tuning_config=tuning_config,
            block_m=block_m,
            validate_routing=validate_routing,
        )

    def forward_indexed(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_layer(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _parameter_tensor(layer: object, name: str) -> torch.Tensor:
    try:
        tensor = getattr(layer, name)
    except AttributeError as exc:
        raise AttributeError(f"layer is missing required tensor {name!r}") from exc
    if isinstance(tensor, torch.nn.Parameter):
        tensor = tensor.data
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"layer.{name} must be a torch.Tensor")
    return tensor


def _validate_indexed_schema(
    weight_schema: object | None, input_schema: object | None
) -> None:
    """Reject schemas the extracted kernel cannot represent."""

    def parse_dtype(value: object) -> object:
        if isinstance(value, str):
            try:
                return dtypes.DataType.from_str(value)
            except (TypeError, ValueError):
                return value
        return value

    def is_bfloat16(value: object) -> bool:
        value = parse_dtype(value)
        return (
            getattr(value, "num_bits", None) == 16
            and getattr(value, "is_floating_point_type", False)
            and getattr(value, "exponent_bits", None) == 8
            and getattr(value, "mantissa_bits", None) == 7
        )

    if weight_schema is not None:
        b_dtype = parse_dtype(getattr(weight_schema, "b_dtype", None))
        if b_dtype is None or (
            getattr(b_dtype, "num_bits", None) != 4
            or not getattr(b_dtype, "is_integer_type", False)
            or getattr(b_dtype, "is_signed", True)
        ):
            raise ValueError("indexed W4A16 requires a 4-bit unsigned weight schema")
        group_size = getattr(weight_schema, "weight_scale_group_size", 32)
        if group_size not in (None, 32):
            raise ValueError("indexed W4A16 requires weight scale group size 32")
        group_size_n = getattr(weight_schema, "weight_scale_group_size_n", 1)
        if group_size_n not in (None, 0, 1):
            raise ValueError(
                "indexed W4A16 requires one scale per output channel "
                "(weight_scale_group_size_n must be 0 or 1)"
            )
        scale_type = getattr(weight_schema, "weight_scale_type", None)
        scale_type = getattr(scale_type, "value", scale_type)
        if scale_type is not None and "group" not in str(scale_type).lower():
            raise ValueError("indexed W4A16 requires group weight scales")
        bs_dtype = getattr(weight_schema, "bs_dtype", dtypes.bfloat16)
        if bs_dtype is not None and not is_bfloat16(bs_dtype):
            raise ValueError("indexed W4A16 requires BF16 group scales")
        if getattr(weight_schema, "has_zero_point", False):
            raise ValueError("indexed W4A16 does not support an explicit zero point")
        if getattr(weight_schema, "has_bias", False):
            raise ValueError("indexed W4A16 does not support a weight bias")
        if getattr(weight_schema, "is_fp_zero_point", False):
            raise ValueError("indexed W4A16 does not support an explicit zero point")
        if getattr(weight_schema, "hadamard_block_size", 0) not in (None, 0, 1):
            raise ValueError("indexed W4A16 does not support Hadamard-rotated weights")
        if getattr(weight_schema, "use_fused_e8m0_scale", False):
            raise ValueError("indexed W4A16 requires BF16, not fused E8M0 scales")
    if input_schema is not None:
        a_dtype = parse_dtype(getattr(input_schema, "a_dtype", None))
        if a_dtype is not None and not is_bfloat16(a_dtype):
            raise ValueError("indexed W4A16 requires BF16 activations")
        if getattr(input_schema, "input_scale_group_size", 0) not in (None, 0):
            raise ValueError("indexed W4A16 does not support input scales")


def _host_profile(
    layer: object,
    kwargs: Mapping[str, Any],
    shape_n: int | None = None,
    shape_k: int | None = None,
) -> IndexedLayerProfile:
    explicit = kwargs.get("profile")
    if explicit is None:
        # A live `indexed_profile` property re-resolves auto layers after they
        # move to CUDA, so it must win over any previously cached value; the
        # `_w4a16_profile` cache only serves foreign host layers that carry no
        # profile attribute of their own.
        explicit = getattr(layer, "indexed_profile", None)
    if explicit is None:
        explicit = getattr(layer, "w4a16_profile", None)
    if explicit is None:
        explicit = getattr(layer, "_w4a16_profile", None)
    if explicit is None:
        candidate = getattr(layer, "profile", None)
        if isinstance(candidate, (IndexedLayerProfile, str)):
            explicit = candidate
    if explicit is None:
        explicit = "auto"
    mode = kwargs.get("mode")
    if mode is None:
        mode = getattr(layer, "w4a16_mode", getattr(layer, "_mode", None))
    if mode is None and isinstance(explicit, IndexedLayerProfile):
        mode = explicit.mode
    device = None
    for name in ("w13_weight", "w2_weight", "weight"):
        candidate = getattr(layer, name, None)
        if isinstance(candidate, torch.Tensor) and candidate.is_cuda:
            device = candidate.device
            break
    # The shard axis is a deployment property, not a device one, so it is taken
    # from an explicit size when the caller states one and otherwise recovered
    # from the projection shapes.  A framework adapter passes the same
    # shape_n/shape_k it always passes, which is how it reaches TP8 without a
    # chord-specific argument; an unrecognised pair infers nothing and keeps the
    # EP8 default.
    tensor_parallel_size = kwargs.get("tensor_parallel_size")
    if tensor_parallel_size is None:
        tensor_parallel_size = getattr(layer, "tensor_parallel_size", None)
    if shape_n is None:
        shape_n = getattr(layer, "shape_n", None)
    if shape_k is None:
        shape_k = getattr(layer, "shape_k", None)
    # Resolved even when an explicit profile short-circuits the auto path, so a
    # stated size that contradicts the shapes is rejected either way.
    axis = resolve_shard_axis(tensor_parallel_size, shape_n, shape_k)
    if explicit == "auto" and device is None and not torch.cuda.is_available():
        # CPU/meta construction cannot inspect an architecture.  Keep the
        # Hopper metadata for the SM90 role (explicit mode, else
        # CHORD_SM90_DECODE, else the prefill default), routed through the
        # backend policy; callers targeting Blackwell should pass
        # profile="blackwell_decode_ep8" before loading.  A TP8 layer takes the
        # mix role, so neither the P/D bit nor the grouped switch reaches it.
        if axis == "tp":
            role: IndexedMode = "mix"
        else:
            role = mode or indexed_mode_from_env() or "prefill"
        explicit = dispatch.profile_name_for_role(resolve_backend_name(role), role)
    profile = select_indexed_profile(
        explicit,
        mode=mode,
        device=device,
        tensor_parallel_size=tensor_parallel_size,
        shape_n=shape_n,
        shape_k=shape_k,
    )
    try:
        setattr(layer, "_w4a16_profile", profile)
    except Exception:
        pass
    return profile


def _host_prepared_weight(layer: object, meta: IndexedLayerMeta) -> object:
    """Read a transformed sublayer straight off a host framework's module."""
    packed = _parameter_tensor(layer, meta.weight_name)
    scale = _parameter_tensor(layer, meta.weight_scale_name)
    expected_packed, expected_scale = dispatch.transformed_shapes(meta)
    if tuple(packed.shape) != expected_packed or tuple(scale.shape) != expected_scale:
        raise RuntimeError(
            f"{meta.sublayer_name or 'weight'} is not transformed to the "
            f"{meta.backend} layout; call "
            "IndexedW4A16Method.transform_humming_layer first"
        )
    return dispatch.build_prepared(packed, scale, meta)


def _transform_host_weight(layer: object, meta: IndexedLayerMeta) -> None:
    weight = _parameter_tensor(layer, meta.weight_name)
    scale = _parameter_tensor(layer, meta.weight_scale_name)
    if not weight.is_cuda:
        raise RuntimeError(
            "indexed W4A16 weight transformation requires CUDA; move the layer "
            "to its serving device before process_weights_after_loading"
        )
    expected_packed, _ = dispatch.transformed_shapes(meta)
    if tuple(weight.shape) == expected_packed:
        # Already transformed by a previous loading hook.
        return
    if tuple(weight.shape) == (meta.num_experts, meta.shape_n, meta.shape_k // 8):
        source_is_packed = True
    elif tuple(weight.shape) != (meta.num_experts, meta.shape_n, meta.shape_k):
        raise ValueError(
            f"{meta.weight_name} must be [E,N,K/8] packed or [E,N,K] unpacked, "
            f"got {tuple(weight.shape)}"
        )
    else:
        source_is_packed = False
    if scale.dtype != torch.bfloat16:
        raise TypeError(f"{meta.weight_scale_name} must be torch.bfloat16")
    prepared = dispatch.pack_weight(weight, scale, meta=meta, packed=source_is_packed)
    _install_prepared_tensors(
        layer,
        prepared,
        weight_name=meta.weight_name,
        scale_name=meta.weight_scale_name,
    )


class IndexedW4A16Method:
    """Tiny method object matching the common framework delegation pattern.

    Tuning rows come from the indexed profile resolver, and unsupported schema
    features are rejected before a weight is transformed.
    """

    @classmethod
    def may_set_param(
        cls, layer: torch.nn.Module, name: str, tensor: torch.Tensor | None
    ) -> None:
        """Install a non-trainable tensor using the upstream helper convention."""
        del cls
        if tensor is None:
            return
        _replace_parameter(layer, name, tensor)

    @staticmethod
    def may_quant_input(
        layer: object | torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        **kwargs,
    ):
        """Pass BF16 inputs through using both Humming call conventions.

        Upstream Humming calls this as ``(layer, inputs, input_scale=...)``;
        the small adapter also accepts ``(inputs)`` or ``inputs=...``.  There is
        no input quantization in this BF16-only extraction.  An explicitly
        supplied scale is rejected instead of being silently ignored.
        """
        del quanted_input, kwargs
        if inputs is None and isinstance(layer, torch.Tensor):
            inputs = layer
        if inputs is None:
            raise TypeError("inputs must be provided to may_quant_input")
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        return inputs, None

    @staticmethod
    def may_hadamard_quant_input(
        layer: object | torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        hadamard_block_size: int | None = None,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        m_major_scale: bool = False,
        sublayer_name: str = "",
        **kwargs,
    ):
        """BF16 passthrough for the companion upstream helper.

        The indexed extraction has no input quantization or Hadamard rotation;
        accepting this call shape lets a framework share its Humming dispatch
        code without silently allocating an unused temporary.
        """
        del layer, quanted_input, m_major_scale, sublayer_name, kwargs
        if hadamard_block_size not in (None, 0, 1):
            raise ValueError(
                "indexed W4A16 accepts BF16 input only; Hadamard quantization is unsupported"
            )
        if inputs is None:
            raise TypeError("inputs must be provided to may_hadamard_quant_input")
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        return inputs, None

    @classmethod
    def get_default_tuning_configs(
        cls,
        layer,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        use_m_major_input_scale: bool = False,
        gemm_type: object = "indexed",
        sublayer_name: str = "",
        **kwargs,
    ):
        del use_m_major_input_scale
        if use_f16_accum or use_batch_invariant:
            raise ValueError(
                "indexed W4A16 tuning rows require use_f16_accum=False and "
                "use_batch_invariant=False"
            )
        gemm_type_value = getattr(gemm_type, "value", gemm_type)
        if str(gemm_type_value).lower() != "indexed":
            raise ValueError(
                "IndexedW4A16Method only provides tuning rows for gemm_type='indexed'"
            )
        if not hasattr(layer, "humming_metas"):
            layer.humming_metas = {}
        meta = layer.humming_metas.get(sublayer_name)
        if not isinstance(meta, IndexedLayerMeta):
            meta_kwargs = dict(kwargs)
            meta = cls.prepare_layer_meta(
                layer, sublayer_name=sublayer_name, **meta_kwargs
            )
        return _indexed_tuning_rows(meta)

    @classmethod
    def prepare_layer_meta(
        cls,
        layer,
        shape_n: int | None = None,
        shape_k: int | None = None,
        weight_schema: object | None = None,
        input_schema: object | None = None,
        num_experts: int | None = None,
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        has_bias: bool = False,
        torch_dtype: torch.dtype | None = None,
        sublayer_name: str = "",
        **kwargs,
    ):
        if not isinstance(sublayer_name, str):
            raise TypeError(
                f"sublayer_name must be a string, got {type(sublayer_name).__name__}"
            )
        kwargs = dict(kwargs)
        for name, requested in (
            ("shape_n", shape_n),
            ("shape_k", shape_k),
            ("num_experts", num_experts),
        ):
            current = getattr(layer, name, None)
            if requested is not None and current is not None and requested != current:
                raise ValueError(
                    f"{name}={requested} does not match layer.{name}={current}"
                )
        pad_n_multiple = pad_n_to_multiple
        pad_k_multiple = pad_k_to_multiple
        if (
            isinstance(pad_n_multiple, bool)
            or not isinstance(pad_n_multiple, int)
            or pad_n_multiple <= 0
            or isinstance(pad_k_multiple, bool)
            or not isinstance(pad_k_multiple, int)
            or pad_k_multiple <= 0
        ):
            raise ValueError("padding multiples must be positive integers")
        shape_n = shape_n if shape_n is not None else getattr(layer, "shape_n", None)
        shape_k = shape_k if shape_k is not None else getattr(layer, "shape_k", None)
        if shape_n is not None and pad_n_multiple and shape_n % pad_n_multiple:
            raise ValueError("indexed W4A16 metadata requires shape_n without padding")
        if shape_k is not None and pad_k_multiple and shape_k % pad_k_multiple:
            raise ValueError("indexed W4A16 metadata requires shape_k without padding")
        if not hasattr(layer, "humming_metas"):
            layer.humming_metas = {}
        if has_bias or kwargs.get("has_zero_point", False):
            raise ValueError(
                "indexed W4A16 supports neither bias nor explicit zero point"
            )
        if torch_dtype is None:
            torch_dtype = getattr(layer, "param_dtype", None)
        if torch_dtype is not None and torch_dtype != torch.bfloat16:
            raise ValueError(
                f"indexed W4A16 requires torch.bfloat16 parameters, got {torch_dtype}"
            )
        _validate_indexed_schema(weight_schema, input_schema)
        profile = _host_profile(layer, kwargs, shape_n, shape_k)
        if hasattr(layer, "_set_humming_meta"):
            layer_profile = getattr(layer, "indexed_profile")
            if layer_profile.name != profile.name:
                raise ValueError(
                    f"profile {profile.name!r} does not match the layer's "
                    f"profile {layer_profile.name!r}"
                )
            return layer._set_humming_meta(sublayer_name)
        num_experts = (
            num_experts
            if num_experts is not None
            else getattr(layer, "num_experts", None)
        )
        if shape_n is None or shape_k is None or num_experts is None:
            raise TypeError(
                "shape_n, shape_k, and num_experts are required for layer metadata"
            )
        meta = IndexedLayerMeta(
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=num_experts,
            profile=profile,
            sublayer_name=sublayer_name,
            pad_shape_n=0,
            pad_shape_k=0,
        )
        layer.humming_metas[sublayer_name] = meta
        return meta

    @classmethod
    def transform_humming_layer(
        cls,
        layer,
        sublayer_name: str = "",
        already_padded: bool = False,
        **kwargs,
    ):
        del already_padded
        if "sublayer_name" in kwargs:
            requested = kwargs.pop("sublayer_name")
            if sublayer_name and requested != sublayer_name:
                raise ValueError("duplicate sublayer_name arguments disagree")
            sublayer_name = requested
        if not isinstance(sublayer_name, str):
            raise TypeError(
                f"sublayer_name must be a string, got {type(sublayer_name).__name__}"
            )
        if isinstance(layer, IndexedW4A16Layer):
            return layer.transform()
        metas = getattr(layer, "humming_metas", None)
        if not isinstance(metas, dict) or not isinstance(
            metas.get(sublayer_name), IndexedLayerMeta
        ):
            cls.prepare_layer_meta(layer, sublayer_name=sublayer_name, **kwargs)
        meta = layer.humming_metas[sublayer_name]
        _transform_host_weight(layer, meta)
        return layer

    @classmethod
    def forward_layer(
        cls,
        layer: IndexedW4A16Layer,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        m_indices: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
        compute_config: object | None = None,
        tuning_config: object | None = None,
        sublayer_name: str = "",
        hadamard_block_size: int | None = None,
        validate_routing: bool = False,
        block_m: int | None = None,
        **kwargs,
    ):
        del kwargs
        if hadamard_block_size not in (None, 0, 1):
            raise ValueError(
                "indexed W4A16 accepts BF16 input only; Hadamard quantization is unsupported"
            )
        # Preserve the small delegation contract for framework shims that only
        # implement ``forward``.  A real vLLM RoutedExperts object takes the
        # named-parameter path below after prepare_layer_meta has installed its
        # indexed metadata.
        if (
            not isinstance(layer, IndexedW4A16Layer)
            and not isinstance(getattr(layer, "humming_metas", None), dict)
            and hasattr(layer, "forward")
        ):
            return layer.forward(
                inputs,
                sorted_ids,
                expert_ids,
                num_tokens_padded,
                top_k,
                outputs=outputs,
                input_scale=input_scale,
                expert_layout=expert_layout,
                m_indices=m_indices,
                valid_shape_m=valid_shape_m,
                validate_routing=validate_routing,
                block_m=block_m,
            )
        if input_scale is not None:
            raise ValueError("indexed W4A16 does not accept an input scale")
        if isinstance(layer, IndexedW4A16Layer):
            return layer.forward(
                inputs,
                sorted_ids,
                expert_ids,
                num_tokens_padded,
                top_k,
                outputs=outputs,
                expert_layout=expert_layout,
                m_indices=m_indices,
                compute_config=compute_config,
                tuning_config=tuning_config,
                sublayer_name=sublayer_name,
                valid_shape_m=valid_shape_m,
                validate_routing=validate_routing,
                block_m=block_m,
            )

        metas = getattr(layer, "humming_metas", None)
        if not isinstance(metas, dict) or not isinstance(
            metas.get(sublayer_name), IndexedLayerMeta
        ):
            raise TypeError(
                f"layer has no indexed metadata for sublayer {sublayer_name!r}; "
                "call prepare_layer_meta first"
            )
        meta = metas[sublayer_name]
        prepared = _host_prepared_weight(layer, meta)
        return dispatch.forward_w4a16(
            prepared,
            meta,
            inputs,
            outputs=outputs,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            m_indices=m_indices,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
            compute_config=compute_config,
            tuning_config=tuning_config,
            block_m=block_m,
            validate_routing=validate_routing,
        )


__all__ = [
    "BLACKWELL_DECODE_EP8",
    "H200_DECODE_EP8",
    "H200_GROUPED_DECODE",
    "H200_GROUPED_PREFILL",
    "H200_PREFILL_EP8",
    "H200_TP8",
    "INDEXED_PROFILES",
    "IndexedBackend",
    "IndexedKernelConfig",
    "IndexedLayerMeta",
    "IndexedLayerProfile",
    "IndexedMode",
    "IndexedW4A16Layer",
    "IndexedW4A16Method",
    "indexed_mode_from_env",
    "select_indexed_profile",
    "unpack_packed_uint4",
    "use_grouped_from_env",
]
