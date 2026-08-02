# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import dataclasses
import functools

import torch

from chord_kernels.operator.kernel.humming import HummingKernel
from chord_kernels.operator.ops import launch_kernel
from chord_kernels.operator.packing import (
    WeightLayout,
    PreparedWeight,
    _validate_layout,
    _validate_supported_architecture,
    pack_w4a16,
)


@dataclasses.dataclass(frozen=True)
class IndexedKernelConfig:
    """Small immutable launch description for the indexed W4A16 kernel.

    The public profiles only use the fields below.  Keeping the description
    separate from :class:`HummingKernel` lets a layer select a narrow tuning
    row without exposing unrelated dense/grouped configuration surfaces.
    """

    block_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]
    num_stages: int = 4
    num_ctas_per_sm: int = 1
    swap_ab: bool = False
    use_stream_k: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("block_shape", self.block_shape),
            ("warp_shape", self.warp_shape),
        ):
            if len(value) != 3 or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in value
            ):
                raise ValueError(f"{name} must contain three positive integers")
        if self.block_shape[0] % 8:
            raise ValueError("indexed block-M must be a multiple of 8")
        if any(self.block_shape[i] % self.warp_shape[i] for i in range(3)):
            raise ValueError("indexed block shape must be divisible by warp shape")
        if self.warp_shape[1] not in (32, 64):
            raise ValueError("indexed warp-N must be 32 (WGMMA) or 64 (MMA)")
        if (
            isinstance(self.num_stages, bool)
            or not isinstance(self.num_stages, int)
            or self.num_stages < 2
        ):
            raise ValueError("num_stages must be an integer of at least 2")
        if (
            isinstance(self.num_ctas_per_sm, bool)
            or not isinstance(self.num_ctas_per_sm, int)
            or self.num_ctas_per_sm <= 0
        ):
            raise ValueError("num_ctas_per_sm must be positive")
        if not isinstance(self.swap_ab, bool):
            raise TypeError("swap_ab must be a bool")
        if not isinstance(self.use_stream_k, bool):
            raise TypeError("use_stream_k must be a bool")
        # The launcher's device-resident lock buffer holds 1024 words and the
        # scheduler peels at most ~2x the grid in stream-K tiles, so 4 CTAs/SM
        # on a 128-SM part is already the ceiling.  The launcher re-checks the
        # exact grid at launch; rejecting the config here fails earlier.
        if self.use_stream_k and self.num_ctas_per_sm > 4:
            raise ValueError(
                "use_stream_k supports at most 4 CTAs/SM (stream-K lock "
                f"capacity), got num_ctas_per_sm={self.num_ctas_per_sm}"
            )

    @property
    def block_m(self) -> int:
        return self.block_shape[0]

    @property
    def block_n(self) -> int:
        return self.block_shape[1]

    @property
    def block_k(self) -> int:
        return self.block_shape[2]

    @classmethod
    def default(
        cls,
        *,
        layout: WeightLayout,
        block_m: int,
        block_n: int,
        swap_ab: bool,
    ) -> "IndexedKernelConfig":
        warp_n = 32 if layout == "wgmma" else 64
        return cls(
            block_shape=(block_m, block_n, 64),
            warp_shape=(block_m, warp_n, 64),
            swap_ab=swap_ab,
            num_ctas_per_sm=4 if swap_ab else 1,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "IndexedKernelConfig":
        if value.get("use_tma_b", False) or value.get("use_tma_bs", False):
            raise ValueError("indexed W4A16 supports cp.async only")
        block_shape = tuple(value["block_shape"])
        warp_shape = tuple(value["warp_shape"])
        return cls(
            block_shape=block_shape,  # type: ignore[arg-type]
            warp_shape=warp_shape,  # type: ignore[arg-type]
            num_stages=value.get("num_stages", 4),
            num_ctas_per_sm=value.get("num_ctas_per_sm", 1),
            swap_ab=value.get("swap_ab", False),
            use_stream_k=value.get("use_stream_k", False),
        )

    def to_dict(self) -> dict:
        return {
            "block_shape": self.block_shape,
            "warp_shape": self.warp_shape,
            "num_stages": self.num_stages,
            "num_ctas_per_sm": self.num_ctas_per_sm,
            "swap_ab": self.swap_ab,
            "use_stream_k": self.use_stream_k,
        }


def _compatible_block_n(shape_n: int) -> int:
    candidate = 1 << min(shape_n, 256).bit_length() - 1
    while candidate > 1 and shape_n % candidate:
        candidate //= 2
    if candidate < 128:
        raise ValueError(
            f"shape_n must have a compatible power-of-two tile >= 128, got {shape_n}"
        )
    return candidate


@functools.lru_cache(maxsize=128)
def _get_indexed_kernel(
    device_index: int,
    shape_n: int,
    shape_k: int,
    num_experts: int,
    layout: WeightLayout,
    config: IndexedKernelConfig,
) -> HummingKernel:
    block_m, block_n, block_k = config.block_shape
    warp_shape = config.warp_shape
    swap_ab = config.swap_ab
    if swap_ab and layout != "mma":
        raise ValueError("swap_ab requires an MMA-packed weight")
    if layout == "mma" and not swap_ab and config.block_m % 16 == 8:
        raise ValueError(
            "MMA indexed configs with block_m=8 (or another 16q+8 tile) "
            "must set swap_ab=True"
        )

    if block_k <= 0 or block_k % 64:
        raise ValueError(
            f"indexed block-K must be a positive multiple of 64, got {block_k}"
        )

    mma_type = "wgmma" if layout == "wgmma" else "mma"
    if mma_type == "wgmma" and block_n < 128:
        raise ValueError("the WGMMA layout requires shape_n/block_n >= 128")

    # WGMMA instructions execute across a full four-warp group. With the
    # minimum block-N of 128, warp-N 32 produces exactly those 128 threads.
    with torch.cuda.device(device_index):
        return HummingKernel(
            shape_n=shape_n,
            shape_k=shape_k,
            block_shape=config.block_shape,
            warp_shape=config.warp_shape,
            num_experts=num_experts,
            num_stages=config.num_stages,
            num_ctas_per_sm=config.num_ctas_per_sm,
            mma_type=mma_type,
            swap_ab=swap_ab,
            use_stream_k=config.use_stream_k,
        )


def _check_cuda_int32(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous and on {device}")


def _validate_prepared_weight(
    weight: PreparedWeight,
    device: torch.device,
    requested_layout: WeightLayout | None,
) -> tuple[torch.Tensor, torch.Tensor, WeightLayout, int, int, int]:
    if not isinstance(weight, PreparedWeight):
        raise TypeError(
            "weight must be a PreparedWeight from pack_w4a16(); "
            "this preserves the MMA/WGMMA layout contract"
        )

    layout = _validate_layout(weight.layout)
    if requested_layout is not None:
        requested_layout = _validate_layout(requested_layout)
        if layout != requested_layout:
            raise ValueError(
                f"weight uses layout={layout!r}, but layout={requested_layout!r} "
                "was requested; repack or select the matching layout"
            )

    metadata = {
        "n": weight.n,
        "k": weight.k,
        "num_experts": weight.num_experts,
    }
    for name, value in metadata.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"weight metadata {name} must be a positive int, got {value!r}"
            )
    if weight.n % 128 or weight.k % 64:
        raise ValueError(
            f"weight metadata is not W4A16-aligned: N={weight.n}, K={weight.k}"
        )

    packed = weight.packed
    scale = weight.scale
    if not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError(
            "PreparedWeight packed and scale fields must be tensors"
        )
    expected_packed_shape = (weight.num_experts, weight.k // 16, weight.n * 2)
    expected_scale_shape = (weight.num_experts, weight.k // 32, weight.n)
    if packed.dtype != torch.int32 or tuple(packed.shape) != expected_packed_shape:
        raise ValueError(
            f"packed weight must be int32 {expected_packed_shape}, got "
            f"{tuple(packed.shape)}/{packed.dtype}"
        )
    if scale.dtype != torch.bfloat16 or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"packed scale must be BF16 {expected_scale_shape}, got "
            f"{tuple(scale.shape)}/{scale.dtype}"
        )
    if packed.device != device or scale.device != device:
        raise ValueError(
            f"inputs, packed weight, and scale must be on the same device; got "
            f"{device}, {packed.device}, and {scale.device}"
        )
    if not packed.is_contiguous() or not scale.is_contiguous():
        raise ValueError("packed weight and scale must be contiguous")
    return packed, scale, layout, weight.num_experts, weight.n, weight.k


def w4a16_indexed(
    inputs: torch.Tensor,
    weight: PreparedWeight,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    top_k: int,
    *,
    outputs: torch.Tensor | None = None,
    block_m: int | None = None,
    layout: WeightLayout | None = None,
    swap_ab: bool | None = None,
    config: IndexedKernelConfig | dict | None = None,
    valid_shape_m: int = 0,
    validate_routing: bool = True,
) -> torch.Tensor:
    """Run BF16 x INT4 group-32 indexed MoE GEMM.

    The three profiles select a narrow shape-aware tile table: H200 prefill uses
    WGMMA, while decode uses MMA (swap-AB for small token blocks). All rows load
    through cp.async rather than TMA; stream-K is enabled per row by the layer's
    tuning table and can be requested here through ``config``. If no explicit
    config is supplied, block-N defaults to the largest compatible power of two
    between 128 and 256 and MMA defaults to swap-AB. ``valid_shape_m`` is
    accepted for layer-path compatibility (where it selects the tuning bracket)
    and has no kernel-level effect here.
    ``sorted_ids`` and ``expert_ids`` may be overallocated routing buffers as
    returned by vLLM's ``moe_align_block_size``.  ``num_tokens_padded`` gives the
    active prefix length; only that prefix is consumed by the kernel.  Each active
    block contains valid row ids followed by sentinels greater than or equal to
    ``inputs.size(0) * top_k``.  Expert-parallel routing may legitimately contain
    only a subset of the flattened route ids.  Routing value validation
    synchronizes the CUDA device; disable it only for buffers from a trusted
    producer. The trusted fast path performs only shape/dtype/device checks and
    never reads ``num_tokens_padded`` back to the host, so it is CUDA-Graph safe.
    """
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"inputs must be a torch.Tensor, got {type(inputs).__name__}")
    if inputs.ndim != 2 or inputs.dtype != torch.bfloat16:
        raise TypeError(
            f"inputs must be a 2D torch.bfloat16 tensor, got {inputs.shape}/{inputs.dtype}"
        )
    if not inputs.is_cuda or not inputs.is_contiguous():
        raise ValueError("inputs must be a contiguous CUDA tensor")
    if inputs.size(0) == 0 or inputs.size(1) == 0:
        raise ValueError(
            f"inputs dimensions must be non-zero, got {tuple(inputs.shape)}"
        )
    _validate_supported_architecture(inputs.device)
    packed_weight, packed_scale, layout, num_experts, shape_n, shape_k = (
        _validate_prepared_weight(weight, inputs.device, layout)
    )
    if inputs.size(1) != shape_k:
        raise ValueError(
            f"inputs K={inputs.size(1)} does not match packed weight K={shape_k}"
        )
    if shape_k % 64:
        raise ValueError(
            f"indexed production kernel requires K divisible by 64, got {shape_k}"
        )

    _check_cuda_int32("sorted_ids", sorted_ids, inputs.device)
    _check_cuda_int32("expert_ids", expert_ids, inputs.device)
    _check_cuda_int32("num_tokens_padded", num_tokens_padded, inputs.device)
    if sorted_ids.ndim != 1:
        raise ValueError(f"sorted_ids must be 1D, got shape {tuple(sorted_ids.shape)}")
    if expert_ids.ndim != 1:
        raise ValueError(f"expert_ids must be 1D, got shape {tuple(expert_ids.shape)}")
    if not (
        num_tokens_padded.ndim == 0
        or (num_tokens_padded.ndim == 1 and num_tokens_padded.numel() == 1)
    ):
        raise ValueError(
            "num_tokens_padded must be a scalar or a one-element tensor, "
            f"got shape {tuple(num_tokens_padded.shape)}"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not isinstance(validate_routing, bool):
        raise TypeError(
            f"validate_routing must be a bool, got {type(validate_routing).__name__}"
        )

    requested_swap_ab = swap_ab
    if swap_ab is not None and not isinstance(swap_ab, bool):
        raise TypeError(f"swap_ab must be a bool or None, got {type(swap_ab).__name__}")
    if config is None:
        swap_ab = layout == "mma" if swap_ab is None else swap_ab
        if layout == "wgmma" and swap_ab:
            raise ValueError("WGMMA-packed weights cannot be used by swap_ab")
        if block_m is None:
            block_m = 8 if swap_ab else 16
        elif isinstance(block_m, bool) or not isinstance(block_m, int):
            raise TypeError(
                f"block_m must be an int or None, got {type(block_m).__name__}"
            )
        if block_m <= 0 or block_m % 8:
            raise ValueError(f"block_m must be a positive multiple of 8, got {block_m}")
        kernel_config = IndexedKernelConfig.default(
            layout=layout,
            block_m=block_m,
            block_n=_compatible_block_n(shape_n),
            swap_ab=swap_ab,
        )
    else:
        if not isinstance(config, (IndexedKernelConfig, dict)):
            raise TypeError(
                "config must be an IndexedKernelConfig, dict, or None; "
                f"got {type(config).__name__}"
            )
        if isinstance(config, IndexedKernelConfig):
            kernel_config = config
        else:
            config_dict = dict(config)
            config_layout = config_dict.get("layout")
            if config_layout is not None and _validate_layout(config_layout) != layout:
                raise ValueError(
                    "config layout does not match the packed weight: "
                    f"config={config_layout!r}, weight={layout!r}"
                )
            # The legacy short form omitted swap_ab; preserve its historical
            # MMA decode default while keeping WGMMA non-swap.
            config_dict.setdefault("swap_ab", layout == "mma")
            kernel_config = IndexedKernelConfig.from_dict(config_dict)
        # Explicit config is authoritative; accepting a second, conflicting
        # schedule makes routing alignment ambiguous.
        if block_m is not None and block_m != kernel_config.block_m:
            raise ValueError("block_m conflicts with config.block_shape[0]")
        if requested_swap_ab is not None and requested_swap_ab != kernel_config.swap_ab:
            raise ValueError("swap_ab conflicts with config")
        block_m = kernel_config.block_m
        swap_ab = kernel_config.swap_ab
        if layout == "wgmma" and swap_ab:
            raise ValueError("WGMMA-packed weights cannot be used by swap_ab")
        if block_m <= 0 or block_m % 8:
            raise ValueError(
                f"config block_m must be a positive multiple of 8, got {block_m}"
            )
    if isinstance(valid_shape_m, bool) or not isinstance(valid_shape_m, int):
        raise TypeError(
            f"valid_shape_m must be an int, got {type(valid_shape_m).__name__}"
        )
    if valid_shape_m < 0:
        raise ValueError(f"valid_shape_m must be non-negative, got {valid_shape_m}")
    if validate_routing:
        # Debug validation intentionally synchronizes once. The production
        # fast path above never reads a device scalar back to Python, which is
        # required for CUDA Graph capture and steady-state vLLM execution.
        padded_count = int(num_tokens_padded.reshape(-1)[0].item())
        if padded_count < 0:
            raise ValueError(
                f"num_tokens_padded must be non-negative, got {padded_count}"
            )
        if padded_count > sorted_ids.numel():
            raise ValueError(
                f"num_tokens_padded must not exceed sorted_ids capacity "
                f"{sorted_ids.numel()}, got {padded_count}"
            )
        if padded_count % block_m:
            raise ValueError(
                f"num_tokens_padded must be a multiple of block_m={block_m}, "
                f"got {padded_count}"
            )
        active_blocks = padded_count // block_m
        if expert_ids.numel() < active_blocks:
            raise ValueError(
                "expert_ids capacity must cover the active routed blocks "
                f"({active_blocks}), got {expert_ids.numel()}"
            )
        routed_limit = inputs.size(0) * top_k
        active_ids = sorted_ids[:padded_count]
        active_experts = expert_ids[:active_blocks]
        if padded_count:
            routed_rows = active_ids.view(-1, block_m)
            valid_rows = routed_rows < routed_limit
            valid_after_sentinel = valid_rows & ((~valid_rows).cumsum(dim=1) > 0)
            valid_route_ids = active_ids[
                (active_ids >= 0) & (active_ids < routed_limit)
            ]
            if valid_route_ids.numel() > 1:
                sorted_route_ids = torch.sort(valid_route_ids).values
                route_ids_are_unique = (
                    sorted_route_ids[1:] != sorted_route_ids[:-1]
                ).all()
            else:
                route_ids_are_unique = torch.ones(
                    (), dtype=torch.bool, device=active_ids.device
                )
            (
                ids_are_nonnegative,
                block_prefixes_are_valid,
                route_ids_are_unique,
                experts_are_nonnegative,
                experts_are_in_range,
            ) = (
                torch.stack(
                    (
                        (active_ids >= 0).all(),
                        ~valid_after_sentinel.any(),
                        route_ids_are_unique,
                        (active_experts >= 0).all(),
                        (active_experts < num_experts).all(),
                    )
                )
                .cpu()
                .tolist()
            )
        else:
            ids_are_nonnegative = True
            block_prefixes_are_valid = True
            route_ids_are_unique = True
            experts_are_nonnegative = True
            experts_are_in_range = True

        if not ids_are_nonnegative:
            raise ValueError("sorted_ids values must be non-negative")
        if not block_prefixes_are_valid:
            raise ValueError(
                "each active sorted_ids block must contain valid routed ids as a "
                "contiguous prefix followed only by sentinel values"
            )
        if not experts_are_nonnegative or not experts_are_in_range:
            if active_experts.numel() == 0:
                minimum_expert = maximum_expert = 0
            else:
                minimum_expert, maximum_expert = torch.aminmax(active_experts)
            minimum_expert, maximum_expert = (
                torch.stack((minimum_expert, maximum_expert)).cpu().tolist()
            )
            raise ValueError(
                f"expert_ids values must be in [0, {num_experts - 1}], got "
                f"[{minimum_expert}, {maximum_expert}]"
            )
        if not route_ids_are_unique:
            raise ValueError("active sorted_ids contains duplicate routed ids")
    block_n = kernel_config.block_n
    if block_n <= 0 or shape_n % block_n:
        raise ValueError(f"config block-N={block_n} must divide shape_n={shape_n}")

    with torch.cuda.device(inputs.device):
        kernel = _get_indexed_kernel(
            inputs.get_device(),
            shape_n,
            shape_k,
            num_experts,
            layout,
            kernel_config,
        )
        return launch_kernel(
            kernel_id=kernel.kernel_id,
            inputs=inputs,
            weight=packed_weight,
            weight_scale=packed_scale,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            outputs=outputs,
            valid_shape_m=valid_shape_m,
        )


__all__ = [
    "IndexedKernelConfig",
    "pack_w4a16",
    "w4a16_indexed",
]
