# Derived from inclusionAI/humming; modified for chord_kernels.

import dataclasses
import functools

import torch

from chord_kernels.operator import cpp_jit
from chord_kernels.operator.kernel.humming import HummingKernel
from chord_kernels.operator.ops import launch_kernel
from chord_kernels.operator.packing import (
    PreparedWeight,
    _validate_supported_architecture,
)


def _default_stream_k_fp32(use_stream_k: bool, warp_shape: tuple[int, int, int]) -> bool:
    """FP32 stream-K reduction for every stream-K config, MMA and WGMMA."""
    del warp_shape
    return use_stream_k


@dataclasses.dataclass(frozen=True)
class IndexedKernelConfig:
    """Launch description for the indexed W4A16 kernel (``tuning.py`` picks it)."""

    block_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]
    num_stages: int = 4
    num_ctas_per_sm: int = 1
    swap_ab: bool = False
    use_stream_k: bool = False
    # Reduce stream-K partials in an FP32 workspace (one BF16 rounding per
    # output, deterministic) instead of BF16 adds into the output.  ``None``
    # resolves to FP32 for every stream-K config the kernel implements it for
    # (see _default_stream_k_fp32); ``False`` forces the BF16 reduction.
    stream_k_fp32: bool | None = None

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
        if self.stream_k_fp32 is None:
            object.__setattr__(
                self, "stream_k_fp32",
                _default_stream_k_fp32(self.use_stream_k, self.warp_shape),
            )
        if not isinstance(self.stream_k_fp32, bool):
            raise TypeError("stream_k_fp32 must be a bool or None")
        # A 1024-word lock buffer covers ~1.1x the grid in stream-K tiles, so
        # 4 CTAs/SM on a 148-SM part is already the ceiling.  The launcher
        # re-checks the exact grid against the caller's buffer at launch;
        # rejecting the config here fails earlier.
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
        layout: str,
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
            stream_k_fp32=value.get("stream_k_fp32"),
        )

    def to_dict(self) -> dict:
        row = {
            "block_shape": self.block_shape,
            "warp_shape": self.warp_shape,
            "num_stages": self.num_stages,
            "num_ctas_per_sm": self.num_ctas_per_sm,
            "swap_ab": self.swap_ab,
            "use_stream_k": self.use_stream_k,
        }
        # Only emitted when it differs from the layout default, so the published
        # rows serialize as before and an explicit override survives a round trip.
        if self.stream_k_fp32 != _default_stream_k_fp32(self.use_stream_k, self.warp_shape):
            row["stream_k_fp32"] = self.stream_k_fp32
        return row


def _compatible_block_n(shape_n: int) -> int:
    candidate = 1 << min(shape_n, 256).bit_length() - 1
    while candidate > 1 and shape_n % candidate:
        candidate //= 2
    if candidate < 128:
        raise ValueError(
            f"shape_n must have a compatible power-of-two tile >= 128, got {shape_n}"
        )
    return candidate


def _config_layout(config: IndexedKernelConfig) -> str:
    """The MMA family a config launches: warp-N 32 is WGMMA, 64 is mma.sync."""
    return "wgmma" if config.warp_shape[1] == 32 else "mma"


@functools.lru_cache(maxsize=128)
def _get_indexed_kernel(
    device_index: int,
    shape_n: int,
    shape_k: int,
    num_experts: int,
    config: IndexedKernelConfig,
) -> HummingKernel:
    block_m, block_n, block_k = config.block_shape
    swap_ab = config.swap_ab
    # The packed bytes are the same for both MMA families; the config's warp-N
    # picks the instruction (32 = WGMMA, 64 = mma.sync).
    layout = _config_layout(config)
    if swap_ab and layout != "mma":
        raise ValueError("swap_ab requires the MMA kernel (warp-N 64)")
    if layout == "mma" and not swap_ab and config.block_m % 16 == 8:
        raise ValueError(
            "MMA indexed configs with block_m=8 (or another 16q+8 tile) "
            "must set swap_ab=True"
        )

    if block_k <= 0 or block_k % 64:
        raise ValueError(
            f"indexed block-K must be a positive multiple of 64, got {block_k}"
        )

    mma_type = layout
    if mma_type == "wgmma" and block_n < 128:
        raise ValueError("the WGMMA kernel requires shape_n/block_n >= 128")

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
            stream_k_fp32=config.stream_k_fp32,
        )


def _check_cuda_int32(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous and on {device}")


def _validate_prepared_weight(
    weight: PreparedWeight, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    if not isinstance(weight, PreparedWeight):
        raise TypeError("weight must be a PreparedWeight from pack_w4a16_indexed()")
    n, k, e = weight.n, weight.k, weight.num_experts
    packed, scale = weight.packed, weight.scale
    if packed.dtype != torch.int32 or tuple(packed.shape) != (e, k // 16, n * 2):
        raise ValueError(f"packed weight must be int32 {(e, k // 16, n * 2)}, got {tuple(packed.shape)}")
    if scale.dtype != torch.bfloat16 or tuple(scale.shape) != (e, k // 32, n):
        raise ValueError(f"packed scale must be BF16 {(e, k // 32, n)}, got {tuple(scale.shape)}")
    if packed.device != device or scale.device != device:
        raise ValueError(f"weight must be on {device}, got {packed.device}")
    if not packed.is_contiguous() or not scale.is_contiguous():
        raise ValueError("packed weight and scale must be contiguous")
    return packed, scale, e, n, k


def _validate_routing(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    block_m: int,
    routed_limit: int,
    num_experts: int,
) -> None:
    """Synchronizing check of the routing contract (``validate_routing=True``)."""
    padded_count = int(num_tokens_padded.reshape(-1)[0].item())
    if padded_count < 0 or padded_count > sorted_ids.numel():
        raise ValueError(
            f"num_tokens_padded must be in [0, {sorted_ids.numel()}], got {padded_count}"
        )
    if padded_count % block_m:
        raise ValueError(
            f"num_tokens_padded must be a multiple of block_m={block_m}, got {padded_count}"
        )
    active_blocks = padded_count // block_m
    if expert_ids.numel() < active_blocks:
        raise ValueError(
            f"expert_ids must cover {active_blocks} active blocks, got {expert_ids.numel()}"
        )
    if not padded_count:
        return
    ids = sorted_ids[:padded_count]
    experts = expert_ids[:active_blocks]
    valid = ids.view(-1, block_m) < routed_limit
    route_ids = torch.sort(ids[(ids >= 0) & (ids < routed_limit)]).values
    unique = (route_ids[1:] != route_ids[:-1]).all() if route_ids.numel() > 1 else ids.new_ones((), dtype=torch.bool)
    nonneg, prefix, unique, in_range = torch.stack((
        (ids >= 0).all(),
        ~(valid & ((~valid).cumsum(dim=1) > 0)).any(),
        unique,
        ((experts >= 0) & (experts < num_experts)).all(),
    )).cpu().tolist()
    if not nonneg:
        raise ValueError("sorted_ids values must be non-negative")
    if not prefix:
        raise ValueError(
            "each active sorted_ids block must hold its routed ids as a prefix "
            "followed only by sentinels"
        )
    if not in_range:
        raise ValueError(f"expert_ids values must be in [0, {num_experts - 1}]")
    if not unique:
        raise ValueError("active sorted_ids contains duplicate routed ids")


def indexed_block_m(
    weight: PreparedWeight, num_tokens: int, top_k: int, *, device=None
) -> int:
    """Routing block size ``indexed`` uses; align ``moe_align_block_size`` to it.

    For down pass gate/up's ``top_k`` here (the kernel call takes ``top_k=1``).
    """
    from chord_kernels.operator.tuning import select_indexed_config

    if not isinstance(weight, PreparedWeight):
        raise TypeError("weight must be a PreparedWeight from pack_w4a16_indexed()")
    if device is None:
        device = weight.packed.device
    return select_indexed_config(
        weight.n, weight.k, weight.num_experts, num_tokens * top_k, device=device
    ).block_m


def w4a16_indexed(
    inputs: torch.Tensor,
    weight: PreparedWeight,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    top_k: int,
    *,
    locks: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    block_m: int | None = None,
    validate_routing: bool = False,
    config: IndexedKernelConfig | None = None,
) -> torch.Tensor:
    """BF16 x INT4 group-32 indexed MoE GEMM; see :func:`chord_kernels.indexed`.

    ``config`` overrides the tuned schedule (tests and tuning sweeps only).
    """
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2 or inputs.dtype != torch.bfloat16:
        raise TypeError("inputs must be a 2D torch.bfloat16 tensor")
    if not inputs.is_cuda or not inputs.is_contiguous() or 0 in inputs.shape:
        raise ValueError("inputs must be a non-empty contiguous CUDA tensor")
    device = inputs.device
    _validate_supported_architecture(device)
    packed_weight, packed_scale, num_experts, shape_n, shape_k = _validate_prepared_weight(
        weight, device
    )
    if inputs.size(1) != shape_k:
        raise ValueError(f"inputs K={inputs.size(1)} does not match the weight's K={shape_k}")
    for name, tensor in (("sorted_ids", sorted_ids), ("expert_ids", expert_ids),
                         ("num_tokens_padded", num_tokens_padded)):
        _check_cuda_int32(name, tensor, device)
    if sorted_ids.ndim != 1 or expert_ids.ndim != 1:
        raise ValueError("sorted_ids and expert_ids must be 1D")
    if num_tokens_padded.numel() != 1 or num_tokens_padded.ndim > 1:
        raise ValueError("num_tokens_padded must be a scalar or a one-element tensor")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")

    if config is None:
        from chord_kernels.operator.tuning import select_indexed_config

        config = select_indexed_config(
            shape_n, shape_k, num_experts, inputs.size(0) * top_k,
            block_m=block_m, device=device,
        )
    elif not isinstance(config, IndexedKernelConfig):
        raise TypeError(f"config must be an IndexedKernelConfig, got {type(config).__name__}")
    elif block_m is not None and block_m != config.block_m:
        raise ValueError("block_m conflicts with config")
    if shape_n % config.block_n:
        raise ValueError(f"config block-N={config.block_n} must divide N={shape_n}")

    if config.use_stream_k:
        if locks is None:
            raise ValueError(
                "this call uses stream-K and needs locks: "
                "torch.zeros(1024, dtype=torch.int32, device=...), allocated once"
            )
        _check_cuda_int32("locks", locks, device)
        if locks.ndim != 1:
            raise ValueError("locks must be 1D")
    else:
        locks = None
    if validate_routing:
        _validate_routing(sorted_ids, expert_ids, num_tokens_padded, config.block_m,
                          inputs.size(0) * top_k, num_experts)

    with torch.cuda.device(device):
        if cpp_jit.enabled(device):
            kernel_id = cpp_jit.get_indexed_kernel(
                inputs.get_device(), shape_n, shape_k, num_experts, config
            )
            return cpp_jit.indexed_gemm(
                kernel_id, inputs, packed_weight, packed_scale, sorted_ids,
                expert_ids, num_tokens_padded, top_k, outputs, locks,
            )
        kernel = _get_indexed_kernel(
            inputs.get_device(), shape_n, shape_k, num_experts, config
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
            locks=locks,
        )


__all__ = ["IndexedKernelConfig", "indexed_block_m", "w4a16_indexed"]
