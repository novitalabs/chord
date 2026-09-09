# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

from chord_kernels.operator.api import (
    IndexedKernelConfig,
    w4a16_indexed,
)
from chord_kernels.operator.dispatch import (
    backend_profile_name,
    build_prepared,
    forward_w4a16,
    pack_weight,
    transformed_shapes,
)
from chord_kernels.operator.env import (
    IndexedMode,
    indexed_mode_from_env,
    resolve_backend_name,
    use_grouped_from_env,
)
from chord_kernels.operator.grouped import (
    GroupedPreparedWeight,
    W4A16GemmConfig,
    W4A16GemmDesc,
    pack_w4a16_grouped,
    select_w4a16_config,
)
from chord_kernels.operator.layer import (
    IndexedW4A16Layer,
    IndexedW4A16Method,
    unpack_packed_uint4,
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
    IndexedBackend,
    IndexedLayerMeta,
    IndexedLayerProfile,
    select_indexed_profile,
)

__all__ = [
    "BLACKWELL_DECODE_EP8",
    "H200_DECODE_EP8",
    "H200_GROUPED_DECODE",
    "H200_GROUPED_PREFILL",
    "H200_PREFILL_EP8",
    "H200_TP8",
    "INDEXED_PROFILES",
    "GroupedPreparedWeight",
    "IndexedBackend",
    "IndexedKernelConfig",
    "IndexedLayerMeta",
    "IndexedLayerProfile",
    "IndexedMode",
    "IndexedW4A16Layer",
    "IndexedW4A16Method",
    "PreparedWeight",
    "W4A16GemmConfig",
    "W4A16GemmDesc",
    "WeightLayout",
    "backend_profile_name",
    "build_prepared",
    "forward_w4a16",
    "indexed_mode_from_env",
    "pack_w4a16",
    "pack_w4a16_grouped",
    "pack_weight",
    "resolve_backend_name",
    "select_indexed_profile",
    "select_w4a16_config",
    "transformed_shapes",
    "unpack_packed_uint4",
    "use_grouped_from_env",
    "w4a16_indexed",
]
