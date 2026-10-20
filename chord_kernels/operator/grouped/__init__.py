# Derived from deepseek-ai/DeepGEMM; modified for chord_kernels.
"""grouped W4A16 integration for chord.

This subpackage vendors DeepGEMM's SM90 W4A16 (BF16 activation x INT4 group-32
weight) persistent WGMMA kernel and hosts the Python-side support around it:
the SM90 launch-config heuristics, the JIT kernel wrapper and the
masked/contiguous operator entry points.  The weight is ``pack_w4a16``'s.
"""

from chord_kernels.operator.grouped.heuristics import (
    W4A16GemmConfig,
    W4A16GemmDesc,
    select_w4a16_config,
)
from chord_kernels.operator.grouped.api import (
    w4a16_contiguous,
    w4a16_masked,
)

__all__ = [
    "W4A16GemmConfig",
    "W4A16GemmDesc",
    "select_w4a16_config",
    "w4a16_contiguous",
    "w4a16_masked",
]
