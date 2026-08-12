"""Humming-name aliases for the dtype descriptors.

The extracted operator has one fixed schema (BF16 activation, unsigned INT4
weight, BF16 group-32 scale); :class:`DataType` carries just enough structure
for framework schema checks.  ``torch_dtype_map`` covers the dtypes this
operator can actually produce.
"""

from __future__ import annotations

import torch

from chord_kernels.operator.dtypes import DataType, bfloat16, uint4

torch_dtype_map = {bfloat16: torch.bfloat16}

__all__ = ["DataType", "bfloat16", "torch_dtype_map", "uint4"]
