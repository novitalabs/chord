# Getting started

The JIT cache, the low-level API, the routing contract, and the layer adapter
for framework integration. Installation is the one-line `pip install` in the
README.

## JIT cache

Cubins from the first JIT, the NVRTC helper and launcher build files default to
`.chord_cache/` beside the source or installed package, with lock files in
`.chord_tmp/`; they are not shared with an upstream Humming `~/.humming` cache.
Read-only install directories and multi-process deployments can set
`CHORD_CACHE_DIR` and `CHORD_TMP_DIR` explicitly.

## Low-level API

Logical weights hold INT4 values as unsigned codes `[0, 15]` in `int32`; with no
explicit zero point the kernel decodes them as `code - 8`, giving the signed range
`[-8, 7]`. Weight and scale are packed for the target layout first:

```python
import torch

from chord_kernels import indexed
from chord_kernels.operator import pack_w4a16

weight = torch.randint(
    0, 16, (num_experts, n, k), dtype=torch.int32, device="cuda"
).contiguous()
scale = torch.ones(
    (num_experts, n, k // 32), dtype=torch.bfloat16, device="cuda"
).contiguous()
packed = pack_w4a16(weight, scale, layout="mma")

output = indexed(
    inputs, packed, sorted_ids, expert_ids, num_tokens_padded, top_k
)
```

`inputs` is a contiguous CUDA BF16 `[M, K]` tensor; `packed` carries the physical
layout and `[E, N, K]` metadata. `N` must be a multiple of 128 and `K` a multiple
of 64. Use `layout="mma"` for the decode profiles and `layout="wgmma"` for H200
prefill; the two packings are not interchangeable. When calling the low-level API
directly, `config` can override the profile's tile; the MMA shorthand adopts the
decode swap-AB default when `swap_ab` is omitted, and block-N defaults to the
largest compatible power of two between 128 and 256.

The returned tensor is BF16 with shape `[M * top_k, N]`. Callers may pass `outputs`
to reuse a buffer; `valid_shape_m` only selects the tuning bracket on the layer
path and has no kernel-level effect. EP subset
routing does not write output rows belonging to remote experts; a host that needs a
complete output should zero the buffer it passes in, or keep to its own workspace
initialization convention.

The `gate_up` and `down` projections and how routing metadata maps onto `top_k` are
described in [shapes.md](shapes.md).

### Routing contract

- `sorted_ids`, `expert_ids` and `num_tokens_padded` must be contiguous CUDA
  `int32`. `num_tokens_padded` accepts both a 0-D scalar and the `[1]` tensor vLLM
  uses.
- `sorted_ids` and `expert_ids` may be the over-allocated buffers returned by
  vLLM's `moe_align_block_size`. `num_tokens_padded` gives the valid prefix length,
  and only the first `num_tokens_padded / block_m` blocks are read. The capacity
  need not be a multiple of `block_m`; `expert_ids` must cover at least the number
  of valid blocks, and the capacity tail is never read.
- Valid route ids live in `[0, M * top_k)` and each may appear only once. Every
  valid block must hold its valid ids as a contiguous prefix followed only by
  sentinels `>= M * top_k`. EP routing may carry only the current rank's subset of
  routes; sentinels and `-1` expert ids in the capacity tail are ignored.
- `validate_routing=True` (the default) synchronizes with CUDA to check the count
  bound, block alignment, route prefixes and expert ids. When routing comes from a
  trusted producer such as vLLM, pass `validate_routing=False`; that fast path
  checks only tensor shape, dtype and device, never reads `num_tokens_padded` back
  to the host, and is therefore usable under CUDA Graph capture. The caller then
  guarantees the count is non-negative, `block_m`-aligned, and within the
  `sorted_ids`/`expert_ids` capacity.

## Layer adapter

`chord_kernels.operator.layer` is a thin wrapper independent of any model weight
format, so a framework such as vLLM can hold the packed weight and reuse the
kernel:

```python
from chord_kernels.operator.layer import IndexedW4A16Layer

layer = IndexedW4A16Layer(
    num_experts=num_experts,
    shape_n=n,
    shape_k=k,
    profile="h200_decode_ep8",
)
layer.load_weight(weight, scale)
output = layer(
    inputs, sorted_ids, expert_ids, num_tokens_padded, top_k
)
```

The layer builds on `IndexedLayerProfile`, `IndexedLayerMeta` and
`IndexedW4A16Method`. `load_weight` accepts unpacked INT4 weight and scale; for the
checkpoint form that carries eight nibbles per `int32`, pass `packed=True` (inferred
from the last dimension by default). CPU and meta loading keeps that compact `K/8`
form, and the first CUDA transform repacks straight from the checkpoint layout
rather than materializing an eight-times-larger `[E, N, K]` temporary.

The forward routing arguments match the low-level API exactly. The layer assumes
routing comes from a trusted upstream aligner and disables the synchronizing check,
so the hot path never reads the CUDA count and can be captured into a CUDA Graph;
pass `validate_routing=True` explicitly when debugging or calling it standalone.
Inputs must be BF16 — an explicit input scale, zero point, Hadamard rotation or a
grouped `expert_layout` is rejected.

For framework integration the module also exposes a narrow adapter:
`prepare_layer_meta` preserves `shape_n`, `shape_k`, the expert count and the
`w13_`/`w2_` tensor names, and `get_default_tuning_configs` returns the
`(min_m, max_m, config)` table swept from the same resolver the forward path
uses, so a row's schedule is exact for every routed-M inside it. vLLM generates
`sorted_ids`/`expert_ids` once from the w13 table; the two projections are
tuned independently, so where the w2 row's block-M differs from w13's, passing
the shared routing block-M through the `block_m` forward argument keeps the
launch aligned with the w13 routing while the w2 projection keeps its own N/K
tile.

This is a single-projection weight and operator adapter, not a complete MoE plugin
with a router, activation and `w13+w2` lifecycle. It registers no vLLM quantization
method and owns no vLLM router: the host framework still converts its own weight
format and routing metadata to the contract above, and selects
`profile="h200_prefill_ep8"`, `"h200_decode_ep8"` or `"blackwell_decode_ep8"`.

### Selecting the P/D role per serving instance

A disaggregated deployment launches prefill and decode instances from the same
code path, so the instance role usually cannot be a Python argument. Mirroring
upstream Humming's `HUMMING_INT_SM90_DECODE` (default off), the layer reads
`CHORD_SM90_DECODE` when a profile is left at `"auto"` on SM90:

| `CHORD_SM90_DECODE` | Resolved SM90 profile |
| --- | --- |
| `1` | `h200_decode_ep8` |
| `0` or unset | `h200_prefill_ep8` |

An explicit `profile=...` or `mode=...` argument always wins over the variable.
Because the profile fixes the physical weight layout, the variable is read when
the profile is resolved and must be set before model load; it cannot switch an
already packed layer at runtime. Blackwell publishes only the decode profile,
so `profile="auto"` resolves to `blackwell_decode_ep8` there and the variable
is ignored.

## Tests and benchmarks

```bash
python tests/test_w4a16.py                 # run every case, print a perf table
python -m pytest -m "not gpu" tests/       # fast CPU-only contract tests
```

Shapes live in `tests/generators.py` (`PERFORMANCE_CASES`); each case is checked
against a plain-PyTorch reference and then timed on the same launch, so a
deployment can reproduce both the correctness and the performance numbers for
its own GPU with one command. The timing methods, the `cos_diff` accuracy
metric, and how to read the throughput columns are in
[benchmarking.md](benchmarking.md). The production shapes and token sweeps come
from [shapes.md](shapes.md).
