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
`profile="h200_prefill_ep8"`, `"h200_tp8"`, `"h200_decode_ep8"` or
`"blackwell_decode_ep8"`.

### Selecting the shard axis

The four profiles cover two different 8-way shardings. EP8 splits the 384 routed
experts across ranks (48 each) and leaves every expert whole; TP8 keeps all 384 on
every rank and slices `moe_intermediate`, narrowing gate/up to `N=512` and down to
`K=256`. The per-expert shapes and tuning tables therefore differ — see
[shapes.md](shapes.md).

The axis is not a property of the device, so it is never guessed from the GPU.
It is recovered from the projection shapes, which are distinct per axis: TP8's
`(512, 7168)` and `(7168, 256)` against EP8's `(4096, 7168)` and `(7168, 2048)`.
A framework adapter therefore reaches TP8 by passing the `shape_n`/`shape_k` it
always passes, with no chord-specific argument. A shape this operator publishes
no tuned schedule for infers nothing and keeps the EP8 default.

To be explicit, ask for the profile by name or pass `tensor_parallel_size=8`
(which also selects TP8 for a model whose shapes are not in the table):

```python
layer = IndexedW4A16Layer(
    num_experts=384,
    shape_n=512,
    shape_k=7168,
    profile="h200_tp8",
)
```

The axes also imply different deployments. The EP8 profiles are disaggregated P/D
roles, each packing its own layout; TP8 targets a single instance (`mode="mix"`)
serving both phases from one packed weight, so it takes no role and ignores both
`CHORD_SM90_DECODE` and `CHORD_USE_GROUPED` (each grouped kernel serves one
phase, so none can back a mix weight).

If `tensor_parallel_size` contradicts a published shape — say `8` alongside EP8's
`(4096, 7168)` — that is rejected rather than resolved, since the two disagree
about which tuned table applies.

### Selecting the P/D role per serving instance

A disaggregated deployment launches prefill and decode instances from the same
code path, so the instance role usually cannot be a Python argument. The layer
reads `CHORD_SM90_DECODE` (default off) when a profile is left at `"auto"` on
SM90:

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

The table covers the EP8 axis only; `h200_tp8` takes no role, as above.

### DeepGEMM-layout SM90 backend (`masked` / `contiguous`)

A second, independent weight layout serves grouped GEMMs on Hopper via the
vendored DeepGEMM W4A16 kernel (TMA + persistent WGMMA with warp-specialized
producer). Two profiles select it explicitly, and both accept EP8/EP16/EP32
shards:

| Profile | Forward convention | Weight buffer |
| --- | --- | --- |
| `h200_grouped_decode` | masked decode: inputs `[G*max_m, K]`, routing `expert_layout` = per-expert valid counts `[G] int32`, `valid_shape_m` = total routed tokens | INT4 bit-permuted at BLOCK_K=128 |
| `h200_grouped_prefill` | contiguous prefill: inputs `[m, K]` with per-expert rows padded to a 128-row boundary (padding zeroed), routing `m_indices` `[m] int32` (`-1` marks padding) | the same reorder at BLOCK_K=64 |

The packed buffer carries its mode and the two layouts are NOT interchangeable
(the reorder perm width is baked in); dispatch validates `compute_config`'s
`gemm_type` (`grouped_masked` / `grouped_contiguous`) against the packed mode
and rejects a mismatch instead of mis-computing. Tile selection is owned by
the ported DeepGEMM SM90 heuristic per call, so the layer's `block_m` and
tuning rows do not apply to this backend. At the operator level the same paths
are exposed as `chord_kernels.masked` / `chord_kernels.contiguous` around
`pack_w4a16_grouped(...)`.

Setting `CHORD_USE_GROUPED=1` before model load makes `profile="auto"` resolve
the SM90 prefill/decode roles to the grouped pair; `CHORD_SM90_DECODE` then
picks which of the two (decode -> masked, prefill -> contiguous). Both are read
before the weight is packed, so both must be set before model load.
`CHORD_W4A16_BM/BN/BK/CM/CN/STAGES` pin a single forced layout for tuning
experiments.

## vLLM integration through the `humming` import root

The distribution ships two import surfaces over the same operator:

- `chord` — the humming-compatible facade (`chord.{dtypes,config,layer,ops}`)
  for adapters written against chord directly.
- `humming` — the upstream import root itself, for frameworks whose integration
  hardcodes the package name. vLLM's lazy facade (`vllm/utils/humming.py`)
  resolves fixed `humming.{dtypes,config,layer,schema,utils.weight}` module
  paths and gates on `find_spec("humming")`; installing `chord_kernels` makes
  both resolve to this repository with no framework change. Do not install
  upstream `inclusionAI/humming` alongside it — the `humming` name is
  claimed by design.

Under the `humming` root the shimming scope is exactly the indexed W4A16 MoE
contract vLLM consumes:

- `humming.layer.HummingMethod` dispatches to `IndexedW4A16Method`; foreign
  host layers are prepared through the `humming_metas` +
  `w13_weight`/`w2_weight` naming convention vLLM already uses. The published
  w2 tuning rows have their M tile re-mapped onto the w13 routing block-M
  (each row keeps its own N/K tile and stream-K choice); this bakes in the
  same adjustment that an explicit `block_m` forward argument performs, so a
  shared `moe_align_block_size` routing stays correct on vLLM's existing call
  shape.
- `humming.schema` implements `HummingWeightSchema` (uint4 + group-32 +
  BF16 scale), the BF16-passthrough `HummingInputSchema`, and a
  compressed-tensors **pack-quantized INT4 group-32** weight schema (the
  checkpoint format vLLM's CT-quantized MoE models ship, e.g. Kimi K2.x), so
  both entry points in vLLM — the WNA16 MoE backend oracle and
  `--quantization humming` — load with no framework change. Every other
  schema name vLLM may import (AWQ/GPTQ/MXFP4/NVFP4/FP8/modelopt/AutoRound/
  Bitnet, online `quantize_weight`, dense GEMM) exists but raises
  `NotImplementedError`, so unsupported quantizations fail closed at load.
  The weight-scale-2 hierarchy (`weight_scale_2_type`) and block/token scale
  types are out of scope.
- `humming.config` provides the `GemmType`/`WeightScaleType` enums. Only
  `GemmType.INDEXED` resolves to a working backend here; the grouped members
  exist so class references work and selecting them raises during schedule
  validation.
- Profile resolution needs no chord-specific argument from the framework:
  the shard axis is recovered from the published projection shapes (see
  *Selecting the shard axis*), and the SM90 P/D role follows
  `CHORD_SM90_DECODE`.

On the vLLM side two things apply. First, the humming MoE experts must admit
group-32 INT4 through `HummingExpertsBase._supports_quant_scheme`; upstream
branches older than the current WNA16 generalization (see vLLM PR #48918,
which admits unsigned-integer WNA16 group scales generically) may need the
group-32 keys added explicitly. Second, the Humming backend is selected with
`moe_backend="humming"` (or `--quantization humming` for the schema route),
since the automatic WNA16 priority order tries other backends first. Keep
`VLLM_HUMMING_MOE_GEMM_TYPE` at its default indexed behavior and leave
`VLLM_HUMMING_USE_F16_ACCUM` / `VLLM_BATCH_INVARIANT` off — the indexed
kernel rejects those compute options. No environment variable is needed for
TP8: `profile='auto'` recovers the shard axis from the projection shapes and
lands on `h200_tp8`.

## Tests and benchmarks

```bash
python tests/test_w4a16_indexed.py    # indexed cases, print a perf table
python tests/test_w4a16_grouped.py    # masked + contiguous cases, same table style
python -m pytest -m "not gpu" tests/  # fast CPU-only contract tests
python -m pytest tests/               # everything except the perf-sized cases
```

Shapes live in `tests/generators.py` (`PERFORMANCE_CASES`); each case is checked
against a plain-PyTorch reference and then timed on the same launch, so a
deployment can reproduce both the correctness and the performance numbers for
its own GPU with one command. The timing methods, the `cos_diff` accuracy
metric, and how to read the throughput columns are in
[benchmarking.md](benchmarking.md). The production shapes and token sweeps come
from [shapes.md](shapes.md).
