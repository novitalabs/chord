<!-- markdownlint-disable MD001 MD033 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/novitalabs/chord/main/docs/assets/logo_lockup_dark.svg">
    <img alt="Chord" src="https://raw.githubusercontent.com/novitalabs/chord/main/docs/assets/logo_lockup.svg" width="240">
  </picture>
</p>

<h3 align="center">
Novita Labs' production MoE CUDA kernel
</h3>

<p align="center">
| <a href="#documentation"><b>Documentation</b></a> | <a href="https://novita.ai"><b>Novita AI</b></a> | <a href="https://blogs.novita.ai"><b>Blog</b></a> |
</p>

---

The `chord` repository publishes the Python package `chord_kernels`: Novita Labs'
in-house W4A16 MoE CUDA operator — BF16 activation, INT4 weight (stored as
unsigned nibbles, decoded as `code - 8`), group-32 scale — through the `indexed`
interface, plus a thin layer adapter for inference framework integration. The
kernel runs in Novita's production inference service; the CUDA template directory
keeps only the minimal dependency closure it needs at runtime.

The operator is open-sourced interface by interface. This release publishes the
`indexed` interface, which fits single-node deployments: its routing metadata
(sorted ids, expert ids) addresses one node's local experts directly. The
`masked` and `contiguous` grouped-GEMM interfaces — the layouts wide-EP,
P/D-disaggregated deployments are built around — follow in a future release.

## Where the name comes from

In music theory a chord is two or more notes of different pitch sounded together.
The name points at how this operator runs: a gathered BF16 activation, an INT4
weight and a group-32 scale are struck at once in one fused GEMM, each keeping its
own pitch.

## Supported configurations

| GPU | Compute capability | Scenario | Profile |
| --- | --- | --- | --- |
| Hopper (H200) | SM90 (9.0) | Prefill, EP8 | `h200_prefill_ep8` |
| Hopper (H200) | SM90 (9.0) | Decode, EP8 | `h200_decode_ep8` |
| Blackwell (B200/B300) | B200: SM100 (10.0); B300: SM103 (10.3) | Decode, EP8 | `blackwell_decode_ep8` |

All three parts are compiled, run and performance-verified.

Each profile fixes a tensor-core instruction family and the physical weight layout
that goes with it, so the profile must be chosen when the weight is packed and
cannot be switched at runtime; a mismatched layout is rejected before launch. A
production deployment prepares the matching profile per P/D instance, or keeps two
packed copies of the weight. The scheduling details (WGMMA vs MMA, swap-AB,
block-M model, stream-K) are in [docs/tuning.md](docs/tuning.md).

`tests/test_w4a16.py` checks every tuning row against a plain-PyTorch reference
on the running device before timing it, so a deployment can reproduce both the
correctness and the performance numbers for its own GPU with one command.

`blackwell_decode_ep8` is the Blackwell decode profile name. B200/SM100 and
B300/SM103 share one schedule, but the JIT targets `sm_100a` and `sm_103a`
separately per actual compute capability and does not reuse a cubin across them.

## Measured performance

`humming` is the public Humming `indexed` path, `chord` is this repository's profile
for that scenario; both are timed on the same GPU at the same shape and the same
routing draw. Times are per-call microseconds, lower is better. `gate_up + down` is
the speedup of the two stages summed, which is what one MoE layer actually pays.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/novitalabs/chord/main/docs/assets/benchmark_chart_dark.svg">
  <img src="https://raw.githubusercontent.com/novitalabs/chord/main/docs/assets/benchmark_chart.svg" alt="Per-call latency versus token count for the public Humming baseline and this repository across the three supported scenarios, lower is better">
</picture>

*Per-call latency from the tables below; each panel annotates the `gate_up + down`
layer speedup range. Regenerate after re-measuring with
`python docs/assets/benchmark_chart.py`.*

### H200 EP8 prefill (`h200_prefill_ep8`)

| Stage | num_tokens_total | humming µs | humming TFLOPS | chord µs | chord TFLOPS | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 1024 | 375.0 | 160.36 | 313.5 | 192 | 1.20 | 1.20 |
| gate_up | 2048 | 466.3 | 257.92 | 383.4 | 314 | 1.22 | 1.19 |
| gate_up | 4096 | 608.4 | 395.34 | 545.8 | 441 | 1.11 | 1.13 |
| gate_up | 8196 | 1085.7 | 443.08 | 997.7 | 482 | 1.09 | 1.11 |
| gate_up | 16384 | 1903.0 | 505.55 | 1743.1 | 552 | 1.09 | 1.11 |
| down | 1024 | 195.9 | 153.43 | 162.9 | 185 | 1.20 | |
| down | 2048 | 235.1 | 255.81 | 204.5 | 294 | 1.15 | |
| down | 4096 | 337.4 | 356.40 | 293.2 | 410 | 1.15 | |
| down | 8196 | 606.4 | 396.64 | 533.4 | 451 | 1.14 | |
| down | 16384 | 1059.4 | 454.07 | 937.3 | 513 | 1.13 | |

### H200 EP8 decode (`h200_decode_ep8`)

Tokens per GPU is `bs * (mtp + 1)`.

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 267.6 | 2845.08 | 221.9 | 3431 | 1.21 | 1.24 |
| gate_up | 30 | 281.3 | 2826.64 | 245.6 | 3237 | 1.15 | 1.20 |
| gate_up | 40 | 284.0 | 2802.95 | 253.7 | 3137 | 1.12 | 1.16 |
| gate_up | 50 | 296.4 | 2687.97 | 259.9 | 3066 | 1.14 | 1.17 |
| down | 20 | 146.2 | 2618.17 | 111.9 | 3421 | 1.31 | |
| down | 30 | 152.2 | 2633.29 | 116.8 | 3432 | 1.30 | |
| down | 40 | 153.3 | 2624.30 | 123.0 | 3271 | 1.25 | |
| down | 50 | 156.3 | 2582.34 | 126.4 | 3195 | 1.24 | |

### B300 EP8 decode (`blackwell_decode_ep8`)

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 319.1 | 2385.78 | 146.0 | 5215 | 2.19 | 2.15 |
| gate_up | 30 | 320.4 | 2481.51 | 162.3 | 4900 | 1.97 | 1.98 |
| gate_up | 40 | 320.6 | 2482.68 | 175.2 | 4544 | 1.83 | 1.88 |
| gate_up | 50 | 320.9 | 2482.44 | 182.9 | 4355 | 1.75 | 1.81 |
| down | 20 | 174.8 | 2190.34 | 83.8 | 4565 | 2.09 | |
| down | 30 | 181.3 | 2211.10 | 91.3 | 4391 | 1.99 | |
| down | 40 | 181.4 | 2217.48 | 91.9 | 4379 | 1.97 | |
| down | 50 | 181.7 | 2221.51 | 95.1 | 4245 | 1.91 | |

The Blackwell speedups are larger than the H200 ones mostly because of the baseline,
not because the Blackwell schedule is better tuned than the Hopper one. Public
Humming has no SM100/SM103 heuristics and falls back to its SM80 rules there, so its
B300 numbers are an untuned reference point — note how its time barely moves from 20
to 50 tokens per GPU, and how it reads roughly the same GB/s on B300 as on H200
despite the wider memory system. Read the H200 ratios as the honest tuned-to-tuned
comparison.

## Requirements and installation

Requires Linux x86_64, Python 3.10+, a compatible PyTorch 2.1+, a CUDA Toolkit
(NVRTC and CUDA headers), a host C++ compiler, and a target GPU. Installation
builds only the Python wheel and compiles no Torch/CUDA extension; the launcher,
repack kernel and indexed kernel are JIT-compiled and cached on first use. An
ordinary isolated build therefore does not pull a second copy of Torch and does not
need `--no-build-isolation`:

```bash
python -m pip install -v -e ".[test]"
```

To build a distributable wheel:

```bash
python -m pip wheel -v --no-deps --wheel-dir dist .
```

That packages the Python code and the CUDA/C++ sources the JIT needs, without
precompiling any GPU kernel; the first GPU test or operator call performs the real
JIT compilation.

Before the first run, confirm the Torch in the active environment matches the
driver and CUDA Toolkit. A machine without a full Toolkit can install the wheel but
cannot complete the first kernel compilation. The Blackwell decode path needs NVRTC
support for its JIT target: `sm_100a` for B200 (generally CUDA 12.8+) and `sm_103a`
for B300 (CUDA 13.0+).

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
described in [docs/shapes.md](docs/shapes.md).

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
against a plain-PyTorch reference and then timed on the same launch. The timing
methods, the `cos_diff` accuracy metric, and how to read the throughput columns are
in [docs/benchmarking.md](docs/benchmarking.md). The production shapes and token
sweeps come from [docs/shapes.md](docs/shapes.md).

## Documentation

- [docs/optimizations.md](docs/optimizations.md) — what the three published
  scenarios changed relative to the public Humming baseline, with measurements.
- [docs/tuning.md](docs/tuning.md) — instruction paths, block-M model, the
  2-CTAs/SM window, and the stream-K reduction.
- [docs/benchmarking.md](docs/benchmarking.md) — timing methods, `cos_diff`, and
  reading TFLOPS/GB-s.
- [docs/shapes.md](docs/shapes.md) — Kimi K2.5 EP8 shape derivation, token-count
  scoping, routing distribution, and the gate/up vs down contract.

## Provenance and license

This repository derives from the public `inclusionAI/humming` commit
[`4351af3a8fcdce1a8dee50104ba49566af2427fb`](https://github.com/inclusionAI/humming/commit/4351af3a8fcdce1a8dee50104ba49566af2427fb).
The extraction scope, W4A16 modifications and retained files are listed in
[chord_kernels/operator/SOURCE.md](chord_kernels/operator/SOURCE.md). The upstream
Apache-2.0 license text is in
[chord_kernels/operator/LICENSE](chord_kernels/operator/LICENSE); the repository as a
whole is Apache-2.0. Humming is its only third-party source lineage.

"Based on Humming" describes source lineage, not a runtime dependency. The retained
CUDA headers are a minimal closure extracted from that public revision and rewritten
for indexed BF16 W4A16; they still use `<humming/...>` as the internal include
prefix, which is not an import of an installed Python `humming` package. Taking full
Humming as a submodule or runtime dependency would pull back quantization modes
unrelated to this operator and tie its behavior to the host's Humming version. vLLM
integration therefore uses `chord_kernels.operator`, and this package passes its own
include root to its own NVRTC compilation.
