<!-- markdownlint-disable MD001 MD033 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo_lockup_dark.svg">
    <img alt="Chord" src="docs/assets/logo_lockup.svg" width="240">
  </picture>
</p>

<h3 align="center">
Novita Labs' production MoE CUDA kernel
</h3>

<p align="center">
| <a href="#documentation"><b>Documentation</b></a> | <a href="https://novita.ai"><b>Novita AI</b></a> | <a href="https://blogs.novita.ai"><b>Blog</b></a> |
</p>

---

Chord (Python package `chord_kernels`) is Novita Labs' in-house W4A16 MoE CUDA
operator — BF16 activation, INT4 weight, group-32 scale — built for serving the
Kimi K2.5 family (K2.5/K2.6/K2.7), the main open-weight LLM family shipping
INT4 weights.

## Supported configurations

| Interface | GPU | Compute capability | Scenario | Profile |
| --- | --- | --- | --- | --- |
| `indexed` | Hopper (H200) | SM90 (9.0) | Prefill, EP8 | `h200_prefill_ep8` |
| `indexed` | Hopper (H200) | SM90 (9.0) | Single-instance (mix), TP8 | `h200_tp8` |
| `indexed` | Hopper (H200) | SM90 (9.0) | Decode, EP8 | `h200_decode_ep8` |
| `indexed` | Blackwell (B200/B300) | B200: SM100 (10.0); B300: SM103 (10.3) | Decode, EP8 | `blackwell_decode_ep8` |
| `masked` | Hopper (H200) | SM90 (9.0) | Decode, EP8/EP16/EP32 | `h200_grouped_decode` |
| `contiguous` | Hopper (H200) | SM90 (9.0) | Prefill, EP8/EP16/EP32 | `h200_grouped_prefill` |

## Performance

Per-call latency against the public Humming `indexed` path; on SM100/SM103
public Humming ships only its default config strategy. Full tables are in
[docs/performance.md](docs/performance.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark_chart_dark.svg">
  <img src="docs/assets/benchmark_chart.svg" alt="Per-call latency versus token count for the public Humming baseline and this repository across the four supported scenarios, lower is better">
</picture>

## Installation

```bash
pip install git+https://github.com/novitalabs/chord.git
```

NVIDIA CUTLASS ships as a git submodule and supplies the arch headers the JIT
compiles against; `pip` fetches it as part of the command above. A source
checkout needs it initialized explicitly:

```bash
git clone --recurse-submodules https://github.com/novitalabs/chord.git
cd chord
pip install -e .
```

For a checkout that already exists, run `git submodule update --init --recursive`
first.

## Quick start

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
# down projection: inputs are the already-routed (top_k-expanded) activations,
# so call with top_k=1
```

The full API surface — packing layouts, the routing contract, the layer adapter
for framework integration and per-instance P/D role selection — is in
[docs/getting_started.md](docs/getting_started.md).

### DeepGEMM-layout grouped paths (SM90)

The `masked` (decode) and `contiguous` (prefill) interfaces consume the
DeepGEMM-native group layouts and pack a different weight buffer via
`pack_w4a16_grouped`. The two modes are not interchangeable: masked packs
with BLOCK_K=128, contiguous with BLOCK_K=64, and the packed buffer records
the mode so a mismatch fails loudly at dispatch.

```python
from chord_kernels import contiguous, masked
from chord_kernels.operator import pack_w4a16_grouped

# decode: activations laid out per expert with a fixed row budget per expert
packed_masked = pack_w4a16_grouped(weight, scale, "masked")
out = masked(a3, packed_masked, masked_m, expected_m)    # [G, max_m, N]

# prefill: activations concatenated per expert, padded to 128-row boundaries
packed_contig = pack_w4a16_grouped(weight, scale, "contiguous")
out = contiguous(a2, packed_contig, m_indices)           # [m, N]
```

Layer-level, `select_indexed_profile("auto")` keeps the indexed phase-1
profiles by default; setting `CHORD_USE_GROUPED=1` before model
load reroutes both SM90 roles to `h200_grouped_{prefill,decode}`.

## Documentation

- [docs/getting_started.md](docs/getting_started.md) — JIT cache, low-level
  API, routing contract, layer adapter, tests.
- [docs/performance.md](docs/performance.md) — the measured tables behind the
  chart above.
- [docs/optimizations.md](docs/optimizations.md) — what the three published
  scenarios changed relative to the public Humming baseline, with measurements.
- [docs/tuning.md](docs/tuning.md) — instruction paths, block-M model, the
  2-CTAs/SM window, and the stream-K reduction.
- [docs/benchmarking.md](docs/benchmarking.md) — timing methods, `cos_diff`, and
  reading TFLOPS/GB-s.
- [docs/shapes.md](docs/shapes.md) — Kimi K2.5 EP8 and TP8 shape derivation,
  token-count scoping, routing distribution, and the gate/up vs down contract.

## Provenance and license

This repository derives from the public `inclusionAI/humming` commit
[`4351af3a8fcdce1a8dee50104ba49566af2427fb`](https://github.com/inclusionAI/humming/commit/4351af3a8fcdce1a8dee50104ba49566af2427fb);
"based on Humming" describes source lineage, not a runtime dependency. The
SM90 `masked`/`contiguous` backend additionally vendors the W4A16 kernel and
its launch heuristics from `deepseek-ai/DeepGEMM`
([public release commit `7f2a703`](https://github.com/deepseek-ai/DeepGEMM/tree/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50),
secondarily developed; MIT), together with the arch-level CUTLASS/CuTe headers
it includes (BSD-3-Clause, NVIDIA). The extraction scope, W4A16 modifications and
retained files of both upstreams are listed in
[chord_kernels/operator/SOURCE.md](chord_kernels/operator/SOURCE.md). The
upstream license texts are in
[chord_kernels/operator/LICENSE](chord_kernels/operator/LICENSE),
[chord_kernels/operator/include/deep_gemm/LICENSE](chord_kernels/operator/include/deep_gemm/LICENSE)
and
[chord_kernels/operator/include/cutlass/LICENSE.txt](chord_kernels/operator/include/cutlass/LICENSE.txt);
the repository as a whole is Apache-2.0.
