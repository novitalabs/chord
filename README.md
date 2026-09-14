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
| Grouped family | Hopper (H200) | SM90 (9.0) | Standalone prefill/decode operator tests | `h200_grouped_*` |

**Grouped vLLM integration is work in progress (WIP).** The current release
provides grouped kernels for standalone correctness and performance testing;
it does not provide a completed grouped integration through vLLM's Humming
backend. The indexed family retains its legacy `HummingMethod` adapter;
compatibility requires a matching vLLM API and INT4 group-32 support. This
revision does not implement vLLM's newer functional Humming API.

## Performance

Per-call latency against public Humming, each backend measured against the
matching Humming path (`indexed` against `indexed`, grouped against Humming's
grouped paths); on SM100/SM103 public
Humming ships only its default config strategy. Full tables are in
[docs/performance.md](docs/performance.md).

These are standalone kernel measurements. Grouped performance results do not
establish grouped vLLM readiness or end-to-end serving speedups.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark_chart_dark.svg">
  <img src="docs/assets/benchmark_chart.svg" alt="Per-call latency versus token count for the public Humming baseline and this repository across the six measured scenarios, lower is better">
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

### Grouped operators (SM90; vLLM integration WIP)

The grouped family adapts DeepGEMM's W4A16 kernel for standalone operator
execution on Hopper. The caller provides the grouped activation layout and
routing tensors. Run its correctness checks and performance tables directly:

```bash
python tests/test_w4a16_grouped.py
```

Integrating this family with vLLM's existing Humming backend is ongoing work.
`CHORD_USE_GROUPED=1` selects the grouped kernels inside Chord, but is not a
ready-to-use vLLM integration switch. Keep it unset or `0` for the existing
indexed Humming adapter, and check that the vLLM version implements the API
that this release provides.

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
SM90 grouped backend additionally vendors the W4A16 kernel and
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
