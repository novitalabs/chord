# Performance

Against [`vllm-project/humming@7ec54ed`](https://github.com/vllm-project/humming/commit/7ec54ed8c673e9b31ebb23de4e99c0968075ad42)
(2026-10-20): `indexed` against Humming's indexed path, `masked`/`contiguous`
against its `grouped_masked`/`grouped_contiguous`, timed with its own
`benchmarks/bench_humming.py` on the same GPU (H200 SXM, B300 SXM6), the same
shape and the same routing draw. Mean of two runs, spread under 2%. Per-call
microseconds; the `gate_up + down` column is speedup(gate_up + down), Humming's
summed time over chord's. The chart in the README is
`docs/assets/benchmark_chart.py` over these tables.

Per layer chord is ahead or level everywhere except EP8 prefill at 2048 tokens
(0.99x). The Humming commits since b4f3fc6 (#153 SM120 eviction threshold, #154
NVRTC cache) leave these GPUs unchanged: every row within 1.6% of b4f3fc6.

## Shapes

Kimi K2.5 MoE: 384 routed experts, `top_k` 8, `hidden_size` 7168,
`moe_intermediate` 2048, on 8 GPUs.

| Shard | Experts per GPU | gate/up N x K | down N x K |
| --- | --- | --- | --- |
| EP8 | 48, whole | 4096 x 7168 | 7168 x 2048 |
| TP8 (`moe_intermediate` / 8) | 384 | 512 x 7168 | 7168 x 256 |

| Phase | Tokens | Scope | Routed rows per GPU |
| --- | --- | --- | --- |
| EP8 prefill | 1024 .. 16384 | system, per chunk | = tokens (8 routes x 48/384 experts) |
| TP8 | 1024 .. 16384 | system, per chunk | tokens * 8 |
| Decode | 20 .. 50 | per GPU, `bs * (mtp + 1)` | tokens * 8 |

Routing is the upstream benchmark's (top `top_k` of random scores, seeded by the
token count). Down is launched with the routed rows and `top_k=1`.

## Method

```bash
python tests/test_w4a16_indexed.py                  # indexed cases for this GPU
python tests/test_w4a16_grouped.py                  # SM90 masked + contiguous
python tests/test_w4a16_indexed.py --method kineto  # kernel time only
```

Every case is checked against a PyTorch reference (BF16 dequant, FP32
accumulate: engineering error, not quantization error) and timed on the same
launch. `triton` (default) is `do_bench(warmup=100, rep=1000)`, the call
`bench_humming.py` makes, launch overhead included; `kineto` is kernel time only.
`GB/s` counts bytes of the experts routing reached, as Humming does.

## H200 EP8 prefill

| Stage | num_tokens_total | humming µs | humming TFLOPS | chord µs | chord TFLOPS | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 1024 | 299.9 | 200.51 | 296.8 | 202 | 1.01 | 1.01 |
| gate_up | 2048 | 364.7 | 329.75 | 367.0 | 328 | 0.99 | 0.99 |
| gate_up | 4096 | 563.1 | 427.16 | 545.9 | 441 | 1.03 | 1.02 |
| gate_up | 8192 | 1032.3 | 466.01 | 999.0 | 482 | 1.03 | 1.01 |
| gate_up | 16384 | 1939.1 | 496.14 | 1698.2 | 566 | 1.14 | 1.11 |
| down | 1024 | 154.5 | 194.59 | 153.3 | 196 | 1.01 |  |
| down | 2048 | 193.9 | 310.07 | 195.9 | 307 | 0.99 |  |
| down | 4096 | 296.6 | 405.49 | 293.9 | 409 | 1.01 |  |
| down | 8192 | 525.3 | 457.86 | 542.1 | 444 | 0.97 |  |
| down | 16384 | 1003.0 | 479.60 | 962.5 | 500 | 1.04 |  |

At 8192 tokens Humming's `96x256x64` tile, whose weights a TMA producer warpgroup
loads, is 3% faster on down; the same tile on chord's cp.async path does not
close it (a bulk-copy weight load without a producer warpgroup measured 5-24%
slower).

## H200 TP8

| Stage | num_tokens_total | humming µs | humming TFLOPS | chord µs | chord TFLOPS | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 1024 | 307.0 | 195.85 | 300.7 | 200 | 1.02 | 1.03 |
| gate_up | 2048 | 373.1 | 322.32 | 377.7 | 319 | 0.99 | 1.01 |
| gate_up | 4096 | 603.3 | 398.67 | 567.4 | 424 | 1.06 | 1.08 |
| gate_up | 8192 | 1070.8 | 449.23 | 1039.7 | 462 | 1.03 | 1.07 |
| gate_up | 16384 | 2064.1 | 466.09 | 1826.1 | 527 | 1.13 | 1.15 |
| down | 1024 | 208.1 | 144.45 | 198.8 | 151 | 1.05 |  |
| down | 2048 | 311.1 | 193.29 | 298.1 | 202 | 1.04 |  |
| down | 4096 | 519.5 | 231.49 | 474.2 | 254 | 1.10 |  |
| down | 8192 | 956.0 | 251.58 | 852.3 | 282 | 1.12 |  |
| down | 16384 | 1841.8 | 261.17 | 1584.9 | 303 | 1.16 |  |

## H200 EP8 decode

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 215.6 | 3530.43 | 210.7 | 3613 | 1.02 | 1.03 |
| gate_up | 30 | 246.8 | 3221.58 | 236.8 | 3360 | 1.04 | 1.07 |
| gate_up | 40 | 253.8 | 3136.10 | 239.6 | 3322 | 1.06 | 1.07 |
| gate_up | 50 | 256.0 | 3112.51 | 248.8 | 3202 | 1.03 | 1.04 |
| down | 20 | 116.2 | 3295.33 | 112.4 | 3404 | 1.03 |  |
| down | 30 | 132.1 | 3034.29 | 118.6 | 3379 | 1.11 |  |
| down | 40 | 134.4 | 2993.28 | 122.5 | 3281 | 1.10 |  |
| down | 50 | 135.1 | 2988.49 | 127.0 | 3180 | 1.06 |  |

## B300 EP8 decode

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 149.1 | 5104.48 | 147.8 | 5149 | 1.01 | 1.01 |
| gate_up | 30 | 168.7 | 4712.05 | 167.2 | 4758 | 1.01 | 1.01 |
| gate_up | 40 | 181.2 | 4391.71 | 181.9 | 4380 | 1.00 | 1.02 |
| gate_up | 50 | 185.6 | 4291.90 | 183.8 | 4336 | 1.01 | 1.03 |
| down | 20 | 85.1 | 4497.88 | 83.8 | 4575 | 1.02 |  |
| down | 30 | 94.0 | 4263.18 | 92.0 | 4359 | 1.02 |  |
| down | 40 | 98.1 | 4100.73 | 92.0 | 4365 | 1.07 |  |
| down | 50 | 102.3 | 3947.51 | 95.0 | 4251 | 1.08 |  |

Humming's SM100 schedule is now chord's: `8x256x64` swap-AB at 4 CTAs/SM with
stream-K (`16x256x64` at 50 tokens), so the B300 lead is down to 1.01-1.03x.

## Stream-K reduction

Stream-K partials are reduced in FP32 and rounded once. Against BF16 partial
adds, same tiles; bit-identity is against a PyTorch BF16 matmul on the
dequantized weight:

| GPU | scenario | stage | tokens | tile | BF16 µs | FP32 µs | Δ | bit-identical, BF16 / FP32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H200 | prefill | gate_up | 4096 | 96x256x64, 6 stages | 544.5 | 545.9 | +0.2% | 97.3% / 99.86% |
| H200 | prefill | gate_up | 8192 | 96x256x64, 6 stages | 994.3 | 998.7 | +0.4% | 97.6% / 99.86% |
| H200 | prefill | gate_up | 16384 | 176x256x64, 6 stages | 1725.3 | 1698.5 | -1.6% | 97.3% / 99.91% |
| H200 | prefill | down | 4096 | 96x256x64, 6 stages | 300.0 | 293.5 | -2.2% | 96.7% / 99.99% |
| H200 | TP8 | gate_up | 4096 | 104x256x64, 6 stages | 568.0 | 568.3 | +0.1% | 94.2% / 99.71% |
| H200 | TP8 | gate_up | 8192 | 96x256x64, 6 stages | 1040.0 | 1042.8 | +0.3% | 96.9% / 99.70% |
| H200 | TP8 | gate_up | 16384 | 128x256x64, 6 stages | 1819.1 | 1828.0 | +0.5% | 98.4% / 99.70% |
| H200 | TP8 | down | 2048 | 56x256x64, 2 CTAs | 305.1 | 298.2 | -2.3% | 99.4% / 100.00% |
| B300 | decode | gate_up | 20 | 8x256x64 swap | 146.6 | 147.4 | +0.6% | 89.1% / 99.76% |
| B300 | decode | gate_up | 30 | 8x256x64 swap | 165.2 | 166.3 | +0.7% | 85.6% / 99.83% |
| B300 | decode | gate_up | 40 | 8x256x64 swap | 178.5 | 179.8 | +0.7% | 86.0% / 99.82% |
| B300 | decode | gate_up | 50 | 16x512x128 | 184.5 | 188.9 | +2.4% | 90.1% / 99.82% |

FP32 is also deterministic across launches; BF16 was not on three rows.

## H200 grouped prefill (`contiguous`)

48 experts, rows per expert aligned to 128 on both sides (`bench_humming.py
--balanced`).

| Stage | rows/E | humming µs | chord µs | chord TFLOPS | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 128 | 750.5 | 596.5 | 603 | 1.26 | 1.23 |
| gate_up | 256 | 1338.2 | 1137.3 | 636 | 1.18 | 1.17 |
| gate_up | 512 | 2251.0 | 2238.8 | 643 | 1.01 | 1.02 |
| down | 128 | 384.2 | 322.8 | 556 | 1.19 |  |
| down | 256 | 673.1 | 581.5 | 620 | 1.16 |  |
| down | 512 | 1178.5 | 1122.7 | 642 | 1.05 |  |

24 experts: 1.10-1.23x speedup(gate_up + down).

## H200 grouped decode (`masked`)

48 experts; the per-GPU batch is `tokens/E * 48 / top_k`.

| Stage | tokens/E | humming µs | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 8 | 253.0 | 241.6 | 3319 | 1.05 | 1.05 |
| gate_up | 16 | 287.3 | 261.1 | 3103 | 1.10 | 1.10 |
| gate_up | 32 | 353.7 | 338.1 | 2452 | 1.05 | 1.04 |
| gate_up | 64 | 531.0 | 464.4 | 1862 | 1.14 | 1.15 |
| down | 8 | 132.0 | 124.5 | 3240 | 1.06 |  |
| down | 16 | 147.2 | 134.3 | 3059 | 1.10 |  |
| down | 32 | 171.9 | 169.6 | 2508 | 1.01 |  |
| down | 64 | 267.8 | 229.4 | 1983 | 1.17 |  |

EP16 (24 experts) 1.04-1.17x, EP32 (12 experts) 1.07-1.17x speedup(gate_up + down). The 8
tokens per expert row was closed by running two resident CTAs per SM, not by a
tile: none of 91 swept layouts beats the heuristic's `16x256x128`.
