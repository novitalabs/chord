# Performance

Against [`vllm-project/humming@b4f3fc6`](https://github.com/vllm-project/humming/commit/b4f3fc6ee1dfc637cb8081e57c02c4bf1791c5f1)
(2026-10-19): `indexed` against Humming's indexed path, `masked`/`contiguous`
against its `grouped_masked`/`grouped_contiguous`, timed with its own
`benchmarks/bench_humming.py` on the same GPU (H200 SXM, B300 SXM6), the same
shape and the same routing draw. Mean of two runs, spread under 2%. Per-call
microseconds; the `gate_up + down` column is speedup(gate_up + down), Humming's
summed time over chord's. The chart in the README is
`docs/assets/benchmark_chart.py` over these tables.

Per layer chord is ahead or level everywhere except grouped contiguous at 512
rows/E (0.96x) and EP8 prefill at 2048 tokens (0.99x). Humming b4f3fc6 retuned
its heuristics and stream-K; against c00b2e6 it is 4-23% faster on B300 decode,
grouped contiguous, masked at 16-32 tokens/E and H200 chunks up to 2048 tokens,
and 11-14% slower on masked at 64 tokens/E.

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
| gate_up | 1024 | 300.0 | 200.40 | 297.1 | 202 | 1.01 | 1.01 |
| gate_up | 2048 | 364.9 | 329.59 | 367.9 | 327 | 0.99 | 0.99 |
| gate_up | 4096 | 563.8 | 426.63 | 545.3 | 441 | 1.03 | 1.03 |
| gate_up | 8192 | 1031.4 | 466.40 | 999.9 | 482 | 1.03 | 1.01 |
| gate_up | 16384 | 1947.2 | 494.07 | 1697.8 | 566 | 1.15 | 1.11 |
| down | 1024 | 154.9 | 194.05 | 154.1 | 195 | 1.01 |  |
| down | 2048 | 195.4 | 307.74 | 195.8 | 307 | 1.00 |  |
| down | 4096 | 297.2 | 404.64 | 293.2 | 410 | 1.01 |  |
| down | 8192 | 524.4 | 458.65 | 540.8 | 445 | 0.97 |  |
| down | 16384 | 1004.5 | 478.91 | 961.0 | 501 | 1.05 |  |

At 8192 tokens Humming's `96x256x64` tile, whose weights a TMA producer warpgroup
loads, is 3% faster on down; the same tile on chord's cp.async path does not
close it (a bulk-copy weight load without a producer warpgroup measured 5-24%
slower).

## H200 TP8

| Stage | num_tokens_total | humming µs | humming TFLOPS | chord µs | chord TFLOPS | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 1024 | 307.1 | 195.83 | 300.4 | 200 | 1.02 | 1.03 |
| gate_up | 2048 | 372.4 | 322.93 | 377.2 | 319 | 0.99 | 1.01 |
| gate_up | 4096 | 603.0 | 398.86 | 566.0 | 425 | 1.07 | 1.08 |
| gate_up | 8192 | 1068.0 | 450.41 | 1042.5 | 461 | 1.02 | 1.07 |
| gate_up | 16384 | 2061.3 | 466.74 | 1821.3 | 528 | 1.13 | 1.15 |
| down | 1024 | 209.2 | 143.72 | 199.0 | 151 | 1.05 |  |
| down | 2048 | 311.8 | 192.87 | 297.9 | 202 | 1.05 |  |
| down | 4096 | 519.5 | 231.50 | 475.4 | 253 | 1.09 |  |
| down | 8192 | 956.4 | 251.47 | 852.5 | 282 | 1.12 |  |
| down | 16384 | 1840.0 | 261.44 | 1585.3 | 303 | 1.16 |  |

## H200 EP8 decode

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 215.0 | 3540.18 | 209.3 | 3637 | 1.03 | 1.03 |
| gate_up | 30 | 247.1 | 3217.38 | 238.8 | 3331 | 1.04 | 1.06 |
| gate_up | 40 | 253.9 | 3134.92 | 239.4 | 3324 | 1.06 | 1.07 |
| gate_up | 50 | 255.8 | 3114.96 | 248.2 | 3210 | 1.03 | 1.04 |
| down | 20 | 116.2 | 3293.81 | 112.2 | 3412 | 1.04 |  |
| down | 30 | 132.2 | 3032.68 | 118.5 | 3382 | 1.12 |  |
| down | 40 | 134.1 | 2999.23 | 123.0 | 3269 | 1.09 |  |
| down | 50 | 136.0 | 2968.96 | 127.2 | 3176 | 1.07 |  |

## B300 EP8 decode

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 149.1 | 5107.17 | 147.7 | 5154 | 1.01 | 1.01 |
| gate_up | 30 | 168.5 | 4718.80 | 167.2 | 4758 | 1.01 | 1.01 |
| gate_up | 40 | 181.5 | 4384.25 | 181.7 | 4384 | 1.00 | 1.02 |
| gate_up | 50 | 185.5 | 4294.63 | 189.5 | 4205 | 0.98 | 1.01 |
| down | 20 | 85.2 | 4492.74 | 83.8 | 4575 | 1.02 |  |
| down | 30 | 94.2 | 4256.66 | 92.0 | 4357 | 1.02 |  |
| down | 40 | 98.0 | 4103.31 | 92.0 | 4367 | 1.07 |  |
| down | 50 | 102.3 | 3948.20 | 95.0 | 4253 | 1.08 |  |

Humming's SM100 schedule is now chord's: `8x256x64` swap-AB at 4 CTAs/SM with
stream-K (`16x256x64` at 50 tokens), so the B300 lead is down to 1.01-1.02x.

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
| gate_up | 128 | 762.3 | 631.2 | 570 | 1.21 | 1.20 |
| gate_up | 256 | 1338.4 | 1220.5 | 592 | 1.10 | 1.09 |
| gate_up | 512 | 2249.0 | 2369.3 | 608 | 0.95 | 0.96 |
| down | 128 | 383.7 | 322.9 | 556 | 1.19 |  |
| down | 256 | 674.4 | 620.5 | 581 | 1.09 |  |
| down | 512 | 1183.3 | 1223.8 | 589 | 0.97 |  |

24 experts: 1.02-1.19x speedup(gate_up + down). At 512 rows/E Humming's
stream-K `88x256x64`/`80x256x64` beats chord's one-pass `128x128x64`.

## H200 grouped decode (`masked`)

48 experts; the per-GPU batch is `tokens/E * 48 / top_k`.

| Stage | tokens/E | humming µs | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 8 | 253.1 | 242.1 | 3311 | 1.05 | 1.05 |
| gate_up | 16 | 287.9 | 261.8 | 3095 | 1.10 | 1.10 |
| gate_up | 32 | 354.5 | 340.6 | 2434 | 1.04 | 1.03 |
| gate_up | 64 | 531.4 | 464.4 | 1862 | 1.14 | 1.15 |
| down | 8 | 131.8 | 124.7 | 3235 | 1.06 |  |
| down | 16 | 147.7 | 134.1 | 3064 | 1.10 |  |
| down | 32 | 172.2 | 170.3 | 2498 | 1.01 |  |
| down | 64 | 267.7 | 229.8 | 1980 | 1.17 |  |

EP16 (24 experts) 1.04-1.17x, EP32 (12 experts) 1.07-1.17x speedup(gate_up + down). The 8
tokens per expert row was closed by running two resident CTAs per SM, not by a
tile: none of 91 swept layouts beats the heuristic's `16x256x128`.
