# Performance

Against [`vllm-project/humming@c00b2e6`](https://github.com/vllm-project/humming/commit/c00b2e645a4b5ef57174f40254968a2996944217)
(2026-10-16): `indexed` against Humming's indexed path, `masked`/`contiguous`
against its `grouped_masked`/`grouped_contiguous`, timed with its own
`benchmarks/bench_humming.py` on the same GPU (H200 SXM, B300 SXM6), the same
shape and the same routing draw. Mean of two runs, spread under 2%. Per-call
microseconds; the `gate_up + down` column is speedup(gate_up + down), Humming's
summed time over chord's. The chart in the README is
`docs/assets/benchmark_chart.py` over these tables.

On H200 chord leads on speedup(gate_up + down) everywhere except EP8 prefill at 8192 tokens (0.99x);
that gap is in the kernel, not the tile choice.

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
| gate_up | 1024 | 334.9 | 179.52 | 297.4 | 202 | 1.13 | 1.12 |
| gate_up | 2048 | 385.4 | 312.08 | 367.4 | 327 | 1.05 | 1.05 |
| gate_up | 4096 | 569.5 | 422.33 | 546.9 | 440 | 1.04 | 1.03 |
| gate_up | 8192 | 999.5 | 481.29 | 1001.1 | 481 | 1.00 | 0.99 |
| gate_up | 16384 | 1952.7 | 492.70 | 1692.5 | 568 | 1.15 | 1.11 |
| down | 1024 | 169.1 | 177.84 | 154.1 | 195 | 1.10 |  |
| down | 2048 | 207.1 | 290.37 | 196.1 | 307 | 1.06 |  |
| down | 4096 | 296.8 | 405.14 | 293.9 | 409 | 1.01 |  |
| down | 8192 | 524.1 | 458.91 | 541.0 | 445 | 0.97 |  |
| down | 16384 | 1003.4 | 479.42 | 959.0 | 502 | 1.05 |  |

At 8192 tokens Humming's `96x256x64` tile, whose weights a TMA producer warpgroup
loads, is 3% faster on down; the same tile on chord's cp.async path does not
close it.

## H200 TP8

| Stage | num_tokens_total | humming µs | humming TFLOPS | chord µs | chord TFLOPS | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 1024 | 340.4 | 176.66 | 300.5 | 200 | 1.13 | 1.10 |
| gate_up | 2048 | 395.2 | 304.32 | 378.5 | 318 | 1.04 | 1.04 |
| gate_up | 4096 | 594.3 | 404.68 | 566.9 | 424 | 1.05 | 1.14 |
| gate_up | 8192 | 1029.3 | 467.34 | 1040.3 | 462 | 0.99 | 1.05 |
| gate_up | 16384 | 2019.6 | 476.37 | 1820.7 | 528 | 1.11 | 1.13 |
| down | 1024 | 207.5 | 144.88 | 197.7 | 152 | 1.05 |  |
| down | 2048 | 309.9 | 194.02 | 298.1 | 202 | 1.04 |  |
| down | 4096 | 597.8 | 201.16 | 476.7 | 252 | 1.25 |  |
| down | 8192 | 955.5 | 251.73 | 853.7 | 282 | 1.12 |  |
| down | 16384 | 1840.9 | 261.30 | 1591.4 | 302 | 1.16 |  |

## H200 EP8 decode

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 226.3 | 3364.14 | 210.3 | 3620 | 1.08 | 1.06 |
| gate_up | 30 | 257.8 | 3084.34 | 239.3 | 3323 | 1.08 | 1.09 |
| gate_up | 40 | 253.4 | 3140.91 | 240.0 | 3316 | 1.06 | 1.07 |
| gate_up | 50 | 255.2 | 3121.97 | 248.0 | 3213 | 1.03 | 1.04 |
| down | 20 | 116.5 | 3287.25 | 112.5 | 3404 | 1.04 |  |
| down | 30 | 131.9 | 3039.01 | 118.5 | 3381 | 1.11 |  |
| down | 40 | 134.8 | 2984.07 | 123.2 | 3264 | 1.09 |  |
| down | 50 | 135.1 | 2987.56 | 127.5 | 3167 | 1.06 |  |

## B300 EP8 decode

| Stage | tokens/GPU | humming µs | humming GB/s | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | 20 | 179.1 | 4251.67 | 147.2 | 5172 | 1.22 | 1.25 |
| gate_up | 30 | 191.4 | 4153.30 | 165.7 | 4800 | 1.16 | 1.19 |
| gate_up | 40 | 192.5 | 4134.89 | 180.1 | 4422 | 1.07 | 1.13 |
| gate_up | 50 | 192.9 | 4129.49 | 188.9 | 4218 | 1.02 | 1.08 |
| down | 20 | 110.1 | 3476.92 | 83.7 | 4578 | 1.32 |  |
| down | 30 | 113.9 | 3517.71 | 91.8 | 4366 | 1.24 |  |
| down | 40 | 114.2 | 3523.22 | 91.8 | 4377 | 1.24 |  |
| down | 50 | 114.3 | 3532.76 | 94.8 | 4260 | 1.21 |  |

Humming's SM100 schedule is `16x128x128`, 2 CTAs/SM, stream-K on both stages.

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
| gate_up | 128 | 865.5 | 635.6 | 566 | 1.36 | 1.35 |
| gate_up | 256 | 1432.3 | 1231.6 | 586 | 1.16 | 1.16 |
| gate_up | 512 | 2781.2 | 2402.7 | 600 | 1.16 | 1.16 |
| down | 128 | 427.9 | 325.2 | 552 | 1.32 |  |
| down | 256 | 731.4 | 628.0 | 574 | 1.16 |  |
| down | 512 | 1441.6 | 1234.0 | 584 | 1.17 |  |

24 experts: 1.16-1.40x speedup(gate_up + down).

## H200 grouped decode (`masked`)

48 experts; the per-GPU batch is `tokens/E * 48 / top_k`.

| Stage | tokens/E | humming µs | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 8 | 253.1 | 247.5 | 3238 | 1.02 | 1.03 |
| gate_up | 16 | 341.5 | 270.2 | 2999 | 1.26 | 1.25 |
| gate_up | 32 | 375.3 | 350.0 | 2368 | 1.07 | 1.08 |
| gate_up | 64 | 468.4 | 457.5 | 1890 | 1.02 | 1.03 |
| down | 8 | 132.0 | 128.1 | 3149 | 1.03 |  |
| down | 16 | 170.7 | 139.6 | 2943 | 1.22 |  |
| down | 32 | 191.1 | 173.1 | 2458 | 1.10 |  |
| down | 64 | 242.1 | 230.5 | 1974 | 1.05 |  |

EP16 (24 experts) 1.04-1.21x, EP32 (12 experts) 1.04-1.31x speedup(gate_up + down). The 8
tokens per expert row was closed by running two resident CTAs per SM, not by a
tile: none of 91 swept layouts beats the heuristic's `16x256x128`.
