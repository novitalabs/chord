# Measured performance

`humming` is the public Humming `indexed` path, `chord` is this repository's
profile for that scenario; both are timed on the same GPU at the same shape and
the same routing draw. Times are per-call microseconds, lower is better.
`gate_up + down` is the speedup of the two stages summed, which is what one MoE
layer actually pays.

The timing method is the upstream `triton.testing.do_bench` call, so rows are
comparable one-to-one with Humming's published tables; the methods, the
`cos_diff` accuracy metric and how to read the throughput columns are in
[benchmarking.md](benchmarking.md). The chart in the README renders these
tables; after re-measuring, update the copies in
`docs/assets/benchmark_chart.py` and re-run that script.

## H200 EP8 prefill (`h200_prefill_ep8`)

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

## H200 EP8 decode (`h200_decode_ep8`)

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

## B300 EP8 decode (`blackwell_decode_ep8`)

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

The Blackwell speedups are larger than the H200 ones mostly because of the
baseline, not because the Blackwell schedule is better tuned than the Hopper
one. Public Humming ships only its default config strategy on SM100/SM103, so
its B300 numbers are an untuned reference point — note how its time barely
moves from 20 to 50 tokens per GPU, and how it reads roughly the same GB/s on
B300 as on H200 despite the wider memory system. Read the H200 ratios as the
honest tuned-to-tuned comparison.
