# Measured performance

`humming` is public Humming at
[`vllm-project/humming@c00b2e6`](https://github.com/vllm-project/humming/commit/c00b2e645a4b5ef57174f40254968a2996944217)
(main, 2026-10-16), timed with its own `benchmarks/bench_humming.py` (BF16
activation, `uint4` weight, BF16 group-32 scale, the `indexed` path unless a
section says otherwise); `chord` is this repository's profile for that
scenario. Both are timed on the same GPU (one H200 SXM, one B300 SXM6) at the
same shape and the same routing draw, each the mean of two runs (run-to-run
spread under 2%); the H200 rows were re-measured on 2026-10-14 after the
H200 retune described below, alternating the two implementations on one GPU.
The tables were measured against `f35c5ff` (2026-10-13); `c00b2e6` re-measures
within ±0.8% of it on every row on both GPUs (its one kernel change is a
shared-memory fence in the TMA warp-specialized path), so they stand.
Times are per-call microseconds, lower is better.
`gate_up + down` is the speedup of the two stages summed, which is what one MoE
layer actually pays. Humming main's output on these cases matches a PyTorch
BF16 matmul on the dequantized weight (cos_diff ≤ 4e-7, 96.5-100%
bit-identical), so its times are for a correct result.

The baseline moved: this repository derives from the July revision
`4351af3` (see `chord_kernels/operator/SOURCE.md`), and upstream has since
retuned its Hopper and Blackwell schedules. Re-measured on the same GPUs, main
is a median 14-15% faster than `4351af3` on the H200 indexed and B300 rows
(up to 39% on TP8 down) but slower on some grouped rows (up to 13% on
contiguous gate/up, 23% on one EP32 masked row); the speedups below are
against main.

The timing method is the upstream `triton.testing.do_bench` call, so rows are
comparable one-to-one with Humming's published tables; the methods, the
`cos_diff` accuracy metric and how to read the throughput columns are in
[benchmarking.md](benchmarking.md). The chart in the README renders these
tables; after re-measuring, update the copies in
`docs/assets/benchmark_chart.py` and re-run that script.

On H200 this profile now leads main on every layer except two near-ties:
EP8 prefill at 8192 tokens (0.99x per layer) and grouped masked decode at 8
tokens per expert (0.95x).  The indexed rows are 0.97-1.25x per stage and
0.99-1.14x per layer.  The retune (see [tuning.md](tuning.md)) adds a
mid-size window of one-pass wide tiles, deeper cp.async rings on the large
tiles, 3 CTAs/SM on decode, and a wider BLOCK_N rule for masked down.

## H200 EP8 prefill (`h200_prefill_ep8`)

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

At 8192 tokens (`tok_e` 171), main's TMA-loaded, warp-specialized
`96x256x64` one-pass tile is 3% faster on down and level on gate/up.  Main
issues the weight and activation tiles from a dedicated producer warpgroup,
which the indexed cp.async path here does not have.  The same `96x256x64`
tile with or without stream-K at 4-6 stages does not close the gap.  A
`176x256x64` stream-K tile is 2-4% faster at this exact routing draw, but
3-28% slower at neighbouring chunk sizes (`tok_e` 176-213), so the table
does not use it.

## H200 TP8, single-instance mix (`h200_tp8`)

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

This profile leads main by 1.04-1.14x per layer across the sweep.  Main
retuned TP8 after the July baseline, and its TMA-loaded, warp-specialized
one-pass tiles led the previous TP8 table below 4096 tokens (0.85-0.90x per
layer).  The mid-size chunks (`tok_e` ~20..70, 1024-3200 tokens here) now
run one-pass wide tiles instead of narrow stream-K ones: gate/up a single
`32|56x512x64` tile that covers the whole `N=512` output per m-block, down
`32|56x256x64` at 2 CTAs/SM.  With a deeper cp.async ring on the large
tiles, the retune measures 1.30x and 1.16x per layer over the previous TP8
tiles at 1024 and 2048 tokens and 1.02-1.04x from 4096 up, interleaved on
the same GPU.  Gate/up at 8192 tokens is the one stage still level with
main (0.99x).

### Stream-K reduction on H200

The prefill and TP8 stream-K rows reduce their K-split partials in an FP32
workspace and round once, like the B300 rows below; the chord columns above
include it.  This table times both reductions back to back on one H200 SXM
with the same cases and tiles (triton `do_bench`), and checks bit-identity
against a PyTorch BF16 matmul on the BF16-dequantized INT4 weight (first 4096
routed rows):

| profile | stage | num_tokens_total | tile | BF16 µs | FP32 µs | Δ | bit-identical, BF16 / FP32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | gate_up | 4096 | 96x256x64, 6 stages | 544.5 | 545.9 | +0.2% | 97.3% / 99.86% |
| prefill | gate_up | 8192 | 96x256x64, 6 stages | 994.3 | 998.7 | +0.4% | 97.6% / 99.86% |
| prefill | gate_up | 16384 | 176x256x64, 6 stages | 1725.3 | 1698.5 | -1.6% | 97.3% / 99.91% |
| prefill | down | 4096 | 96x256x64, 6 stages | 300.0 | 293.5 | -2.2% | 96.7% / 99.99% |
| TP8 | gate_up | 4096 | 104x256x64, 6 stages | 568.0 | 568.3 | +0.1% | 94.2% / 99.71% |
| TP8 | gate_up | 8192 | 96x256x64, 6 stages | 1040.0 | 1042.8 | +0.3% | 96.9% / 99.70% |
| TP8 | gate_up | 16384 | 128x256x64, 6 stages | 1819.1 | 1828.0 | +0.5% | 98.4% / 99.70% |
| TP8 | down | 2048 | 56x256x64, 2 CTAs | 305.1 | 298.2 | -2.3% | 99.4% / 100.00% |

The other rows are one-pass: the mid-size chunks (1024 and 2048 tokens on
prefill and TP8 gate/up, 1024 on TP8 down) run wide tiles that fill the grid
without a K split, and prefill down from 8192 and TP8 down past 2048 are
one-pass as before.  Over the eight stream-K rows the FP32 reduction ranges
from 2.3% faster to 0.5% slower.  With BF16 reduction the prefill gate/up
4096 output also differed between launches; with FP32 every row is
deterministic.

## H200 EP8 decode (`h200_decode_ep8`)

Tokens per GPU is `bs * (mtp + 1)`.

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

Gate/up runs 3 CTAs/SM rather than 4 (5-8% faster on the same GPU; down's
block-M 8 rows keep 4).

## B300 EP8 decode (`blackwell_decode_ep8`)

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

The gate/up rows split K with stream-K and reduce the partials in an FP32
workspace (one BF16 rounding, deterministic; see
[optimizations.md](optimizations.md)).  The
inherited reduction added BF16 partials into the output instead.  Against it,
same cases and tiles (B300, triton `do_bench`), down has no stream-K row at
these sizes:

| gate_up tokens/GPU | tile | BF16 reduction µs | FP32 reduction µs | Δ | bit-identical to PyTorch, BF16 / FP32 |
| --- | --- | --- | --- | --- | --- |
| 20 | 8x256x64 swap | 146.6 | 147.4 | +0.6% | 89.1% / 99.76% |
| 30 | 8x256x64 swap | 165.2 | 166.3 | +0.7% | 85.6% / 99.83% |
| 40 | 8x256x64 swap | 178.5 | 179.8 | +0.7% | 86.0% / 99.82% |
| 50 | 16x512x128 | 184.5 | 188.9 | +2.4% | 90.1% / 99.82% |

Bit-identity is against a PyTorch BF16 matmul on the BF16-dequantized INT4
weight; the down rows are 100% bit-identical either way.  With BF16 reduction
the 20 and 30 tokens/GPU outputs also differ between launches.

The Blackwell gap has narrowed. The July baseline shipped only its default
config strategy on SM100/SM103 (one-CTA `16x256x128` tiles); re-measured
alongside main on the same B300 it is 1.26-1.48x per layer behind this
profile.  The 1.76-2.14x previously on this page rested on July numbers this
setup no longer reproduces (gate/up 319 us then, 215 us now for the same
revision; the H200 July indexed rows still reproduce within -2..+5%).  Main now has SM100
heuristics (`16x128x128` at 2 CTAs/SM, stream-K on both projections, gate/up
read at 4.1-4.3 TB/s), which leaves this profile 1.08-1.25x ahead, mostly on
down and on the small-token gate/up rows.

## H200 EP8 grouped prefill (`h200_grouped_prefill`, contiguous)

The grouped rows compare against public Humming's own `grouped_contiguous`
path, not its indexed one, since that is the contract this backend replaces
(see [optimizations.md](optimizations.md)). Rows per expert are multiples of the
128-row tile boundary on both sides — `bench_humming.py --balanced` and the
aligned cases in `tests/test_w4a16_grouped.py` — so each row is the same GEMM
shape for both implementations and no tile is spent on padding. 48 local
experts (EP8); `rows/E` x 48 is the total routed rows.

| Stage | rows/E | humming µs | chord µs | chord TFLOPS | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 128 | 865.5 | 635.6 | 566 | 1.36 | 1.35 |
| gate_up | 256 | 1432.3 | 1231.6 | 586 | 1.16 | 1.16 |
| gate_up | 512 | 2781.2 | 2402.7 | 600 | 1.16 | 1.16 |
| down | 128 | 427.9 | 325.2 | 552 | 1.32 |  |
| down | 256 | 731.4 | 628.0 | 574 | 1.16 |  |
| down | 512 | 1441.6 | 1234.0 | 584 | 1.17 |  |

## H200 EP8 grouped decode (`h200_grouped_decode`, masked)

Tokens per expert; the per-GPU batch is `tokens/E * 48 / top_k`, so the 8-token
row is the one that matches the decode counts in [shapes.md](shapes.md).

| Stage | tokens/E | humming µs | chord µs | chord GB/s | Speedup | gate_up + down |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 8 | 253.1 | 268.1 | 2983 | 0.94 | 0.95 |
| gate_up | 16 | 341.5 | 286.6 | 2828 | 1.19 | 1.18 |
| gate_up | 32 | 375.3 | 356.9 | 2322 | 1.05 | 1.06 |
| gate_up | 64 | 468.4 | 455.1 | 1896 | 1.03 | 1.03 |
| down | 8 | 132.0 | 138.2 | 2913 | 0.95 |  |
| down | 16 | 170.7 | 147.5 | 2787 | 1.16 |  |
| down | 32 | 191.1 | 175.4 | 2424 | 1.09 |  |
| down | 64 | 242.1 | 233.6 | 1948 | 1.04 |  |

EP16 (24 experts) and EP32 (12 experts) run the same tables; the grouped
profiles are EP-width agnostic and measure 1.04-1.21x and 1.04-1.31x per layer
over the same baseline (contiguous at 24 experts: 1.16-1.40x).  Down now keeps
BLOCK_N 256 wherever the two-wave guard allows (5.5% faster at 64 tokens per
expert on 48 experts, 8% at 32 on 12).

At 8 tokens per expert on 48 experts main's masked path is still 5% faster.
That row runs the heuristic's `16x256x128` layout at 4 stages, and none of
the 91 other layouts swept through `CHORD_W4A16_BM/BN/BK/CM/CN/STAGES` on
either projection beats it by more than noise.  BM 8 is 22-91% slower,
BM 24/32 7-120%, BN 64/128 1-70%, other stage depths -0.4% to +6.5%, and
2-CTA clusters 1-2%.  The gap is in the kernel, not the layout choice.
