# Design and tuning

Background for changing the tuning tables or kernels. Measured on the Kimi K2.5
shapes in [performance.md](performance.md#shapes), H200 SXM and B300 SXM6.

## Indexed kernel

| Schedule | Instruction | Config warp-N |
| --- | --- | --- |
| H200 EP8 prefill, H200 TP8 | WGMMA (`wgmma.mma_async`) | 32 |
| H200 EP8 decode, Blackwell decode | MMA (`mma.sync`), swap-AB for small token blocks | 64 |

Both families read one packed weight: the WGMMA dequant reads each word's nibble
pairs in register-A order, so one pack serves every schedule. The dequantized
weight sits in registers, the activation in shared memory.

- **WGMMA `wait<1>`.** One warp-K iteration of WGMMAs per commit, one group in
  flight over the next load and dequant: 3-6% on gate/up, 1-5% on down.
- **Swap-AB decode.** Weight on mma-M, tokens on mma-N, so a few-token block is
  one `m16n8k16` and block-M 8 is legal.
- **Semi-static token tiles.** Tile `j=0` is unconditional, only `j >= 1` is
  guarded: -7% cycles against fully static, 186 vs 216 us against fully dynamic.
- **Fused dequant.** `lop3 -> hsub2 -> hmul2`; subtract-then-scale keeps
  `w - 8` exact, so the result is the single rounding `RN((w-8)*bs)`.

### Stream-K

A tile's K tail is split across CTAs, sequenced by a serial chain (up to three
slices) or a counter (`utils/ptx/barrier.cuh`). Both leave the lock word at 0,
so the caller's `locks` is reused without a reset while launches sharing it are
ordered. Partials are FP32 in per-launch scratch from the PyTorch caching
allocator; the `k = 0` slice adds them in CTA order and rounds once, so the
output is deterministic. `stream_k_fp32=False` keeps BF16 partial adds into the
output ([cost](performance.md#stream-k-reduction)).

## Indexed schedules

`indexed` picks the config in `chord_kernels/operator/tuning.py` from the GPU,
the weight's N/K/E and `routed_m = M * top_k`:

| GPU | Shape | Table |
| --- | --- | --- |
| H200 | TP8 (512x7168, 7168x256) | TP8 |
| H200 | other, below 18.75 routed rows per expert | EP8 decode (swap-AB MMA) |
| H200 | other, from 18.75 | EP8 prefill (WGMMA) |
| B200/B300 | any | Blackwell decode (MMA) |

The H200 crossover is measured: decode is 4-29% faster through routed_m 888 at
48 experts, the two tie at 904-936, prefill is 24-55% faster from 1120. Untuned
shapes get a fallback tile (block-M 16 WGMMA or 8 swap-AB, block-N the largest
power of two up to 256 dividing N, block-K 64). `block_m=` replaces the M tile
and keeps the N/K tile.

### Block-M

Block-M follows routed tokens per expert, `tok_e = routed_m / num_experts`.

| Range | Rule |
| --- | --- |
| `tok_e` < 80 | fewest blocks over a sampled routing (`RandomState(0).randint`, as Humming) |
| mid window (`tok_e` ~20..70) | minimize `sum(ceil(count_e / block_m)) * (block_m + 32)` |
| EP8 prefill, `tok_e` >= 80 | cover `tok_e * 1.1` in fewest blocks, block-M <= 176 |
| TP8, `tok_e` >= 80 | gate/up: row-work argmin to 128; down: one block per expert; then 96, 128 |
| H200 decode | 8 for `tok_e / 0.9` <= 6 (gate/up 5.5), 16 to 13, else 24 |
| Blackwell decode | fixed routed-M brackets (below) |

### Tiles

Mid window, one-pass wide tiles:

| Projection | Tile | CTAs/SM | Stream-K |
| --- | --- | --- | --- |
| deep-K gate/up | `block_m x 512 x 64` | 1 | when the last wave is < 90% full |
| EP8 down | `block_m x 256 x 64` | 2 | off |
| TP8 down | `block_m x 256 x 64` (`x 128` past block-M 72) | 2 | block-M 56..72 |

EP8 prefill outside the window:

| Block-M | gate/up | down |
| --- | --- | --- |
| <= 32 | `128x256` | `256x128` |
| 40..80 | `256x64`, 2 CTAs/SM | `256x64`, 2 CTAs/SM |
| above | `256x64`, 1 CTA/SM | `256x64`, 1 CTA/SM |

Stream-K is on for gate/up, and for down below routed_m 5120.

TP8 outside the window:

| Projection | block-N | block-K | CTAs/SM | Stream-K |
| --- | --- | --- | --- | --- |
| gate/up | 128 to block-M 64, then 256 | 256 (<= 32), 128 (<= 64), else 64 | 1 | on |
| down | 128 | 128 (<= 32), else 64 | 2 | off |

Stage depth is the deepest cp.async ring that keeps the CTAs resident (6, else
5, else 4; `_h200_num_stages`), worth 1-5%.

H200 decode: `(8|16|24, 256, 64)` swap-AB, one-pass, 3 CTAs/SM (down block-M 8
keeps 4).

Blackwell decode, one table for SM100/SM103, tuned on B300:

| Projection | routed_m | Tile | CTAs/SM | Stream-K |
| --- | --- | --- | --- | --- |
| gate/up | <= 320 | `8x256x64` swap-AB | 4 | on |
| gate/up | 321..736 | `16x256x64` swap-AB | 4 | on |
| gate/up | 737..1424 / above | `32x512x64` / `48x512x64` | 1 | on |
| down | <= 272 / 273..848 | `8x256x64` / `16x256x64` swap-AB | 4 | off |
| down | 849..1424 / above | `32x512x64` / `48x512x64` | 1 | on |

## Grouped kernel (`masked` / `contiguous`, SM90)

The SM90 W4A16 GEMM vendored from DeepGEMM: TMA weight loads from a producer
warpgroup, WGMMA consumers.

- **Weight.** The `pack_w4a16` layout, the one indexed reads, so one packed copy
  serves all three operators. The producer TMAs each stage as one unswizzled
  `BLOCK_K/16 x BLOCK_N` box of uint64 (one per N column of a K16 row); a
  consumer thread's word per K16 is its RS-WGMMA A fragment, and the
  lane-permuted scales give rows `r` and `r+8` as one BF16 pair. With BLOCK_N
  256 each warp takes two adjacent 16-row slabs, one per wave, so one
  `ld.shared.v2` feeds both (one 32-bit load per wave is a 4-way bank conflict
  and cost masked em=64 1-2%).
- **Tiles** (`grouped/heuristics.py`, per call): contiguous `128x256x128`, 3
  stages, no cluster (`128x128x64` for short-K below 256 rows/E, BM64 for one
  small problem); masked BLOCK_M from `expected_m` (`1.3x` gate/up,
  `ceil(1.25x, 8)` down), BN 256 only while `G x ceil(N/256) >= 2 x num_sms`,
  stages for ~512 buffered K (~768 from BM 72).
- **Residency** (`_ctas_per_sm`): small masked tiles (`block_m <= 24`) run two
  CTAs per SM -- a persistent grid of `num_sms * 2`, halved smem ring and
  register budget -- which hides the single TMA producer's latency the way
  Humming's compact-MoE path does; one CTA everywhere else.

`CHORD_W4A16_BM/BN/BK/CM/CN/STAGES` pin one layout, `CHORD_W4A16_CTAS` the
residency. The masked kernel's limits
are in [getting_started.md](getting_started.md).
