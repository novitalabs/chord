# Tuning and scheduling internals

How the indexed W4A16 operator picks a kernel schedule, and the stream-K
reduction it uses for prefill. This is background for anyone changing the tuning
tables or the kernel; day-to-day use does not need it.

## Instruction paths and swap-AB

| Profile | Tensor core instruction | Kernel family label |
| --- | --- | --- |
| `h200_prefill_ep8` | WGMMA (`wgmma.mma_async`) | `wgmma` |
| `h200_tp8` | WGMMA (`wgmma.mma_async`) | `wgmma` |
| `h200_decode_ep8` | MMA (`mma.sync`) | `mma` |
| `blackwell_decode_ep8` | MMA (`mma.sync`) | `mma` |

Prefill and decode differ in more than their routed-M range: they use different
tensor core instruction families. Prefill sees a large routed-M, where WGMMA's
asynchronous large tiles amortize instruction overhead. Decode sees a very small
routed-M — often only a few rows per expert — where MMA gives finer control over
the narrow dimension. Both families read one packed weight layout: the WGMMA
dequant reads each word's nibble pairs in its register-A order, so the repack no
longer transposes the mini-block (upstream Humming unified the layout the same
way). One SM90 indexed weight therefore serves the prefill, TP8 and decode
profiles, and a tuning row's warp-N (32 WGMMA, 64 MMA) picks the family; the
profile only picks the tuning table. When the profile is left at `"auto"` on
SM90, the per-instance role comes from the `CHORD_SM90_DECODE` environment
variable (`1` decode; `0`/unset prefill; see
[getting_started.md](getting_started.md)). Blackwell publishes only the
decode profile, so `"auto"` needs no role bit there.

Both instruction paths put the dequantized weight in registers and the activation
in shared memory, so in the general sense every profile "swaps" A and B relative
to a textbook `A @ B`. For WGMMA that swap is unconditional and lives in the
operand assignment itself: the emitted instruction is
`wgmma.mma_async.m{N}n{M}k{K}` with the activation's smem descriptor as operand B
and the weight registers as operand A (RS mode). The `swap_ab` flag in the Python
API is a narrower, MMA-only thing: it selects a runtime schedule that counts
populated 8-token tiles per m-block so all-padding tiles can be skipped. That
machinery does not apply to WGMMA, which is why `kSwapAb` is hard-coded false in
`wgmma.cuh` — it means "this runtime schedule is inapplicable," not "operands are
in textbook order".

## Shapes covered by tuning

Each profile carries a narrow tuning table keyed on `(profile, N, K)` for the
target model shapes and routed-M ranges, not a general performance guarantee for
every SM90/SM100/SM103 shape. The table supplies `block_m`, `block_n`, `block_k`,
weight layout and the swap-AB setting to routing alignment and kernel launch at
once, so both agree by construction. A device or scenario with no matching profile
is rejected explicitly.

| Shape | N | K | Tuning status |
| --- | --- | --- | --- |
| EP8 gate/up | 4096 | 7168 | Tuned |
| EP8 down | 7168 | 2048 | Tuned |
| TP8 gate/up | 512 | 7168 | Tuned |
| TP8 down | 7168 | 256 | Tuned |
| Anything else | — | — | Falls back to generic defaults |

The shard axis is part of the key, and it is not detectable from the device. It
is instead recovered from the projection shapes above, which are disjoint between
the axes — so `profile="auto"` lands on TP8 for a TP8 shape and EP8 otherwise,
with no chord-specific argument. Request it explicitly by name or with
`tensor_parallel_size=8`; a stated axis that contradicts a published shape is
rejected rather than resolved.

`h200_tp8` is also a single-instance (`mode="mix"`) profile rather than a
disaggregated P/D role, so `CHORD_SM90_DECODE` does not apply to it and its
schedule has to hold across the whole routed-M range.

Shapes outside the table still execute correctly, but `block_m` does not vary
with routed-M and performance is untuned.

These tables are a simplified schedule kept for the indexed path. They are not a
general autotune result; treat measurements on the target machine as
authoritative.

## Prefill block-M

Prefill sizes its block-M from routed tokens per expert
(`tok_e = routed_m / num_experts`) rather than from `routed_m` alone. Each
expert's rows are padded up to a whole block independently, so two layers with
equal `routed_m` but different expert counts want different tiles.

- Above `tok_e` 80 the tile is sized to cover one expert's padded rows in the
  fewest blocks under a 176 block-M register ceiling.
- Below `tok_e` 80 block-M comes from minimizing the total block count over a
  sampled routing, since the regime there is block count and occupancy rather than
  padding. The sample reproduces upstream's
  `np.random.RandomState(seed=0).randint(0, num_experts, size=routed_m)` exactly,
  so the chosen block-M matches Humming's for the same shape.
- The mid-size window inside that range is the exception (next section).

Decode uses its own `tok_e` thresholds with swap-AB.

### The mid-size window

For mid-size chunks the block-count argmin over-grows block-M (for example
40 at EP8 1024 tokens, 72 at TP8 2048). The tiles it fed (2-CTA `256x64` and
128-wide stream-K tiles) also split N into 4-16 n-blocks, and each n-block
gathers the same activation rows again. These chunks instead size block-M by modelled row work,
`sum(ceil(count_e / block_m)) * (block_m + 32)` over the same sampled routing,
which charges the padding a tall tile leaves inside every expert as well as a
fixed per-block cost. The window is where that model picks block-M 32..80:
routed_m 980..3370 at 48 experts and 7725..26148 at 384, about `tok_e`
20..70, on both shard axes. Inside it both profiles run one-pass wide tiles:

| Projection | Tile | CTAs/SM | Stream-K |
| --- | --- | --- | --- |
| deep-K gate/up (EP8 `N=4096`, TP8 `N=512`) | `block_m x 512 x 64` | 1 | only when the grid ends in a partial wave |
| EP8 down (`K=2048`) | `block_m x 256 x 64` | 2 | off |
| TP8 down (`K=256`) | `block_m x 256 x 64` (`x 128` past block-M 72) | 2 | block-M 56..72 |

The 512-wide gate/up tile gathers the activation once per 512 output columns,
so TP8 gate/up covers its whole output in one m-block. When the tile count
(from the same sampled routing) leaves the last wave over the 132 SMs less
than 90% full, the K split rebalances it: 4-14% faster at 76-80% wave
efficiency, and 1-5% slower near full waves, where it stays off. Measured
on the benchmark chunks: EP8 gate/up and down 1.07-1.10x at 1024-2048 tokens,
TP8 gate/up 1.19-1.29x and down 1.12-1.30x. The model's 88/96 picks just under
`tok_e` 80 are left on the argmin and its tiles, because there the 512-wide
gate/up tile ran 30-50% slower.

### Stage depth

The mid-size window's tiles, the `256x64` tiles from `tok_e` 80 (prefill
gate/up, prefill down while it splits K, TP8 gate/up) and TP8 down's
`128x64` tiles run the deepest cp.async ring that keeps their CTAs resident:
6 stages, else 5, else the default 4. Shared memory is computed from the
kernel's `SharedStorage` layout plus the driver's 1 KiB per-CTA reservation
(`_h200_num_stages` in `tuning.py`). The deeper ring is worth 1-5% on those
tiles. It never costs a resident CTA: TP8 down's `128x128x64` at 2 CTAs/SM
takes 5 stages, where 6 would leave room for one CTA and run 42% slower.
These keep 4 stages:

- one-pass mid-K prefill down from routed_m 5120 (6 measured 1.0-1.4%
  slower);
- TP8 down's stream-K rows (6 measured 3% slower);
- the small-block tiles and the 2-CTA 40..80 window.

### TP8 block-M

TP8 keys on the same `tok_e`, but the windows are flatter. EP8 stays wide in the
dimension that fills the grid (`N / block_n >= 16` tiles per m-block), so
per-expert M-padding dominates and block-M grows to cover an expert's padded rows.
TP8 narrows that dimension to as few as 2 n-blocks, so block count and occupancy
dominate instead, and EP8's tall tiles would hit the register cliff without a full
grid to pay for it.

| `tok_e` | block-M |
| --- | --- |
| mid-size window (~20..70) | row-work argmin (as EP8) |
| other < 80 | block-count argmin (as EP8) |
| 80–128 | gate/up: row-work argmin; down: one block per expert, `round(tok_e * 1.1 / 8) * 8` |
| 128–190 | 96 |
| > 190 | 128 |

The whole 96..144 band beats the argmin's taller choice here; in the 80-128
band gate/up's row-work size measured 5-12% faster than the padded-rows
size. At the five benchmark token counts (1024..16384, i.e. routed_m
8192..131072) this gives block-M 32, 56, 104, 96, 128 for gate/up and
32, 56, 96, 96, 128 for down.

### TP8 block-N/K tiles

Outside the mid-size window the two projections differ in block-N and
occupancy:

| Projection | block-N | block-K | CTAs/SM |
| --- | --- | --- | --- |
| gate/up (`N=512 K=7168`) | 128 to block-M 64, then 256 | 256 (block-M ≤ 32), 128 (≤ 64), else 64 | 1 |
| down (`N=7168 K=256`) | 128 | 128 (block-M ≤ 32), else 64 | 2 |

Below block-M 64 a short m-block leaves the mainloop too little work per
tile, so block-K compensates; the wide 256 tile pays once block-M passes 64.
down goes the other way: at block-N 256 the tall tiles' WGMMA accumulator
plus B-smem are too large to fit two CTAs on an SM, pinning occupancy at
1 CTA/SM, and with only 4 K-blocks the kernel is latency-bound rather than
compute-bound. Halving block-N to 128 halves both and unlocks 2 CTAs/SM,
which hides the cp.async + dequant latency. At the mid-size window's block-M
32..72, a 256-wide tile still fits two CTAs and halves the activation
gathers, so down uses it there.

### Block-N/K tiles

Outside the mid-size window, small blocks (block-M <= 32) keep upstream's output-width split: gate/up
(N=4096) uses the narrow-deep `128x256` tile, the wide down projection uses
`256x128`. Above block-M 32 both projections use `256x64`. For gate/up
block-M 40..64 upstream's generic `N <= 4096` heuristic would pick `128x128`;
`256x64` with the 2-CTA window measures 12-15% faster on H200 for these EP8
shapes (routed_m 863..2073), so this table deliberately diverges from the
upstream tile there.

### The 2-CTAs/SM window

For the 40..80 block-M window at `block_n=256`, prefill forces 2 CTAs/SM. Doubling
resident CTAs there hides the cp.async + dequant latency, measured ~11% faster on
the down 1024/2048 tiles, whose register use stays under the 2-CTA cap on this
kernel. Outside the window (small block_m at `block_n=128`, or block_m > 80) it
stays at 1 CTA/SM. Inside the mid-size window down runs 2 CTAs/SM from
block-M 32 and gate/up's 512-wide tile runs one.

### Decode CTAs/SM

H200 decode runs its swap-AB `(8|16|24, 256, 64)` tiles at 3 CTAs/SM. On
gate/up, a fourth CTA adds more shared-memory and register pressure than
latency cover: 4 CTAs/SM is 5-8% slower at every decode size. On down it is
1% slower from block-M 16. Only down's block-M 8 tile keeps 4.

## Stream-K

H200 prefill uses stream-K: the K dimension of a tile's tail is split across CTAs
whose partial sums reduce into the output. Under EP8, gate/up enables it at every
size outside the mid-size window; the mid-K down projection (512 < K < 4096)
turns it off once `routed_m` reaches 5120, where the M*N tiles already fill the
grid and the K-split is pure overhead. Inside the mid-size window down is
one-pass, and gate/up's 512-wide tile splits only when its grid ends in a
partial wave (see above). H200 decode is one-pass.

TP8 gate/up splits at every size outside the mid-size window.  With the
6-stage ring the split is 4-5% faster than the same tile one-pass at
`routed_m` 65536-131072 too, where the earlier 4-stage table had turned it
off. TP8 down has 4 K-blocks, which give
the split almost nothing to work with, so it is one-pass everywhere except the
mid-size window's 256-wide tiles at block-M 56..72 (3-12% faster there,
10-12% slower below block-M 56). With 5 stages the `128x128x64` tile at
`routed_m` 131072 no longer needs it (3% faster one-pass).

Blackwell decode splits by projection: the deep-K gate/up projection (K=7168,
112 K-blocks) keeps stream-K at every decode size — the long K loop leaves a
tail imbalance that repays the locks (routed 160: 165 -> 146 us) — while the
short-K down projection (K=2048) stays one-pass through its swap-AB window and
enables the split only on the large non-swap tiles past routed 848.

Two lock protocols sequence the CTAs sharing a tile, selected by slice count
(`utils/ptx/barrier.cuh`):

- **serial chain** (≤ 3 slices): each CTA waits until the lock word equals its
  slice id, adds its partial with a plain read-modify-write, then bumps the lock;
  the last slice resets it to 0. The lock enforces order, so the add need not be
  atomic.
- **counter** (> 3 slices): the slice-0 CTA writes the tile and sets the lock
  negative to release the rest, which then accumulate concurrently with atomicAdd
  and bump the lock back toward 0.

Both leave the lock at 0 when the tile finishes, so a device-resident
zero-initialized int32 lock buffer is reused across launches without a reset,
provided the launches sharing it are ordered. As in upstream Humming, the owner
is the caller: `IndexedW4A16Layer` registers a zeroed 1024-word `locks` buffer
(non-persistent, freed with the layer), `forward_layer` passes a host layer's
own `locks` when it has one, and `indexed(..., locks=...)` takes one directly.
A launch without one falls back to a buffer the launcher keeps per
(device, stream) for the life of the process, since launches are ordered only
within one CUDA stream.

That reduction adds BF16-rounded partials, so the output is not bit-identical
to one-pass. The default FP32 reduction (`IndexedKernelConfig.stream_k_fp32`)
replaces it: every slice but the one at `k = 0` stores its FP32 partial and
bumps the tile's lock, and the `k = 0` slice waits for the others, adds their
partials in CTA order and rounds once. The partials live in per-launch scratch,
one `block_m x block_n` FP32 tile per CTA, taken from the PyTorch caching
allocator like upstream's own per-launch scratch: it returns to the pool as
soon as the launch is queued, and under CUDA-graph capture it comes from the
graph's private pool.

## Relationship to the upstream kernel

The BF16-activation/INT4-weight path is functionally equivalent to upstream
Humming on the same shapes. The small-chunk block-M argmin reproduces its
sampled routing exactly. The tables above no longer follow its choices:
upstream main (`f35c5ff`) picks TMA-loaded, warp-specialized one-pass tiles
on SM90 prefill, which the cp.async indexed path here does not implement,
so the tiles are tuned against this kernel. The measured comparison is in
[performance.md](performance.md). Absolute numbers depend on the GPU and its
clock/power state, so compare on one device.
