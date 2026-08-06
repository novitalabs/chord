# Tuning and scheduling internals

How the indexed W4A16 operator picks a kernel schedule, and the stream-K
reduction it uses for prefill. This is background for anyone changing the tuning
tables or the kernel; day-to-day use does not need it.

## Instruction paths and swap-AB

| Profile | Tensor core instruction | Weight layout |
| --- | --- | --- |
| `h200_prefill_ep8` | WGMMA (`wgmma.mma_async`) | `wgmma` |
| `h200_tp8` | WGMMA (`wgmma.mma_async`) | `wgmma` |
| `h200_decode_ep8` | MMA (`mma.sync`) | `mma` |
| `blackwell_decode_ep8` | MMA (`mma.sync`) | `mma` |

Prefill and decode differ in more than their routed-M range: they use different
tensor core instruction families, so the physical weight order differs too.
Prefill sees a large routed-M, where WGMMA's asynchronous large tiles amortize
instruction overhead. Decode sees a very small routed-M — often only a few rows
per expert — where MMA gives finer control over the narrow dimension. The layout
is baked into the weight when it is packed, so the profile must be chosen at
packing time and cannot be switched at runtime; a mismatched layout is rejected
before launch. A production deployment prepares the matching profile per P/D
instance, or keeps two packed copies of the weight. When the profile is left at
`"auto"` on SM90, the per-instance role comes from the `CHORD_SM90_DECODE`
environment variable (`1` decode; `0`/unset prefill; see
[getting_started.md](getting_started.md)), mirroring
upstream Humming's `HUMMING_INT_SM90_DECODE`. Blackwell publishes only the
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

The shard axis is part of the key, not just the shape, and it is not detectable
from the device, so `profile="auto"` never resolves to TP8 — guessing would pack
the weight against the wrong table. Request it by name, or pass
`tensor_parallel_size=8`.

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

Decode uses its own `tok_e` thresholds with swap-AB.

### TP8 block-M

TP8 keys on the same `tok_e`, but the windows are flatter. EP8 stays wide in the
dimension that fills the grid (`N / block_n >= 16` tiles per m-block), so
per-expert M-padding dominates and block-M grows to cover an expert's padded rows.
TP8 narrows that dimension to as few as 2 n-blocks, so block count and occupancy
dominate instead, and EP8's tall tiles would hit the register cliff without a full
grid to pay for it.

| `tok_e` | block-M |
| --- | --- |
| < 80 | block-count argmin (as EP8) |
| 80–128 | one block per expert, `round(tok_e * 1.1 / 8) * 8` |
| 128–190 | 96 |
| > 190 | 128 |

The whole 96..144 band beats the argmin's taller choice here. At the five benchmark
token counts (1024..16384, i.e. routed_m 8192..131072) this gives block-M
40, 72, 96, 96, 128 for both projections.

### TP8 block-N/K tiles

A short m-block leaves the mainloop too little work per tile, so block-K
compensates and relaxes as block-M grows. The two projections differ in block-N and
occupancy.

| Projection | block-N | block-K | CTAs/SM |
| --- | --- | --- | --- |
| gate/up (`N=512 K=7168`) | 128 to block-M 64, then 256 | 256 (block-M ≤ 32), 128 (≤ 64), else 64 | 1 |
| down (`N=7168 K=256`) | 128 always | 128 (block-M ≤ 32), else 64 | 2 |

gate/up's narrow output is already covered by a 128-wide tile in 4 n-blocks, so the
wide 256 tile only pays once block-M passes 64 and the deep-K mainloop has enough
rows per tile to feed it. down goes the other way: at the generic block-N 256 the
WGMMA accumulator plus B-smem are too large to fit two CTAs on an SM, pinning
occupancy at 1 CTA/SM, and with only 4 K-blocks the kernel is latency-bound rather
than compute-bound. Halving block-N to 128 halves both and unlocks 2 CTAs/SM, which
hides the cp.async + dequant latency. That is the largest single TP8 win — down
measures 1.27-1.51x against the baseline, against gate/up's 1.09-1.19x. down's
block-K is also one notch shallower throughout, because `K=256` simply has less
depth to spend.

### Block-N/K tiles

Small blocks (block-M <= 32) keep upstream's output-width split: gate/up
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
stays at 1 CTA/SM.

## Stream-K

H200 prefill uses stream-K: the K dimension of a tile's tail is split across CTAs
whose partial sums reduce into the output. Under EP8, gate/up enables it at every
size; the mid-K down projection (512 < K < 4096) turns it off once `routed_m`
reaches 5120, where the M*N tiles already fill the grid and the K-split is pure
overhead. H200 decode is one-pass.

TP8 crosses over in *opposite directions* for the two projections, both at
`routed_m` 65536:

| Projection | ≤ 65536 | > 65536 |
| --- | --- | --- |
| gate/up (`K=7168`) | on | off |
| down (`K=256`) | off | on |

gate/up behaves like its EP8 counterpart, with the crossover pushed out to where
TP8's larger routed-M fills the grid on its own. down is the mirror image: 4
K-blocks give the split almost nothing to work with, so it stays one-pass until the
workload is imbalanced enough for the load balancing to repay the locks.

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
zero-initialized int32 lock buffer is reused across launches without a reset.
Because launches are ordered only within one CUDA stream, the launcher keeps one
lock buffer per (device, stream) and zero-fills it asynchronously on that stream
before its first stream-K launch. Because the reduction accumulates across CTAs
in an undefined float order, the output is not bit-identical to one-pass, but
stays within the correctness tolerance.

## Relationship to the upstream kernel

The BF16-activation/INT4-weight path is functionally equivalent to upstream
Humming on the same shapes, and the block-M model, 2-CTAs/SM window and
stream-K gate reproduce its choices, so kernel time matches the upstream tables
within a few percent on the same machine. Absolute numbers depend on the GPU
and its clock/power state, so compare on one device.
