# Optimizations relative to the public Humming baseline

This repository's indexed W4A16 operator derives from the public
[inclusionAI/humming](https://github.com/inclusionAI/humming) revision
`4351af3a8fcdce1a8dee50104ba49566af2427fb`. On top of that baseline it carries a
set of kernel and scheduling optimizations for the three supported scenarios.
This page records what changed and why; the mechanics live in
[tuning.md](tuning.md) and the resolver tables in
`chord_kernels/operator/layer.py`.

Measurements below are BF16 activation x INT4 weight, group-32 scale, indexed
MoE routing, on the Kimi K2.5 shapes for the relevant shard axis: EP8 splits the
expert list (gate/up `N=4096 K=7168`, down `N=7168 K=2048`, 48 local experts),
while TP8 slices `moe_intermediate` (gate/up `N=512 K=7168`, down `N=7168 K=256`,
all 384 experts local); `top_k=8` throughout.  H200 numbers come from H200 SXM and
Blackwell numbers from B300 SXM6; every profile is compiled, run and timed on
each supported part.

| Scenario | Baseline behavior | This repository |
| --- | --- | --- |
| H200 EP8 indexed prefill | WGMMA, per-instruction commit+wait, generic block-M argmin, stream-K always on | WGMMA with batched `wait<1>` pipelining, tok/E block-M model, 2-CTAs/SM window, shape-aware stream-K gate, EP8-tuned tiles |
| H200 TP8 indexed, single-instance mix | WGMMA, generic block-M argmin, stream-K always on | Same WGMMA improvements, plus the TP-scale block-M windows, down's block-N halving for 2 CTAs/SM, and per-projection stream-K crossovers |
| H200 EP8 indexed decode | WGMMA (no decode-specific path) | MMA `swap-AB` decode kernel: 4 CTAs/SM, semi-static token-tile schedule, fused dequant+scale |
| Blackwell (B200/B300) EP8 indexed decode | Default config strategy only (no SM100 heuristics) | Same MMA `swap-AB` kernel with EP8-tuned tile tables (stream-K on deep-K gate/up), `sm_100a`/`sm_103a` JIT targets |

## H200 EP8 indexed prefill (`h200_prefill_ep8`, WGMMA)

**WGMMA batched `wait<1>` pipelining.** The baseline issues
`commit; wait<0>` after every WGMMA instruction, stalling the warpgroup on
`WARPGROUP.DEPBAR` each time. For the BF16 W4A16 path the accumulator is only
read at the epilogue and the dequantized weight registers are double-buffered,
so the mainloop now issues a whole warp-K iteration of WGMMAs, commits once,
and relaxes the wait to `wait<1>` — one group stays in flight and overlaps the
next shared-memory load and weight dequant. The epilogue drains with
`wait<0>`. Bit-identical output; measured gate/up -3~6% and down -1~5% across
the token sweep (e.g. gate/up routed 16384: 1.99 ms -> 1.92 ms).

**Tokens-per-expert block-M model.** The baseline picks block-M by an argmin
over total block count, which over-grows block-M for indexed MoE because it
ignores per-expert padding and the WGMMA accumulator register cost. The
governing quantity is routed tokens per expert (`tok_e = routed_m /
num_experts`), not `routed_m`: above `tok_e` 80 each expert's padded rows are
split into the fewest blocks under a 176 block-M register ceiling and block-M
sized to just cover them (measured up to 1.23x over the baseline choice at
`tok_e` ~149); deep-K gate/up earns one extra 176 window at `tok_e <= 352`
while short-K down goes to 128. Below `tok_e` 80 the baseline argmin is kept,
sampling the identical seeded routing so choices reproduce exactly.

**2-CTAs/SM window.** For block-M 40..80 at `block_n=256` the tiles naturally
use ~157 registers and land at 1 CTA/SM, latency-bound. Forcing 2 CTAs/SM via
`__launch_bounds__` caps registers at 128 (mild spill) but doubles resident
warps, hiding the cp.async-gather + dequant latency: measured +5..14% on down
and +3..13% on gate/up in that window. Outside the window it regresses and
stays at 1 CTA/SM.

**Shape-aware stream-K gate.** The baseline enables stream-K unconditionally.
For the mid-K down projection (512 < K < 4096) at `routed_m >= 5120` the M*N
tiles already fill the grid, so the K-split only adds lock and reduction
overhead: disabling it there is a consistent 1-4% win, bit-identical. Gate/up
(deep K) keeps stream-K at every size.

**EP8 tile table.** Small blocks (block-M <= 32) split by output width:
gate/up keeps the narrow-deep `128x256` tile, the wide down projection uses
`256x128`. For gate/up block-M 40..64 the wide `256x64` tile plus the 2-CTA
window replaces the baseline `128x128` choice, measured 12-15% faster at
routed 863..2073.

## H200 TP8 indexed, single-instance mix (`h200_tp8`, WGMMA)

TP8 runs the same WGMMA kernel and inherits the kernel-level changes above, which
are shape independent. What differs is the schedule: slicing `moe_intermediate`
narrows exactly the dimension the EP8 tables assume is wide. Being a single-instance
(`mix`) profile, one packed weight also has to cover both phases. Measurements are
H200 SXM over `num_tokens_total` 1024..16384 (`routed_m` 8192..131072).

**TP-scale block-M windows.** With as few as 2 n-blocks per m-block, grid fill and
occupancy replace per-expert M-padding as the binding constraint, so the EP8 padding
model would over-grow block-M into the register cliff without a full grid to pay for
it. TP8 uses flatter windows instead — one block per expert to `tok_e` 128, then 96
to 190, then 128 — and the whole 96..144 band beats the block-count argmin. Below
`tok_e` 80 the argmin is kept, sampling the identical seeded routing.

**down block-N halving for 2 CTAs/SM.** At the generic `block_n=256` the WGMMA
accumulator plus B-smem pin down's occupancy at 1 CTA/SM, and with `K=256` giving
only 4 K-blocks the mainloop is short and latency-bound rather than compute-bound.
Halving to `block_n=128` halves both and unlocks 2 CTAs/SM, hiding the cp.async +
dequant latency. This is scoped to short K so the compute-bound gate/up, where
halving block-N regresses, is untouched.

**Per-projection stream-K crossovers.** The two projections flip in opposite
directions at `routed_m` 65536: deep-K gate/up keeps the split until the M*N tiles
fill the grid on their own and then drops it, while short-K down starts one-pass —
4 K-blocks give the split almost nothing to work with — and only enables it past
65536, where the workload is imbalanced enough for load balancing to repay the
locks.

Against the public baseline this measures 1.09-1.19x on gate/up and 1.27-1.51x on
down, for 1.17-1.33x per layer ([performance.md](performance.md)); the block-N
halving is why down gains most. The table also replaces the generic fallback these
shapes previously landed on.

## H200 EP8 indexed decode (`h200_decode_ep8`, MMA swap-AB)

The baseline runs decode on the same WGMMA path as prefill. At decode token
counts (a few routed rows per expert) WGMMA is barrier- and latency-bound: one
large tile per SM at 1 CTA/SM. This repository adds an `mma.sync` decode
kernel with A and B swapped:

**Swap-AB layout.** The dequantized weight fills mma operand A (`n_out` on
mma-M) and the gathered activations fill operand B (tokens on mma-N), so a
few-token block costs one `m16n8k16` instead of a WGMMA tile, and the small
tile runs at 4 CTAs/SM. `block_m` becomes the token dimension on mma-N, which
legalizes block-M 8 (mma-N floor) — the non-swap MMA path would need an
illegal `m8n8k8` BF16 instruction. The activation tile is read straight into
the B-fragment with non-transposing `ldmatrix.x2` over the same shared-memory
swizzle the cp.async writer produced. Weights are packed in the MMA layout at
transform time, so the profile is a pack-time choice. Measured vs the WGMMA
baseline at decode token counts: down 1.10-1.34x, gate/up 1.05-1.24x.

**Semi-static token-tile schedule.** Each m-block counts its populated 8-token
tiles (tokens are front-packed per expert; a tile is padding iff its first
routed id is the sentinel). The MMA loop issues tile `j=0` unconditionally —
every block has at least one populated tile — and guards only `j >= 1` on the
populated count. That gives ptxas a predicate-free static anchor to schedule
around, with even same-accumulator reissue spacing: measured against the
fully-static form, warpgroup stall-wait 0.77 vs 1.19 and -7% cycles; against
the fully-dynamic form at 9-15 tok/E, 186 us vs 216 us.

**Fused dequant + group scale.** For the BF16/INT4/group-32 case the group
scale multiply is fused into the nibble-extraction loop (one
`lop3 -> hsub2 -> hmul2` chain per register pair) instead of a separate
apply-scale pass. The fusion uses subtract-then-scale rather than
`hfma2(x, bs, -136*bs)`: `(128+w)-136 = w-8` is exact in BF16, so the result
is the single-rounding `RN((w-8)*bs)`, bit-identical to the unfused path,
where the fma form would round `-136*bs` into a bias that accumulates
coherently over K.

**Decode block-M table.** `tok_e`-calibrated: block-M 8 for `tok_e <= 6`, 16
to 13, else 24, with tile `(block_m, 256, 64)`, warp `(block_m, 64, 64)`,
4 CTAs/SM, one-pass (no stream-K).

**P/D role selection.** Because prefill and decode want different physical
weight layouts, the role must be fixed before weights are packed.
`CHORD_SM90_DECODE=1` selects the decode profile for `profile="auto"` on SM90;
unset/0 selects prefill. See [getting_started.md](getting_started.md).

## Blackwell EP8 indexed decode (`blackwell_decode_ep8`)

The public baseline has no SM100 heuristics and ships only its default config
strategy there. This repository runs the same MMA swap-AB decode kernel — `mma.sync`
`m16n8k16` is native on SM100/SM103; no tcgen05 path is required for these
token counts. SM100 and SM103 share one tile table, tuned on B300 (148 SMs):

- Per-shape routed-M brackets, tuned independently per projection (see
  `_select_indexed_kernel_config` in `layer.py`). A host that builds one
  routing for both projections passes its block-M through the `block_m`
  forward argument; the override swaps only the M tile and keeps the row's
  N/K tile.
- **gate/up keeps stream-K at every decode size.** Its K loop is deep (112
  blocks of 64), so the tail balance repays the lock overhead: routed 160
  measures 165 -> 146 us (+13%) over the same tile one-pass, and the win
  holds through routed 2048. Small routed-M uses the swap-AB `(8, 256, 64)`
  tile at 4 CTAs/SM; routed 321..736 uses non-swap `(16, 512, 128)`; larger
  routed-M shifts to `(32|48, 512, 64)`, all with stream-K.
- **down stays one-pass through its swap-AB window.** The swap-AB
  `(8|16, 256, 64)` x4 padding-skip rows win to routed 848 — 7..21% over a
  `(16, 512, 64)` one-CTA tile across routed 480..848 — because down's short
  K (32 blocks) gives stream-K nothing to balance there. Past 848 the
  non-swap `(32|48, 512, 64)` tiles win, and there stream-K is
  neutral-to-positive so it stays on.
- The semi-static token-tile schedule and fused dequant above apply
  unchanged.
- JIT targets `sm_100a` (B200) and `sm_103a` (B300) separately per detected
  capability; cubins are not shared across the two.

On B300 the decode sweep (`tests/test_w4a16.py`, triton `do_bench`) measures
gate/up 146/162/181/183 us and down 84/91/92/95 us at 20/30/40/50 tokens per
GPU.
