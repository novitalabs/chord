# Production shapes and routing

Where the benchmark shapes come from, and how the routing metadata is built.

## Kimi K2.5 EP8

The production shapes are the Kimi K2.5 MoE architecture — a main open-weight
LLM family shipping INT4 weights (K2.5/K2.6/K2.7), and so the reason a W4A16
kernel exists at all. The relevant config is 384 routed experts, `top_k` 8,
`hidden_size` 7168 and `moe_intermediate` 2048.

**EP8 means expert parallelism over 8 GPUs**: the 384 routed experts are split
into 8 groups, so each GPU owns `384 / 8 = 48` of them. It does not mean any tensor
is sliced 8 ways — every expert stays whole, which is why the per-expert N and K
are the model's full dimensions:

| Projection | N | K | Derivation |
| --- | --- | --- | --- |
| gate/up | 4096 | 7168 | `N = 2 * moe_intermediate` (gate and up fused into one GEMM before SwiGLU), `K = hidden_size` |
| down | 7168 | 2048 | `N = hidden_size`, `K = moe_intermediate` |

## Token counts

Prefill and decode sweep different token counts because they model different
serving phases:

| Phase | Tokens | Scope | Local routed rows |
| --- | --- | --- | --- |
| Prefill | 1024, 2048, 4096, 8192, 16384 | Whole EP8 system, per chunk (16384 is chunked 16k) | Equal to the token count |
| Decode | 20, 30, 40, 50 | Per GPU, per step: `bs_per_gpu * (mtp + 1)` | `tokens * top_k` |

The two phases count tokens at different scopes, which changes how the local routed
row count is derived.

Prefill numbers are system-wide. A token's `top_k = 8` routes spread over all 384
experts and this GPU owns 48 of them, so the routes landing locally are
`T * 8 * (48 / 384) = T` — the local routed row count happens to equal the whole
system's token count. A case built from `T` therefore uses `m = T / top_k`, which
makes `routed_m = T`.

Decode runs EP8 together with DP8, so each GPU already receives its own token batch
and those counts need no rescaling; each token still fans out to `top_k` routes on
the local experts, giving `routed_m = tokens * top_k`.

Getting this wrong matters because `routed_m` is what selects the tuning bracket:
treating the prefill numbers as per-GPU would model eight times the work that
reaches one GPU and would collapse the whole sweep onto a single bracket.

## Routing distribution

Routing defaults to a random distribution matching the upstream Humming benchmark:
score every expert per token and take the top `top_k`. Expert load is therefore
skewed, which exercises intra-block padding and the empty-expert path.
`IndexedCase(distribution="balanced")` switches to an even distribution where every
expert receives a near-identical token count and padding is minimal.

## gate/up and down projections

`IndexedCase(projection=...)` selects between the two routing contracts of an MoE
expert GEMM:

| projection | input rows | output rows | `top_k` passed to the kernel |
| --- | --- | --- | --- |
| `gate_up` | `M` | `M * top_k` | `top_k` |
| `down` | `M * top_k` | `M * top_k` | `1` |

The kernel recovers the source token row as `sorted_ids[i] / top_k`. Gate/up reads
each token once and fans it out to `M * top_k` rows. Down consumes the activations
gate/up already routed, so input and output row counts are equal; passing
`top_k=1` degenerates that division into an identity map and each routed row indexes
itself. In both cases routing metadata is generated with the real `top_k`, so
`sorted_ids` values always span `[0, M * top_k)`.
