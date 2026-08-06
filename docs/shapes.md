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

## Kimi K2.5 TP8

TP8 shards the same model over the same 8 GPUs along the other axis. Instead of
splitting the expert list, **every rank keeps all 384 routed experts and slices
`moe_intermediate` 2048 into 8 pieces of 256**. So TP8 narrows exactly the
dimension EP8 leaves whole:

| Projection | N | K | Derivation |
| --- | --- | --- | --- |
| gate/up | 512 | 7168 | `N = 2 * (moe_intermediate / 8)`, `K = hidden_size` |
| down | 7168 | 256 | `N = hidden_size`, `K = moe_intermediate / 8` |

gate/up loses output width and down loses K depth; `hidden_size` 7168 is untouched
in both. Because the narrowed dimension is the one that decides how many tiles a
block covers, TP8 needs a different schedule and not just different brackets — see
[tuning.md](tuning.md).

## Token counts

Prefill and decode sweep different token counts because they model different
serving phases:

| Phase | Tokens | Scope | Local routed rows |
| --- | --- | --- | --- |
| Prefill, EP8 | 1024, 2048, 4096, 8192, 16384 | Whole 8-GPU system, per chunk (16384 is chunked 16k) | Equal to the token count |
| TP8 (mix) | 1024, 2048, 4096, 8192, 16384 | Whole 8-GPU system, per chunk | `tokens * top_k` |
| Decode | 20, 30, 40, 50 | Per GPU, per step: `bs_per_gpu * (mtp + 1)` | `tokens * top_k` |

The phases count tokens at different scopes, and the two shard axes turn the same
count into very different local row counts, so the derivation matters. TP8's sweep
is chunk-shaped even though it serves a mixed instance; decode counts land at the
low end of the same table.

Chunked numbers are system-wide in both cases. Under **EP8** a token's `top_k = 8`
routes spread over all 384 experts while this GPU owns 48 of them, so the routes
landing locally are `T * 8 * (48 / 384) = T` — the local routed row count happens
to equal the whole system's token count. A case built from `T` therefore uses
`m = T / top_k`, which makes `routed_m = T`.

Under **TP8** nothing is split by expert. Every rank holds a slice of every
expert's intermediate dimension, so all `T * top_k` routes are local:
`routed_m = T * 8`, eight times the EP8 count at the same `num_tokens_total`
(16384 tokens is 131072 rows). That is why the TP8 tuning table has to reach far
past EP8's largest bracket.

Decode runs EP8 together with DP8, so each GPU already receives its own token batch
and those counts need no rescaling; each token still fans out to `top_k` routes on
the local experts, giving `routed_m = tokens * top_k`.

Getting this wrong matters because `routed_m` is what selects the tuning bracket.
Reading the EP8 prefill numbers as per-GPU would model eight times the work that
reaches one GPU; conversely, applying the EP8 rule to TP8 would model one eighth of
it. Either mistake collapses the whole sweep onto the wrong brackets.

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
