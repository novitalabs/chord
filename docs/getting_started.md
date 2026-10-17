# Getting started

## indexed

```python
import torch
from chord_kernels import indexed, indexed_block_m, pack_w4a16_indexed

weight = pack_w4a16_indexed(weight_uint4, scale)        # once, at load time
locks = torch.zeros(1024, dtype=torch.int32, device="cuda")

block_m = indexed_block_m(weight, num_tokens, top_k)     # per call
sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(topk_ids, block_m, num_experts)
out = indexed(x, weight, sorted_ids, expert_ids, num_tokens_padded, top_k, locks=locks)
```

| Argument | Contract |
| --- | --- |
| `pack_w4a16_indexed(weight_uint4, scale)` | int32 codes `[E, N, K]` in `[0, 15]`, decoded as `code - 8`; `packed=True` takes the checkpoint form `[E, N, K/8]`. BF16 scale `[E, N, K/32]`. `N % 128 == 0`, `K % 64 == 0`. |
| `x` | contiguous BF16 `[M, K]`. For down, the routed rows `[M * top_k, K]` with `top_k=1`. |
| `sorted_ids`, `expert_ids`, `num_tokens_padded` | contiguous CUDA int32, as `moe_align_block_size` returns them, aligned to `indexed_block_m`. Buffers may be over-allocated; only the first `num_tokens_padded / block_m` blocks are read. Each block holds its routed ids (`< M * top_k`, each once) as a prefix followed by sentinels. EP routing may carry only this rank's routes. |
| `locks` | zeroed int32 `[1024]`, allocated once and reused: every launch indexes it from word 0 and leaves it zeroed. Launches sharing one buffer must be ordered (one stream, or one graph); launches that may run concurrently, e.g. on different streams or graphs replayed on different streams, need one buffer each. Needed when the tuned schedule uses stream-K. |
| `block_m=` | a host that aligns one routing for both projections passes that block size; the tile keeps its N/K. |
| `outputs=` | reuse an output buffer `[M * top_k, N]`. Rows of remote experts are not written. |
| `validate_routing=True` | synchronizing check of the routing contract, for debugging. Off by default: a routing that breaks the contract gives wrong results or out-of-bounds reads without an error. |

The schedule is picked inside `indexed` from the GPU, the weight's N/K and
`M * top_k`. Calls are CUDA-graph safe: no host sync, and no allocation outside
the PyTorch caching allocator.

## grouped (SM90)

```python
from chord_kernels import contiguous, masked, pack_w4a16_grouped

weight = pack_w4a16_grouped(weight_uint4, scale)
out = masked(x3, weight, masked_m, expected_m)
out = contiguous(x2, weight, m_indices)
```

| Operator | Inputs |
| --- | --- |
| `masked` (decode) | `x3` BF16 `[G, max_m, K]`, `masked_m` int32 `[G]` valid rows per expert, `expected_m` their typical count. Returns `[G*max_m, N]`. |
| `contiguous` (prefill) | `x2` BF16 `[m, K]`, each expert's rows padded to 128 (padding zeroed), `m_indices` int32 `[m]` (`-1` for padding). Returns `[m, N]`. |

One packed weight serves both operators. Keep
`max_m >= 16` and `expected_m` close to the real counts: the masked kernel
mis-writes when `max_m` is below its BLOCK_M or a group's `masked_m` exceeds it.

## Environment variables

| Variable | Effect |
| --- | --- |
| `CHORD_CACHE_DIR`, `CHORD_TMP_DIR` | JIT cache and lock directories (default `.chord_cache/`, `.chord_tmp/` beside the package). |
| `CHORD_CPP_JIT=1` | JIT and launch every kernel from C++ ([cpp_jit.md](cpp_jit.md)). |
| `CHORD_W4A16_BM/BN/BK/CM/CN/STAGES` | pin one grouped layout, for tuning sweeps. |

## Tests

```bash
python -m pytest -m "not gpu" tests/  # CPU-only
python -m pytest tests/               # GPU, without the perf-sized cases
python tests/test_w4a16_indexed.py    # indexed perf table for this GPU
python tests/test_w4a16_grouped.py    # grouped perf table (SM90)
```
