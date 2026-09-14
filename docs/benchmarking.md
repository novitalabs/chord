# Benchmarking and correctness reporting

How `tests/test_w4a16_indexed.py` times the kernel and what its columns mean.
`tests/test_w4a16_grouped.py` prints the masked (decode) and contiguous
(prefill) tables in the same column layout and roofline model, so everything
below about reading the columns applies to both; its shape columns carry `G` and
`m/grp` instead of a routing distribution, and it fixes the timing method to
CUDA events.

## Running

Shapes are declared in `tests/generators.py` as `PERFORMANCE_CASES`. Adding or
removing a shape is an edit to that table; the test module only iterates over
whichever cases the current device supports and prints one row each.

```bash
python tests/test_w4a16_indexed.py                 # run everything, print the table
python tests/test_w4a16_indexed.py --device cuda:1 # pick a specific GPU
```

Every case is checked against a reference and then timed, so each row carries both
its accuracy (`cos_diff`) and its throughput for the same launch. Rows are grouped
into a block per profile and projection, and a shape column is shown only where it
varies within its block — the heading states the rest.

The same cases are collected by pytest. GPU cases need `-s` for the table to reach
the terminal, and the production-sized ones additionally need `--run-perf` because
they are marked `perf` and skipped by default:

```bash
python -m pytest -m "not gpu" tests/test_w4a16_indexed.py
python -m pytest -m gpu -s tests/test_w4a16_indexed.py
python -m pytest --run-perf -m "gpu and perf" -s tests/test_w4a16_indexed.py
```

## Timing methods

`--method` selects the timing method (default `triton`):

- **`triton`** calls `triton.testing.do_bench(warmup=100, rep=1000)`, the exact
  call the upstream Humming `bench_humming.py` uses, so numbers line up directly
  with its published tables. `do_bench` clears L2 with its own cache buffer between
  iterations and times the wall clock around the call, so launch overhead is
  included.
- **`kineto`** reads the kernel's own GPU duration from the profiler, excluding
  launch overhead. It reports the kernel's intrinsic speed and reads a few percent
  higher than `triton` on these shapes; use it to compare kernels, not to compare
  against the Humming tables.
- **`events`** is a plain CUDA-event loop, the fallback when neither Triton nor the
  profiler is available.

The `kineto` and `events` methods flush L2 with 256 MB between iterations
(H200's L2 is 50 MB, B300's is 132 MB, so this evicts either). Set
`CHORD_FLUSH_L2_BYTES=8000000000` for DeepGEMM's 8 GB flush instead; its purpose is
to give the GPU idle time between iterations rather than to evict L2, so it reports
the peak a single kernel can reach and reads higher.

## Reading the throughput columns

`TFLOPS` and `GB/s` are printed as `achieved/roofline`, with the utilization
percentage in parentheses on whichever column is the binding wall: `TFLOPS` for
the compute-bound prefill blocks (indexed prefill, contiguous prefill), `GB/s`
for the bandwidth-bound decode blocks (indexed decode, masked decode). Both ceilings describe the same operating
point clipped by the running device's BF16 tensor peak and HBM bandwidth
(H200 SXM: 989 TFLOPS / 4800 GB/s; B200/B300: 2250 TFLOPS / 7700 GB/s at the
shipping memory clock), so that percentage is the same in either unit; showing
both makes it visible which wall is binding. The INT4 weight is dequantized to BF16
before the MMA, so the compute ceiling is the BF16 rate.

`GB/s` counts weight and scale bytes only for the experts routing actually reached,
reported as `act.E`. Under a random router with a small token count most experts
are never read, so counting all `num_experts` would overstate achieved bandwidth.
This matches the upstream Humming benchmark's byte model, but it makes `GB/s`
sensitive to the routing sample: two runs with different seeds activate different
expert counts (each expert is ~16 MB of weight+scale), so `GB/s` can differ by tens
even at the same kernel time.

The `us` column is itself sensitive to the routing draw: per-expert block
padding makes the active block count — and therefore the launched work — vary
by several percent between seeds. The cases here reproduce the upstream
`bench_humming.py` draw exactly (the global CUDA RNG seeded with the token
count), so `us` is comparable one-to-one with a Humming run at the same token
count. A comparison under any other routing must feed both kernels the same
`sorted_ids`/`expert_ids` buffers. Absolute `us` also depends on the GPU and
its clock/power state — a shared or thermally throttled card reads slower — so
cross-implementation comparisons must run on the same idle device.

## What cos_diff measures

The reference is a plain-PyTorch W4A16 GEMM that dequantizes in BF16 with the BF16
group-32 scale and accumulates in FP32 — the same arithmetic the kernel performs.
`cos_diff` therefore reports **engineering error only**: layout, routing, and
pipelining differences between the kernel and a straightforward implementation of
the same math.

It deliberately does not measure the accuracy cost of 4-bit quantization. A
reference that dequantized in FP32 instead would fold the quantization algorithm's
own error into the number, which is both much larger and irrelevant to whether the
kernel is correct — real kernel bugs would hide underneath it.
