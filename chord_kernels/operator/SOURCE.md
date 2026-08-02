# Operator source provenance

- Public upstream: <https://github.com/inclusionAI/humming>
- Public baseline: `4351af3a8fcdce1a8dee50104ba49566af2427fb`
- License: Apache-2.0 (see `LICENSE` in this directory)
- Upstream NOTICE: none at the public baseline

This directory is a derivative of that public baseline. It contains the source
closure needed by one BF16 activation, INT4 weight, group-32 scaled, indexed
MoE GEMM and its weight repacking kernel. The upstream autotuning framework,
unrelated quantization modes, other GEMM families, benchmarks, and tests are
not part of this distribution. The root `NOTICE` and this file form the central
provenance and modification record; every source file in this directory carries
a two-line pointer back here rather than repeating the description.

## Modified source inventory

Files retained from, rewritten from, or otherwise modified relative to the
public baseline are listed below. Files in this directory that are not listed
were copied without content changes apart from their package location.

Python integration and runtime:

- `__init__.py`
- `api.py`
- `config/__init__.py`
- `config/mma.py`
- `dtypes.py`
- `jit/__init__.py`
- `jit/compiler.py`
- `jit/runtime.py`
- `kernel/__init__.py`
- `kernel/humming.py`
- `kernel/repack_weight.py`
- `layer.py`
- `ops.py`
- `packing.py`
- `utils/cuda.py`
- `utils/jit.py`
- `utils/nvrtc.py`

CUDA launcher:

- `csrc/nvrtc_compile.cpp`
- `csrc/launcher/elf.h`
- `csrc/launcher/launcher.cpp`
- `csrc/launcher/tensor.h`
- `csrc/launcher/torch_api.h`
- `csrc/launcher/utils.h`

CUDA operator and repacking kernels:

- `include/humming/arith/mainloop_arith.cuh`
- `include/humming/datatype/dequant.cuh`
- `include/humming/epilogue/gmem_writer.cuh`
- `include/humming/epilogue/pipeline.cuh`
- `include/humming/epilogue/smem_reducer.cuh`
- `include/humming/epilogue/smem_writer.cuh`
- `include/humming/kernel/humming.cuh`
- `include/humming/kernel/repack.cuh`
- `include/humming/memory/g2s_loader/loader_a.cuh`
- `include/humming/memory/g2s_loader/loader_b.cuh`
- `include/humming/memory/g2s_loader/loader_bs.cuh`
- `include/humming/memory/g2s_pipeline.cuh`
- `include/humming/memory/s2r_loader/loader_a.cuh`
- `include/humming/memory/s2r_loader/loader_b.cuh`
- `include/humming/memory/s2r_loader/loader_bs.cuh`
- `include/humming/memory/s2r_pipeline.cuh`
- `include/humming/mma/wgmma.cuh`
- `include/humming/mma/wmma.cuh`
- `include/humming/scheduler.cuh`
- `include/humming/utils/base.cuh`
- `include/humming/utils/ptx/barrier.cuh`
- `include/humming/utils/ptx/cp_async.cuh`
- `include/humming/utils/ptx/math.cuh`
- `include/humming/utils/ptx/shared.cuh`
- `include/humming/utils/ptx/wgmma.cuh`
- `include/humming/utils/storage.cuh`

## Extraction changes

- The public execution surface was narrowed to INT4 packing and indexed
  W4A16 execution. `PreparedWeight` carries the MMA/WGMMA physical layout and
  logical shape so incompatible launches are rejected. The public Python
  package and import namespace are both `chord_kernels`.
- The Python generator and CUDA configuration are fixed to BF16 activation,
  INT4 weight, BF16 group-32 scale, FP32 accumulation, and indexed routing.
  The retained MMA generator emits only the MMA/WGMMA instruction forms used
  by these profiles; generic dtype, quantization, and GEMM configuration
  objects were removed.
- `layer.py` is a framework-facing adapter around the indexed API. It owns
  profile selection and packed weight lifecycle but does not copy an upstream
  model, router, or quantization registry.
- The launcher now accepts only the tensors needed by indexed W4A16. Generic
  bias, zero-point, input-scale, global-scale, and TMA descriptor handling was
  removed; it does allocate and pass the int32 stream-K lock buffer (see below).
- `include/humming/kernel/repack.cuh` is the retained INT4-to-W4A16 weight
  layout path extracted from the upstream generic processing kernel. The
  Python repacker supplies only packed/unpacked input and MMA/WGMMA layout
  choices.
- The GEMM templates were specialized to BF16 activation, INT4 weight, BF16
  group-32 weight scale, indexed routing, and `cp.async` loads. Dense/grouped
  scheduling, other data types, bias and zero-point paths, activation scaling,
  partial-thread barriers, TMA loads, and TMA multicast
  were removed. Retained CUDA headers include their exact dependencies instead
  of an umbrella utility header.
- Stream-K is retained for the H200 prefill profile: the tail tiles' K dimension
  is split across CTAs that reduce partial sums into the output, gated per the
  upstream Humming rule (gate/up always; the mid-K down projection only below
  routed_m 5120). The indexed, non-multicast subset of the upstream mechanism is
  kept — the two lock protocols (`utils/ptx/barrier.cuh`), the scheduler K-split,
  and the epilogue reduction — using a device-resident zero-initialized int32
  lock buffer that the barrier protocol resets to 0 for reuse; the launcher
  keys the buffer per (device, stream) because reuse-without-reset requires
  ordered launches. Decode stays
  one-pass. Cross-CTA float accumulation makes stream-K output non-bit-identical
  to one-pass, within the group-32 correctness tolerance.
- The inner kernel (WGMMA/MMA mainloop, g2s/s2r pipelines, group-32 dequant)
  keeps the upstream WGMMA batched `wait<1>` pipelining and group-32 dequant
  rounding. The BF16-activation/INT4-weight path is functionally equivalent to
  upstream on the same shapes; block-M (via the same sampled routing), the
  2-CTAs/SM window for block-M 40..80 at block_n 256, and the stream-K gate
  reproduce upstream's choices, so throughput matches its tables within a few
  percent.
- Production kernel JIT uses NVRTC only. The runtime compiler writes a cubin
  into a content-addressed cache, validates its ELF kernel symbol, and loads it
  through the CUDA driver on the target device. CUDA Toolkit and header
  discovery is constrained to the CUDA major version reported by PyTorch.
- The retained CUDA headers keep the upstream `humming/` internal include
  prefix. This package does not import or require the full Python
  `humming` distribution at runtime; its JIT include path is scoped to this
  source closure. JIT artifacts default to `.chord_cache/` and `.chord_tmp/`
  beside the package, with `CHORD_CACHE_DIR` and `CHORD_TMP_DIR` overrides.
- Package-owned integration names remain Chord-specific: the Python package is
  `chord_kernels`, the Torch operator namespace is `chord`, and the launcher
  and cache namespaces use `chord_*`. These are not upstream kernel symbols.
- H200 prefill uses WGMMA. H200 and Blackwell decode use MMA swap-AB for the
  selected small routed-M rows and legal non-swap MMA rows elsewhere. Runtime
  kernel instances and launcher registrations are scoped by CUDA device.
- `blackwell_decode_ep8` is the Blackwell profile name. B200 SM100 and B300
  SM103 share the schedule while JIT compilation targets `sm_100a` and
  `sm_103a` respectively, so their cubins are not shared.

## Validation scope

This extraction supports the H200 SM90 prefill/decode profiles and the
Blackwell decode profile on B200 SM100 and B300 SM103; all three parts have
been compiled, run and timed. Every tuning row is exercised by
`tests/test_w4a16.py`, which checks the kernel output against a plain-PyTorch
reference on the running device before timing it. Public APIs reject
unsupported compute capabilities or profile combinations.
