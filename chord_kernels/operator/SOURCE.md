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

## DeepGEMM W4A16 provenance

The DeepGEMM-layout SM90 W4A16 backend (masked decode and contiguous prefill
grouped GEMMs) is a secondary development on a second public upstream:

- Public upstream: <https://github.com/deepseek-ai/DeepGEMM>
- Public baseline: `7f2a703ed51ac1f7af07f5e1453b2d3267d37d50`
- License: MIT (see `include/deep_gemm/LICENSE`)
- Third-party CUDA headers carried inside that closure: NVIDIA CUTLASS/CuTe
  (BSD-3-Clause), supplied unmodified by the `third_party/cutlass` submodule
  pinned to `v4.2.0` (`59b61c606fe7aa4d33e60d3e6352cbfde98361c3`)

The vendored unit is the persistent SM90 WGMMA kernel upstream calls
`deep_gemm::sm90_fp8_gemm_1d2d_impl`, instantiated for W4A16 (BF16 activation,
packed-INT4 weight, BF16 group-32 scales on the SFA TMA slot), its static
include closure under `include/deep_gemm/`, and the CUTLASS/CuTe arch headers
that closure needs under `include/cute/` + `include/cutlass/`. The upstream
Python package (`deep_gemm`), its JIT cache, its pybind module, and all other
GEMM families are NOT part of this distribution: the kernel is compiled by the
chord NVRTC path and launched by the chord cubin launcher, exactly like the
phase-1 indexed kernel.

### Vendored DeepGEMM-derived files

Copied under `chord_kernels/operator/include/` apart from package location,
with development-log references in comments neutralized (code unchanged):

- `deep_gemm/impls/sm90_w4a16_gemm.cuh` (the kernel, with the W4A16
  instantiation family) — upstream `impls/sm90_fp8_gemm_1d2d.cuh`, renamed
  along with its entry point `sm90_fp8_gemm_1d2d_impl` →
  `sm90_w4a16_gemm_impl`, because W4A16 is the only instantiation this
  distribution ships and the FP8 name misdescribes it. Contents are otherwise
  unchanged apart from the `ld_shared` overload noted below.
- `deep_gemm/common/{compile,exception,math,tma_copy,types,utils}.cuh`
- `deep_gemm/mma/sm90.cuh` (int4 dequant RS-WGMMA additions)
- `deep_gemm/epilogue/transform.cuh`
- `deep_gemm/ptx/{ld_st,utils,wgmma}.cuh`
- `deep_gemm/scheduler/gemm.cuh`
- `deep_gemm/LICENSE` (MIT, DeepSeek)
The CUTLASS/CuTe headers are not copied into this repository. `third_party/cutlass`
is a git submodule pinned to `v4.2.0`, and the build stages the transitive
include closure of the kernel sources (58 arch-level headers under `cute/` and
`cutlass/`; no CUTLASS library layer) into `include/`, together with
`cutlass/LICENSE.txt` (BSD-3-Clause, NVIDIA). `scripts/cutlass_closure.py`
computes that closure by scanning the `#include` graph, so the packaged set
follows the kernel sources instead of a hand-maintained list. The staged
directories `include/cute/` and `include/cutlass/` are gitignored build output.

Ported to Python/C++ (rewritten against the upstream sources, not copied):

- `grouped/heuristics.py` — the `is_w4a16` branches of
  `csrc/jit_kernels/heuristics/sm90.hpp` (`SM90ArchSpec`): masked/contiguous
  BLOCK_M/N/K tables, stage targeting, the L1/L2 cycle model for the
  (cluster-1/2) layout candidate pick, and the environment override hooks
  renamed `CHORD_W4A16_*`.
- `grouped/packing.py` — the closed-form `(row, nibble)` bit-permutation
  weight reorder from `tests/generators.py::reorder_w4a16` (excess-8
  pre-flip), checkpoint-unpacked and INT32-packed input forms, and the
  one-time scale transpose into the kernel's MN-major `[G, K/32, N]` SFA
  layout.
- `grouped/kernel.py` — the code-generation template of
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp::generate_impl` (SHAPE_M
  dynamic, `compiled_dims="nk"`, identity epilogue, upstream's `kMajorSFB=K`
  spelling; the slot is dead under the W4A16 instantiation), plus
  `extern "C" __constant__` metadata exports consumed by the cubin launcher.
- `csrc/launcher/launcher.cpp` (DeepGEMM section) — the TMA descriptor
  builders of `csrc/jit_kernels/impls/runtime_utils.hpp`
  (`make_tma_{a,b,cd,sf}_desc`) specialized to K-major A/B, BF16 D and the
  MN-major BF16 scale layout, plus the cluster/PDL launch plumbing of
  `csrc/jit/handle.hpp`.
- `grouped/api.py` — the tensor contracts of `csrc/apis/gemm.hpp`
  (`sm90_m_grouped_w4a16_gemm_nt_masked` / `..._contiguous`), with the
  packed-mode guard reversed into a hard error (masked BLOCK_K=128 vs
  contiguous BLOCK_K=64 buffers are not interchangeable).

Weight-pack semantics were independently re-derived from the vendored kernel:
the packed nibble equals the unsigned checkpoint code, and the reorder is a
pure permutation, so the offline flip introduces no arithmetic change.

### Adapting the kernel to NVRTC

Upstream DeepGEMM compiles this kernel with NVCC (`DG_JIT_USE_NVRTC` defaults
to 0). This distribution compiles it with NVRTC instead, like the phase-1
indexed kernel, which keeps the deploy-without-a-CUDA-toolkit property. Three
changes carry that over; they are not upstream behaviour changes.

- **Device prelude.** NVRTC provides no host standard library, and CUTLASS
  v4.2.0 routes `<type_traits>` through `CUDA_STD_HEADER(...)` under
  `__CUDACC_RTC__` (`include/cutlass/cutlass.h`), so nothing in the include
  closure includes a bare `<type_traits>` and the `--header` shims this package
  already used for phase-1 never fire. `Compiler.device_prelude()`
  (`jit/compiler.py`) is a hook, empty in the base class so phase-1 codegen and
  its cache keys are unchanged; `compile()` prepends the returned text and folds
  it into the cache signature. `GroupedNVRTCCompiler.device_prelude()`
  (`grouped/kernel.py`) maps the three `std::` names the kernel spells
  (`conditional_t`, `min`, `forward`) onto libcu++, and declares the two
  programmatic-dependent-launch builtins NVRTC does not provide
  (`cudaGridDependencySynchronize`,
  `cudaTriggerProgrammaticLaunchCompletion`) as `griddepcontrol` inline asm. The
  `std::` import list is a module-level constant shared with the phase-1
  `--header` shims, so the two paths cannot drift. Both mechanisms are retained
  deliberately: the shims fire only for translation units that include the named
  header, which is the right granularity for the phase-1 closure, whereas the
  prelude is unconditional and is needed precisely because this closure includes
  no such header. Moving phase-1 onto the prelude would inject text into every
  translation unit and invalidate its cubin cache for no functional gain.
- **`include/deep_gemm/ptx/ld_st.cuh`** gains an `ld_shared` overload for
  `const __nv_bfloat16*`. The W4A16 instantiation types the SFA slot as
  `__nv_bfloat16` (`TSF`), so the FP8 path's call in
  `impls/sm90_w4a16_gemm.cuh` must name-resolve for that type. That call site
  sits after the `if constexpr (kIsW4A16)` block's `continue`, so it is
  discarded for W4A16: the overload satisfies overload resolution only and never
  executes. NVCC rejects the unresolved name too, so this is independent of the
  compiler choice.
- **`utils/jit.py`** — `find_kernel_name_in_cubin` previously matched mangled
  names with the regex `^_ZN(?:\d+[A-Za-z_]\w*)*\d+{keyword}`, which backtracks
  catastrophically on the long template-argument suffixes this kernel mangles to
  (a single symbol did not resolve in five minutes). `_mangled_name_matches()`
  walks the Itanium length-prefixed components directly instead, with the same
  observable behaviour. Phase-1 was unaffected only because its mangled names
  are short.

NVCC was measured as the alternative on the phase-1 kernel and found equivalent
(identical instruction count and register/shared usage; differences confined to
instruction selection), so the choice rests on the packaging property rather
than on codegen.

## Modified source inventory

Files retained from, rewritten from, or otherwise modified relative to the
public baseline are listed below. Files in this directory that are not listed
were copied without content changes apart from their package location.

Python integration and runtime:

- `__init__.py`
- `api.py`
- `config/__init__.py`
- `config/mma.py`
- `dispatch.py`
- `dtypes.py`
- `env.py`
- `jit/__init__.py`
- `jit/compiler.py`
- `jit/runtime.py`
- `kernel/__init__.py`
- `kernel/humming.py`
- `kernel/repack_weight.py`
- `layer.py`
- `ops.py`
- `packing.py`
- `profiles.py`
- `tuning.py`
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

Relative to the DeepGEMM baseline (see "DeepGEMM W4A16 provenance" for the
ported Python and launcher files, which are rewritten rather than copied):

- `include/deep_gemm/ptx/ld_st.cuh` — one added `ld_shared` overload for
  `const __nv_bfloat16*`, required for name resolution in a branch the W4A16
  instantiation discards
- `grouped/{__init__,api,heuristics,kernel,packing}.py` — ported, not copied

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
- `layer.py` is a framework-facing adapter around the operator APIs. It owns
  the packed weight lifecycle but does not copy an upstream model, router, or
  quantization registry, and it is backend-agnostic: the published profiles
  live in `profiles.py`, the process-level switches in `env.py`, the indexed
  launch tables in `tuning.py`, and every backend-specific pack/forward call
  goes through `dispatch.py`.
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
`tests/test_w4a16_indexed.py`, which checks the kernel output against a plain-PyTorch
reference on the running device before timing it. Public APIs reject
unsupported compute capabilities or profile combinations.

The DeepGEMM-backend masked and contiguous paths are exercised by
`tests/test_w4a16_grouped.py` against an independent BF16 dequant reference
on the running SM90 device; the launch heuristics are additionally covered by
torch-free unit tests of the ported selection tables.
