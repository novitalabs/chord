#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

namespace deep_gemm::ptx {

// Compatibility: 256 bits LD/ST instructions
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
using longlong4_t = longlong4_32a;
#define make_longlong4_t make_longlong4_32a
#else
struct alignas(32) longlong4_t { long long x, y, z, w; };
CUTLASS_HOST_DEVICE longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};
}
#endif

/// W4A16 helper: swizzle-aware column permutation for the INT4 weight ldmatrix
template <uint32_t COL_BYTES, uint32_t NV>
CUTLASS_DEVICE auto permute_col(const uint32_t sRow, const uint32_t sCol) {
    constexpr uint32_t strd = 128 / std::min(uint32_t(128), COL_BYTES);
    const uint32_t pCol = (sCol / NV) ^ (sRow % 8 / strd);
    return pCol * NV;
}

// W4A16 helper: dequantize 8 int4 nibbles (packed in one uint32) into 8 bf16 values, written as
// 4 uint32 (each holds a bf16x2). `outputs[ii]` packs values `ii` (low) and `ii+4` (high).
// Standard W4A16 LOP3 dequant (cf. Marlin / lmdeploy / TRT-LLM): one `lop3` does the
// `(i4s & 0x000F000F) | 0x43004300` mask+exponent-inject; `0x4300|u` is `128.0f + u` in bf16,
// and `hsub2(136)` recovers the signed value.
// NOTE: the input nibbles must already be in **excess-8** form (two's-complement XOR 0x8); this
// sign-flip is baked into the weight reorder/pack offline, saving one LOP3 per convert on the
// binding ALU pipe (vs flipping `^0x88888888` here every K-iteration).
// NOTE: output value ordering (ii, ii+4) is what the INT4 reorder (tests/w4a16_perm*.json) is
// calibrated against.
CUTLASS_DEVICE void fast_int4_to_bf16_convert(uint32_t outputs[4], uint32_t input) {
    uint32_t i4s = input;  // already excess-8 (sign-flip folded into the pack)
    constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
    constexpr uint32_t kMask = 0x000F000Fu, kMagic = 0x43004300u;
    const __nv_bfloat16 bias = __float2bfloat16(136.0f);
    __nv_bfloat162 bias2;
    reinterpret_cast<__nv_bfloat16*>(&bias2)[0] = bias;
    reinterpret_cast<__nv_bfloat16*>(&bias2)[1] = bias;
    #pragma unroll
    for (int ii = 0; ii < 4; ++ ii) {
        uint32_t w;
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(w) : "r"(i4s), "n"(kMask), "n"(kMagic), "n"(kImmLut));
        __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&w);
        v = __hsub2(v, bias2);
        outputs[ii] = *reinterpret_cast<uint32_t*>(&v);
        i4s >>= 4;
    }
}

// int4 -> bf16 dequant with the per-(N,K-group) weight scale folded directly into the bf16 weight
// fragment (W4A16). Each of the 4 output registers (one K16 RS A-operand pair) maps entirely to ONE
// weight-N row: register `ii` -> row r_0 if ii%2==0, else r_1 (from the verified bf16 RS A-fragment
// formula m = (t/4)%8 + 8*((j/2)%2) + 16*(t/32), where (j/2)%2 == ii%2). Folding the scale here is
// numerically exact (just bf16-rounds w*scale) and removes the separate fp32 scale-promote FMA and the
// dual fp32 accumulator bank from the K-loop, cutting both the FMA pipe pressure and register usage.
CUTLASS_DEVICE void fast_int4_to_bf16_convert_scaled(uint32_t outputs[4], uint32_t input,
                                                     __nv_bfloat16 s0, __nv_bfloat16 s1) {
    uint32_t i4s = input;  // already excess-8
    constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
    constexpr uint32_t kMask = 0x000F000Fu, kMagic = 0x43004300u;
    // out = (raw - 136) * scale. Keep the bias-subtract SEPARATE from the scale-multiply: raw is 128+u
    // (exact in bf16), so (raw-136) is computed without cancellation error, then scaled. Fusing into
    // hfma2(raw, scale, -136*scale) was tried and REJECTED — the bf16-rounded -136*scale plus the
    // raw*scale ≈ 136*scale near-cancellation pushed rel error to ~1.5e-3, over the 1e-3 gate.
    const __nv_bfloat16 bias = __float2bfloat16(136.0f);
    __nv_bfloat162 bias2; reinterpret_cast<__nv_bfloat16*>(&bias2)[0] = bias;
    reinterpret_cast<__nv_bfloat16*>(&bias2)[1] = bias;
    __nv_bfloat162 sc[2];  // sc[ii%2] broadcasts the row scale across both bf16 lanes of the register
    reinterpret_cast<__nv_bfloat16*>(&sc[0])[0] = s0; reinterpret_cast<__nv_bfloat16*>(&sc[0])[1] = s0;
    reinterpret_cast<__nv_bfloat16*>(&sc[1])[0] = s1; reinterpret_cast<__nv_bfloat16*>(&sc[1])[1] = s1;
    #pragma unroll
    for (int ii = 0; ii < 4; ++ ii) {
        uint32_t w;
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(w) : "r"(i4s), "n"(kMask), "n"(kMagic), "n"(kImmLut));
        __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&w);
        v = __hmul2(__hsub2(v, bias2), sc[ii & 1]);
        outputs[ii] = *reinterpret_cast<uint32_t*>(&v);
        i4s >>= 4;
    }
}

/// LD/ST matrix
// TODO: remove `struct`
struct SM90_U32x2_LDSM_N {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst_0), "=r"(dst_1)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

struct SM90_U32x4_LDSM_N {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    // `kTrans` selects the transposed store, needed by the swap-AB (W4A16) epilogue
    template<bool kTrans = false>
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        if constexpr (kTrans) {
            asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16.trans [%0], {%1, %2};\n"
                         :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
        } else {
            asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                         :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
        }
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_T {
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
                                 *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)),
                        "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

template <typename dtype_t>
struct SM100_U8x4_STSM_T {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src = *reinterpret_cast<uint32_t*>(&src_0);
        asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src));
    }
};

template <typename dtype_t>
struct SM100_U8x8_STSM_T {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

/// Shared memory
CUTLASS_DEVICE uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

// The W4A16 instantiation types the SFA slot as `__nv_bfloat16` (`TSF`), so the
// FP8 path's `ld_shared(smem_sfa[...])` call has to name-resolve for it too. That
// call sits after the `if constexpr (kIsW4A16)` block's `continue`, so it is
// discarded for W4A16 and this overload only has to satisfy overload resolution.
CUTLASS_DEVICE float ld_shared(const __nv_bfloat16* ptr) {
    uint16_t ret;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(ret));
}

CUTLASS_DEVICE void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val));
}

CUTLASS_DEVICE void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

CUTLASS_DEVICE void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

CUTLASS_DEVICE void st_shared(const __int128_t* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}

CUTLASS_DEVICE void st_shared_bulk(void* smem_ptr, const uint32_t& num_bytes) {
    // `size` must be 64-bit before PTX ISA 9.0
    asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;" ::
                 "l"(__cvta_generic_to_shared(smem_ptr)), "l"(static_cast<uint64_t>(num_bytes)));
}

/// Global memory
CUTLASS_DEVICE uint64_t ld_volatile(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_sys(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE void st_relaxed_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("st.L1::no_allocate.relaxed.sys.u64 [%0], %1;" :: "l"(ptr), "l"(value));
}

/// Atomics
CUTLASS_DEVICE uint64_t atomic_add(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

CUTLASS_DEVICE uint64_t atomic_add_sys(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

CUTLASS_DEVICE uint32_t atomic_add_rel(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__forceinline__ __device__ void red_add(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_or_rel_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.sys.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

CUTLASS_DEVICE void red_or_rel_gpu(uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

CUTLASS_DEVICE void red_add_rel(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_add_rel_sys(const int* ptr, const int& value) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE int ld_acq_sys(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq_sys(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_gpu(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

/// Predicated loads
CUTLASS_DEVICE longlong4_t ld_gez_pred(const longlong4_t* ptr, const int& pred) {
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %5, 0;\n\t"
        "  @p ld.global.L2::256B.v4.s64 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)
        : "l"(ptr), "r"(pred)
        : "memory");
    return ret;
}

/// Prefetch
CUTLASS_DEVICE void prefetch_l1(void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

} // namespace deep_gemm::ptx
