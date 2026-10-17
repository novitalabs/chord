#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

namespace deep_gemm::ptx {

/// W4A16 helper: swizzle-aware column permutation for the INT4 weight ldmatrix
template <uint32_t COL_BYTES, uint32_t NV>
CUTLASS_DEVICE auto permute_col(const uint32_t sRow, const uint32_t sCol) {
    constexpr uint32_t strd = 128 / std::min(uint32_t(128), COL_BYTES);
    const uint32_t pCol = (sCol / NV) ^ (sRow % 8 / strd);
    return pCol * NV;
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

} // namespace deep_gemm::ptx
