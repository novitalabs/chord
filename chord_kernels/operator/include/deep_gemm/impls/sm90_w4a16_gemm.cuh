#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t,
          bool kIsW4A16 = false,
          uint32_t kScaleGroup = 0>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_w4a16_gemm_impl(float* sfb, int* grouped_layout,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(kIsW4A16 ? (BLOCK_K == 64 or BLOCK_K == 128 or BLOCK_K == 256) : (BLOCK_K == 128), "Invalid BLOCK_K for the activation dtype");
    DG_STATIC_ASSERT(kIsW4A16 or
        math::constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or
        (math::constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    // W4A16: activation is BF16 (2 bytes), weight is INT4 (packed 2-per-byte)
    using TA = std::conditional_t<kIsW4A16, __nv_bfloat16, __nv_fp8_e4m3>;
    constexpr int WEIGHT_RATIO = kIsW4A16 ? 2 : 1;
    // W4A16 RS-mode swaps layout↔compute dimensions: WGMMA M=64 tiles along BLOCK_N, MMA_N=BLOCK_M
    constexpr uint32_t COMPUTE_M = kIsW4A16 ? BLOCK_N : BLOCK_M;
    constexpr uint32_t COMPUTE_N = kIsW4A16 ? BLOCK_M : BLOCK_N;
    constexpr int MMA_N = kIsW4A16 ? (SHAPE_M != 0 ? SHAPE_M : COMPUTE_N) : COMPUTE_N;
    using WGMMA = std::conditional_t<kIsW4A16,
                                     typename mma::sm90::BF16MMASelector<MMA_N>::type,
                                     typename mma::sm90::FP8MMASelector<MMA_N>::type>;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    if constexpr (kIsW4A16) {
        DG_STATIC_ASSERT(BLOCK_M <= 256, "keep BLOCK_M <= 256, if use w4a16.");
        DG_STATIC_ASSERT(BLOCK_N % WGMMA::M == 0, "keep BLOCK_N % WGMMA::M == 0, if use w4a16.");
    } else {
        DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");
    }

    // W4A16: number of INT4 weight scale sub-groups per BLOCK_K (scale group along K = kScaleGroup, e.g. 32)
    constexpr uint32_t kScaleSub = kIsW4A16 ? (BLOCK_K / kScaleGroup) : 1;
    if constexpr (kIsW4A16) {
        DG_STATIC_ASSERT(kScaleGroup > 0 and BLOCK_K % kScaleGroup == 0, "Invalid W4A16 scale group");
        DG_STATIC_ASSERT(kScaleGroup % WGMMA::K == 0, "Scale group must be a multiple of WGMMA::K (16)");
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = kIsW4A16 ? true : (BLOCK_K % BLOCK_N == 0);
    // W4A16 weight scales are BF16 (matches the real model; halves SFA bandwidth + smem).
    using TSF = std::conditional_t<kIsW4A16, __nv_bfloat16, float>;
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(TA);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3) / WEIGHT_RATIO;
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = kScaleSub * COMPUTE_M * sizeof(TSF);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = math::ceil_div(shape_n, BLOCK_K);
    const uint32_t smem_sfb_size = kIsW4A16 ? 0 : math::align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier));

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(TA);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<TA*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<TSF*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    auto full_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA A
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, TA, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                             num_tma_multicast_a, batch_idx);
                    // W4A16: SFA slot carries weight scales (block_n dim, indexed by n_block_idx)
                    if constexpr (kIsW4A16) {
                        // INT4 weight scales with per-`kScaleGroup` granularity along K.
                        // SF tensor K-length = shape_k / kScaleGroup; per block load `kScaleSub` rows.
                        // Group offset into the [G, K/group, N] scale tensor: masked tracks the group in
                        // `current_group_idx`, contiguous resolves it per-M-block from `grouped_layout`
                        // (m_indices), matching how the weight B is indexed via get_global_idx.
                        const uint32_t sf_k_len = shape_k / kScaleGroup;
                        uint32_t group_off = 0;
                        if constexpr (kGemmType == GemmType::MGroupedMasked)
                            group_off = scheduler.current_group_idx * sf_k_len;
                        else if constexpr (kGemmType == GemmType::MGroupedContiguous)
                            group_off = cute::max(0, scheduler.grouped_layout[m_block_idx * BLOCK_M]) * sf_k_len;
                        #pragma unroll
                        for (uint32_t s = 0; s < kScaleSub; ++ s) {
                            tma::copy<COMPUTE_M, kScaleGroup, 0>(&tensor_map_sfa, &full_barrier,
                                     smem_sfa[stage_idx] + s * COMPUTE_M, n_block_idx * BLOCK_N,
                                     group_off + k_block_idx * kScaleSub + s,
                                     num_tma_multicast_b);
                        }
                    } else {
                        tma::copy<COMPUTE_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                                 smem_sfa[stage_idx], m_block_idx * BLOCK_M,
                                 scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales,
                                                                                      1,
                                                                                      k_block_idx,
                                                                                      (kGemmType == GemmType::MGroupedContiguous) ? m_block_idx : 0),
                                 num_tma_multicast_a);
                    }

                    // Issue TMA B
                    tma::copy<BLOCK_K / WEIGHT_RATIO, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                             smem_b[stage_idx], k_idx / WEIGHT_RATIO, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b, batch_idx);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                                                      SMEM_SFA_SIZE_PER_STAGE);
                }
            }

            // PDL post-processing: hint that the dependent (secondary) grid may start being scheduled.
            // NOTES: this is the producer's counterpart to the `cudaGridDependencySynchronize()`
            // pre-processing wait above, and is governed by the same switch (`deep_gemm.set_pdl`),
            // which toggles the `ProgrammaticStreamSerialization` launch attribute on the host side.
            // Without that attribute the instruction is simply a no-op, so it is always compiled in.
            //
            // Placement: every global load for every tile this CTA owns has been issued by now, so only
            // the pipeline drain and the epilogue stores remain. That matches where CUTLASS triggers
            // (right before the epilogue), and is a few k-blocks earlier because the producer runs
            // ahead of the math warp-groups by at most `kNumStages`.
            //
            // Safety: the kick-off only enables *scheduling* of the secondary kernel and carries no
            // memory-visibility guarantee, so firing it before the epilogue writes D is safe -- a
            // dependent kernel that reads D must itself call `cudaGridDependencySynchronize()`, which
            // blocks until this grid has completed and flushed its results to global memory.
            // One trigger per CTA is enough: the kick-off waits until every CTA has either triggered
            // or exited, so the single elected producer thread here covers the whole CTA.
            cudaTriggerProgrammaticLaunchCompletion();

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);

        const auto r_0 = warp_idx * 16 + lane_idx / 4;
        const auto r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + (kIsW4A16 ? 0 : math_wg_idx * WGMMA::M * BLOCK_K), 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0] + (kIsW4A16 ? math_wg_idx * WGMMA::M * BLOCK_K : 0), 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // W4A16: Precompute thread-invariant values for ldmatrix (hoisted out of k-loop)
        constexpr uint32_t NV = 16 / sizeof(__nv_fp8_e4m3);
        const uint32_t tidG = threadIdx.x % 128;
        const uint32_t tRow = (tidG & 15) | ((tidG >> 5) << 4);
        const uint32_t tCol = ((tidG >> 4) & 1) * NV;
        const uint32_t sRow = math_wg_idx * WGMMA::M + tRow;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

            // W4A16: weight scales ride the SFA slot, so there is no global SFB to load
            if constexpr (not kIsW4A16) {
                // Load B scales with math warp-groups
                // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
                if (threadIdx.x >= 32) {
                    auto previous_group_offset = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                    const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                    const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
                    auto local_sfb = sfb + previous_group_offset + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;

                    #pragma unroll
                    for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                        ptx::st_shared(smem_sfb + i, i < shape_k_scales ? local_sfb[i * stride_k_sfb] : local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
            }

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = COMPUTE_M <= WGMMA::M
                                              ? COMPUTE_M
                                              : WGMMA::M * 2;
            DG_STATIC_ASSERT(COMPUTE_M % WAVE_BLOCK_M == 0, "Invalid block sizes");

            constexpr int WAVE_WGMMA = COMPUTE_M / WAVE_BLOCK_M;
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * WAVE_WGMMA] = {0};

            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(COMPUTE_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // W4A16 wait<1> overlap: release a *specific* stage's empty barrier (the previous
            // k-block's) rather than the current `stage_idx`. Used when we keep one WGMMA group in flight
            // (wait<1>) so smem must only be released after that group has actually drained.
            auto empty_barrier_arrive_stage = [&](uint32_t st) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[st]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[st]->arrive(target_cta) : void();
                }
            };

            // Skip useless computations
            if (scheduler.is_computation_valid(m_block_idx, kIsW4A16 ? 0 : math_wg_idx * WGMMA::M)) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / math::constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = math::constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // Dispatch `num_former_iters` and launch MMAs
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    // W4A16 wait<1> overlap: keep the previous k-block's
                    // WGMMA group in flight so it overlaps this k-block's ldmatrix + int4->bf16 dequant,
                    // instead of draining (wait<0>) after every group. We must defer each stage's empty
                    // barrier release until its WGMMA group has drained (the activation operand B is read
                    // from smem for the whole async op); track the previous stage here.
                    // GATED: fire wait<1> when the kernel is compute/latency-bound so the
                    // overlap pays, and NOT when it is bandwidth-bound (there, deferring the empty-barrier
                    // release just delays the TMA producer → measured −6..10%). Two cases qualify:
                    //   (a) MMA_N>=72 (m56/m64): WGMMA is long (~44 cyc) → tensor-bound regardless of BN.
                    //   (b) COMPUTE_M>=256 (BN256, i.e. tile-rich gateup) AND MMA_N>=48 (m32-48): the "valley"
                    //       where DRAM~48-54% AND tensor~48-61% (NEITHER wall) — barrier-bound. wait<1>
                    //       drops barrier stall (m48: 1.44→0.97, tensor 61→65%), +4..8% on g24 gate m32-48.
                    // Excluded: small MMA_N (<48, m8-24 bandwidth-bound) and BN128 mid-m (g12 gate, few
                    // tiles → memory-bound → deferring release hurts, −6%).
                    constexpr bool kW4A16Wait1 = kIsW4A16 and (MMA_N >= 72 or (COMPUTE_M >= 256 and MMA_N >= 48));
                    bool w4a16_prev_valid = false;
                    uint32_t w4a16_prev_stage = 0;
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        float scale_b_0;
                        float scale_b_1;

                        if constexpr (not kIsW4A16) {
                            // Read B scales
                            scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx);
                            // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                            if constexpr (not kMustUseUniformedScaleB)
                                scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);
                        }

                        // Wait TMA arrivals
                        full_barriers[stage_idx]->wait(phase);

                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < WAVE_WGMMA; ++ local_idx) {
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // W4A16: INT4 weights in registers (operand A), BF16 activations in smem (operand B desc).
                            // BLOCK_K = 64 -> one ldmatrix.x4 (32 nibbles/thread) -> BLOCK_K/16 K=16 RS operands.
                            // Weight scales have per-`kScaleGroup` granularity along K (kScaleSub sub-groups per block);
                            // each sub-group is promoted independently into `final_accum`.
                            if constexpr (kIsW4A16) {
                                using WGMMARS = typename mma::sm90::BF16MMASelectorRS<MMA_N>::type;
                                constexpr uint32_t kOpsPerSub = kScaleGroup / WGMMA::K;
                                constexpr uint32_t kNumK16 = BLOCK_K / WGMMA::K;          // K=16 operands per block
                                // One ldmatrix.x4 covers 32 weight bytes = 64 nibbles = 4 K16 operands.
                                constexpr uint32_t kLdsmChunks = (BLOCK_K / 2) / 32;       // = BLOCK_K / 64
                                // Activation (operand B desc) swizzle-atom size along K (in TA elements).
                                constexpr uint32_t BLOCK_ATOM_K = kSwizzleAMode / sizeof(TA);

                                // Read all per-sub-group weight scales up front (before `warpgroup_arrive`).
                                // W4A16 scales are BF16; kept as BF16 to fold directly into the dequantized
                                // weight (scale-on-weight), removing the fp32 scale-promote FMA + dual bank.
                                __nv_bfloat16 scale_w_0[kScaleSub], scale_w_1[kScaleSub];
                                #pragma unroll
                                for (uint32_t s = 0; s < kScaleSub; ++ s) {
                                    scale_w_0[s] = smem_sfa[stage_idx][s * COMPUTE_M + r_0 + m_offset];
                                    scale_w_1[s] = smem_sfa[stage_idx][s * COMPUTE_M + r_1 + m_offset];
                                }

                                // Interleaved software pipeline (cf. PR287 fp8-W4): overlap the ALU-heavy
                                // int4->bf16 dequant of the *next* K16 operand with the WGMMA of the *current*
                                // one, using a 2-deep ping-pong of `unpackB`. One ldmatrix.x4 yields 4 operands
                                // (a 64-nibble chunk); we dequant operand-by-operand to keep the convert close
                                // to its WGMMA. Per-`kScaleGroup` subgroup keeps its own accum bank (scale on C).
                                // Interleaved software pipeline: overlap the int4->bf16 dequant of the
                                // *next* K16 operand with the WGMMA of the *current* one (2-deep ping-pong).
                                // Scale-on-weight: the per-(N,K-group) weight scale is folded into the bf16
                                // weight in the convert (scale_w_0 -> even regs / row r_0, scale_w_1 -> odd
                                // regs / row r_1), so WGMMA accumulates straight into the single `final_accum`
                                // across the whole K-loop — no separate fp32 promote, no dual accum bank.
                                const __nv_fp8_e4m3* smem_b_w4_ptr = smem_b[stage_idx] + (sRow + m_offset) * (BLOCK_K / 2);
                                uint32_t fragB[kNumK16];   // one ldmatrix reg == one K16 operand
                                uint32_t unpackB[2][4];    // ping-pong: only 2 operands' bf16 live at a time
                                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;

                                auto ldsm_chunk = [&](uint32_t c) {
                                    const uint32_t pCol = ptx::permute_col<BLOCK_K / 2 * sizeof(__nv_fp8_e4m3), NV>(tRow, c * 32 + tCol);
                                    uint32_t* fb = fragB + c * 4;
                                    ptx::SM90_U32x4_LDSM_N::copy(fb[0], fb[1], fb[2], fb[3], (void*)(smem_b_w4_ptr + pCol));
                                };
                                auto a_desc_for = [&](uint32_t rr) {
                                    const uint32_t k_elem = rr * WGMMA::K;
                                    const uint32_t atom_k_idx = k_elem / BLOCK_ATOM_K;
                                    return mma::sm90::advance_gmma_desc_lo<
                                        cute::UMMA::Major::K, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, TA>(
                                        a_desc_base_lo, 0, k_elem % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                                };
                                auto cvt_scaled = [&](uint32_t* out, uint32_t frag, uint32_t rr) {
                                    const uint32_t s = rr / kOpsPerSub;  // scale subgroup for this K16 op
                                    ptx::fast_int4_to_bf16_convert_scaled(out, frag, scale_w_0[s], scale_w_1[s]);
                                };

                                // All ldmatrix up front (cheap LSU, lets dequant/wgmma overlap below).
                                #pragma unroll
                                for (uint32_t c = 0; c < kLdsmChunks; ++ c) ldsm_chunk(c);

                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(shifted_accum[i]);

                                // Prologue: dequant+scale operand 0.
                                cvt_scaled(unpackB[0], fragB[0], 0);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t rr = 0; rr < kNumK16; ++ rr) {
                                    if (rr + 1 < kNumK16)
                                        cvt_scaled(unpackB[(rr + 1) & 1], fragB[rr + 1], rr + 1);
                                    // Accumulate into `final_accum` across the entire K-loop: overwrite only
                                    // on the very first WGMMA (first op of the first k-block), add thereafter.
                                    const bool accumulate = (k_block_idx != 0) or (rr != 0);
                                    a_desc.reg32_[0] = a_desc_for(rr);
                                    WGMMARS::wgmma(unpackB[rr & 1], a_desc, shifted_accum, accumulate);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(shifted_accum[i]);
                                if constexpr (kW4A16Wait1) {
                                    // wait<1>: keep THIS k-block's WGMMA group in flight so it overlaps the
                                    // next k-block's ldmatrix + int4->bf16 dequant. `shifted_accum` regs are
                                    // only read at the epilogue (no read between groups), and register operand
                                    // A (unpackB) is latched at issue (the existing distance-2 ping-pong
                                    // already relies on this), so 1 group in flight is bit-safe. The only
                                    // lifetime constraint is the activation operand B in smem: don't let TMA
                                    // overwrite a stage until its WGMMA group has drained → release the
                                    // PREVIOUS k-block's empty barrier now (by the time we commit this group,
                                    // the previous one has completed, since at most 1 group is in flight).
                                    ptx::warpgroup_wait<1>();
                                    if (local_idx == WAVE_WGMMA - 1) {
                                        if (w4a16_prev_valid)
                                            empty_barrier_arrive_stage(w4a16_prev_stage);
                                        w4a16_prev_valid = true;
                                        w4a16_prev_stage = stage_idx;
                                    }
                                } else {
                                    ptx::warpgroup_wait<0>();
                                    if (local_idx == WAVE_WGMMA - 1)
                                        empty_barrier_arrive();
                                }
                                continue;
                            }

                            // Read A scales
                            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                            const float scale_a_0 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
                            const float scale_a_1 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_1 + m_offset) : 0;

                            // Commit WGMMA instructions
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();

                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                WGMMA::wgmma(a_desc, b_desc, accum, k);
                            }

                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // Notify barrier arrival at the last warpgroup wave
                            if (local_idx == WAVE_WGMMA - 1)
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store)
                                continue;

                            // Promote with scales
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                            float scale_0_1, scale_1_1;
                            if constexpr (not kMustUseUniformedScaleB)
                                scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                                const bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                                shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                            }
                        }
                    }

                    // W4A16 wait<1> tail: the last k-block left its WGMMA group in flight and did not
                    // release its stage's empty barrier. Drain it now (accumulator is about to be read by
                    // the epilogue) and release that final stage back to the producer.
                    if constexpr (kW4A16Wait1) {
                        if (w4a16_prev_valid) {
                            ptx::warpgroup_wait<0>();
                            empty_barrier_arrive_stage(w4a16_prev_stage);
                        }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < WAVE_WGMMA; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        constexpr uint32_t kNumBankGroupBytes = 16;

                        if constexpr (kIsW4A16) {
                            auto row = i * 8 + lane_idx % 8;
                            auto col = (warp_idx % 4) * 2 + lane_idx / 8;
                            col ^= row % (kSwizzleDMode / 16);

                            auto n_atom_idx = m_offset / WGMMA::M + math_wg_idx;
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                                    n_atom_idx * BLOCK_M * kSwizzleDMode +
                                    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                        } else {
                            // Calculate the swizzling atom offset and in-atom offset
                            auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                            // Calculate the index of the bank group to be written in the atom
                            auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                            // Reshape the atom in another view and swizzle
                            //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                            //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                            constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                            col ^= row % (kSwizzleDMode / 16);

                            // Add back into the base pointer
                            // NOTES: think twice before modifying this, as changes may affect the number of instructions
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                                warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                                m_offset * kSwizzleDMode +                                 // Wave offset
                                atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                        }
                    } else {
                        if constexpr (kIsW4A16) {
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d +
                                                                  m_offset +
                                                                  warp_idx * WGMMA_M_PER_WARP +
                                                                  lane_idx / 8 * 8 + (lane_idx % 8) * BLOCK_N +
                                                                  BLOCK_N * i * 8);
                        } else {
                            // No swizzling, just padding
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                        }
                    }

                    // NOTES: only 16 lanes' addresses are used
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::template copy<kIsW4A16>(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                  n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
