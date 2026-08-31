// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

// A prefill (large-M) GEMM for u2 weights that reaches the XMX systolic array.
//
// WHY THIS EXISTS
//   oneDNN is the only fully-connected path on this plugin that uses the matrix engine, and it
//   cannot take u2: fully_connected_onednn.hpp restricts compressed weights to {u8,i8,u4,i4}, and
//   separately restricts the decompression zero point to {i4,u4,u8,i8}. The second restriction is
//   the harder one -- the SEQ/Q2_0 grid is w = (q - 1.5) * scale, and no integer type holds 1.5 --
//   so widening the first would not be enough. u2 therefore falls to fully_connected_gpu_bf_tiled,
//   which has no DPAS at all. Measured on Arc B390 at 1024 tokens, that costs 5-6x against f16:
//   4.7-5.3 TFLOP/s versus 24.7-28.2.
//
//   The systolic path is taken directly here instead. Adapted from the u2 DPAS GEMM in
//   moe_3gemm_swiglu_mlp.cl, with the expert dimension and its block list removed, the fixed-width
//   per-group weight gather replaced by a per-k-step load so any group size works, and the zero
//   point taken from the jit constant rather than a buffer.
//
// WHY NOTHING IS DEQUANTISED
//   The weight enters the systolic array as the biased integer 1024 + 256*w, produced by OR-ing the
//   2-bit code into the mantissa of f16 1024 (0x6400, whose bits 8-9 are worth 256 each). That is
//   an affine function of the code, so the whole correction lands on the f32 accumulator:
//       out = sum_g (s_g / 256) * ( D_g - 256 * (4 + zp) * Sa_g )
//   where D_g is the raw DPAS accumulation over quant group g and Sa_g the plain sum of that
//   group's activations. Because zp only ever appears multiplied into a float, a fractional zero
//   point such as 1.5 costs exactly nothing here -- which is the property oneDNN's interface lacks.
//   The 1024 offset contributes even with no zero point, so Sa_g is computed unconditionally.
//
// JIT Parameters:
//   U2_DPAS_SG   - sub-group size, must be 16 (see below)
//   U2_MSUB      - token rows per work-group / 8
//   U2_N_SG      - sub-groups per work-group; each owns 16 output channels
//   WEIGHTS_K, WEIGHTS_N, BATCH_SIZE, DECOMPRESSION_SCALE_GROUP_SIZE

#define U2_MTILE     ((U2_MSUB) * 8)
#define U2_N_PER_WG  ((U2_N_SG) * 16)
#define U2_GROUP     (DECOMPRESSION_SCALE_GROUP_SIZE)
#define U2_NUM_G     ((WEIGHTS_K) / (U2_GROUP))
#define U2_XG_ITEMS  ((U2_MTILE) * (U2_NUM_G))

// The block2d activation load needs a pitch that is a multiple of 16 bytes and at least 64, so
// K % 32 != 0 is rejected on the host; a quant group must also not straddle a 16-deep DPAS step.
#if (WEIGHTS_K) % 32 != 0
#    error "fully_connected_gpu_u2_dpas.cl - WEIGHTS_K % 32 == 0 required"
#endif
#if (U2_GROUP) % 32 != 0
#    error "fully_connected_gpu_u2_dpas.cl - decompression group size % 32 == 0 required"
#endif
#if (WEIGHTS_N) % (U2_N_PER_WG) != 0
#    error "fully_connected_gpu_u2_dpas.cl - WEIGHTS_N % (U2_N_SG * 16) == 0 required"
#endif
#if (U2_DPAS_SG) != 16
// intel_sub_group_f16_f16_matrix_mad_k16 *compiles* at the xe2 default sub-group size of 32 and
// silently returns wrong results under every lane mapping. This kernel must never reference SIMD.
#    error "fully_connected_gpu_u2_dpas.cl - U2_DPAS_SG must be 16"
#endif

// A group smaller than 8*16 leaves lanes idle in the row-sum pass -- correct, just wasteful. The
// per-channel case this kernel is for has K itself as the group, so that does not arise.
#if (U2_GROUP) % 128 == 0
#    define U2_KCHUNK 128
#else
#    define U2_KCHUNK 32
#endif

__attribute__((intel_reqd_sub_group_size(U2_DPAS_SG))) KERNEL(fully_connected_gpu_u2_dpas)(
    OPTIONAL_SHAPE_INFO_ARG __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* scales,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    ,
    const __global BIAS_TYPE* bias
#endif
#if HAS_FUSED_OPS_DECLS
    ,
    FUSED_OPS_DECLS
#endif
) {
    const int lane = get_sub_group_local_id();
    const int sgid = (int)get_local_id(2);

    const int M = BATCH_SIZE;
    const int N = WEIGHTS_N;
    const int K = WEIGHTS_K;

    const int token_start = (int)get_group_id(2) * (U2_MTILE);
    const int n_tokens = min((int)(U2_MTILE), M - token_start);  // sub-group uniform
    // get_group_id(0) is the N tile and varies fastest, so the work-groups sharing an activation
    // tile stay co-resident and hit it in L2.
    const int n0 = (int)get_group_id(0) * (U2_N_PER_WG) + sgid * 16;

    // The only SLM. Per token, per quant group, the plain sum of activations, for the zero-point
    // correction. This is the FULL sum -- it is consumed directly by lane-private accumulators in
    // the main loop, with no reduction afterwards, so it must not be pre-divided by anything.
    //
    // One sub-group per (token, quant group), eight halfs per lane per instruction. The obvious
    // scalar version moves the same bytes but issues eight times the loads: a lane-strided scalar
    // read covers 16 halfs per sub-group instruction where the block2d load in the main loop covers
    // 128, and with one quant group per row this pass reads as much as the main loop does. That is
    // why it dominates here and did not in the grouped MoE kernel this came from.
    __local float xg[U2_XG_ITEMS];
    for (int item = sgid; item < (U2_XG_ITEMS); item += (U2_N_SG)) {
        const int m = item / (U2_NUM_G);
        const int g = item % (U2_NUM_G);
        float s = 0.0f;
        if (m < n_tokens) {
            const __global INPUT0_TYPE* r = input + (size_t)(token_start + m) * K + g * (U2_GROUP);
            for (int e = lane * 8; e < (U2_GROUP); e += (U2_DPAS_SG) * 8) {
                const half8 v = vload8(0, r + e);
                s += convert_float(v.s0) + convert_float(v.s1) + convert_float(v.s2) + convert_float(v.s3) +
                     convert_float(v.s4) + convert_float(v.s5) + convert_float(v.s6) + convert_float(v.s7);
            }
        }
        s = sub_group_reduce_add(s);
        if (lane == 0) {
            xg[item] = s;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // This lane's output channel, K innermost, so 4 bytes = 16 consecutive K.
    const __global uint* wrow = (const __global uint*)(weights + (size_t)(n0 + lane) * (K / 4));

#if DECOMPRESSION_ZP_TERM
    // float, never an integer type: (char)1.5 truncates to 1 and would corrupt every weight
    // without failing anything. The host rejects a non-scalar zero point, so this is uniform.
    const float zeff = 256.0f * (4.0f + (float)(DECOMPRESSION_ZP_VALUE));
#else
    const float zeff = 256.0f * 4.0f;
#endif

    float8 acc[U2_MSUB];
    unroll_for(int i = 0; i < (U2_MSUB); i++) {
        acc[i] = (float8)(0.0f);
    }

    for (int g = 0; g < (U2_NUM_G); g++) {
        // lane == channel, so this group's 16 scales are contiguous: one block read. Indexing is
        // S[group * N + n], which is what the verified GEMV path uses; the IR constant reads as
        // [N, G] and would imply n * G + group, so something reorders scales in between.
        const float sf = convert_float(
            as_half(intel_sub_group_block_read_us((const __global ushort*)(scales + (size_t)g * N + n0))));

        float8 g8[U2_MSUB];
        unroll_for(int i = 0; i < (U2_MSUB); i++) {
            if (i * 8 < n_tokens) {
                unroll_for(int m = 0; m < 8; m++) {
                    g8[i][m] = -zeff * xg[(i * 8 + m) * (U2_NUM_G) + g];
                }
            }
        }

        // U2_KCHUNK K values per weight load: 8 uints when the group allows it, 2 otherwise. The
        // MoE kernel this came from hoisted the whole quant group into registers, which is what
        // confined it to group sizes 32/64/128; per-channel scales need an arbitrary group. Loading
        // one k-step at a time lifts that limit but leaves 64 dependent 8-byte loads at K = 2048
        // with nothing to overlap them, so a chunk is loaded instead and then consumed in place.
        for (int kc = 0; kc < (U2_GROUP) / (U2_KCHUNK); kc++) {
#if (U2_KCHUNK) == 128
            const uint8 pk = vload8(kc, wrow + g * ((U2_GROUP) / 16));
#else
            const uint2 pk = vload2(kc, wrow + g * ((U2_GROUP) / 16));
#endif
            unroll_for(int kb = 0; kb < (U2_KCHUNK) / 32; kb++) {
                const uint p0 = pk[2 * kb];
                const uint p1 = pk[2 * kb + 1];

                int8 b0, b1;
                // Magic bias: 0x6400 is f16 1024, whose mantissa bits 8-9 are worth 256 each, so
                // OR-ing the 2-bit code into them yields exactly f16(1024 + 256*w) in 5 ops per
                // k-pair. b[j] packs k=k0+2j low and k=k0+2j+1 high (VNNI).
                unroll_for(int j = 0; j < 8; j++) {
                    const uint u = p0 >> (4 * j);
                    b0[j] = (int)(0x64006400u | ((u << 8) & 0x00000300u) | ((u << 22) & 0x03000000u));
                }
                unroll_for(int j = 0; j < 8; j++) {
                    const uint u = p1 >> (4 * j);
                    b1[j] = (int)(0x64006400u | ((u << 8) & 0x00000300u) | ((u << 22) & 0x03000000u));
                }

                const int k0 = g * (U2_GROUP) + kc * (U2_KCHUNK) + kb * 32;
                unroll_for(int i = 0; i < (U2_MSUB); i++) {
                    // n_tokens is sub-group uniform, so the mads stay in uniform control flow.
                    if (i * 8 >= n_tokens) {
                        continue;
                    }
                    // Rows past M read back as hardware zeros, which self-pads the ragged last
                    // token tile; those rows are dropped again at the store.
                    const ushort16 araw = intel_subgroup_block_read_u16_m8k16v2((__global void*)input,
                                                                                K * 2,
                                                                                M,
                                                                                K * 2,
                                                                                (int2)(k0, token_start + i * 8));
                    g8[i] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(araw.lo), b0, g8[i]);
                    g8[i] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(araw.hi), b1, g8[i]);
                }
            }
        }

        const float sc = sf * (1.0f / 256.0f);
        unroll_for(int i = 0; i < (U2_MSUB); i++) {
            if (i * 8 < n_tokens) {
                unroll_for(int m = 0; m < 8; m++) {
                    acc[i][m] = fma(g8[i][m], sc, acc[i][m]);
                }
            }
        }
    }

#if BIAS_TERM
    const float bias_v = convert_float(bias[n0 + lane]);
#endif

    // lane == channel, so a whole 16-channel row goes out in one block write.
    unroll_for(int i = 0; i < (U2_MSUB); i++) {
        unroll_for(int m = 0; m < 8; m++) {
            const int row = i * 8 + m;
            if (row < n_tokens) {
                ACTIVATION_TYPE v = TO_ACTIVATION_TYPE(acc[i][m]);
#if BIAS_TERM
                v += TO_ACTIVATION_TYPE(bias_v);
#endif
                // out_row / out_ch are what the host's index order refers to. A sub-group block
                // write puts lane L at channel n0 + L, so the two agree by construction.
                const int out_row = token_start + row;
                const int out_ch = n0 + lane;
#if HAS_FUSED_OPS
                FUSED_OPS
                const OUTPUT_TYPE res = FUSED_OPS_RESULT;
#else
                const OUTPUT_TYPE res = TO_OUTPUT_TYPE(ACTIVATION_TYPED(v, ACTIVATION_PARAMS_TYPED));
#endif
                intel_sub_group_block_write_us((__global ushort*)(output + (size_t)out_row * N + n0),
                                               as_ushort(res));
            }
        }
    }
}

#undef U2_MTILE
#undef U2_N_PER_WG
#undef U2_GROUP
#undef U2_NUM_G
#undef U2_XG_ITEMS
#undef U2_KCHUNK
