/*******************************************************************************
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Reference BatchGatherMatmul for u2 (2-bit unsigned, 4 values per byte, LSB-first)
// compressed weights. gemmstone micro-kernels and oneDNN have no u2 support, so u2
// weights take this correctness-first path: one work-item per output element.
// Dispatch: x = output channel n, y = flat (token_idx, expert_slot).
// Scales/zp are physically [E, G, N] (bfyx when G == 1, byfx after prepare_quantization
// otherwise), same as the micro-kernel path.

KERNEL(bgm_u2_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const global half* input_ptr,
    const global uchar* weight_ptr,
    global half* out_ptr,
    const global int* indices,
    int m,
    int k
#ifdef BIAS_DT
    , const global BIAS_DT* bias_ptr
#endif
    , const global WEIGHT_SCALE_DT* weight_scales
#ifdef WEIGHT_ZP_DT
    , const global WEIGHT_ZP_DT* weight_zps
#endif
) {
    uint n = get_global_id(0);
    if (n >= (uint)m)
        return;

    uint flat_idx = get_global_id(1);
    uint top_k = TOP_K;
    uint token_idx = flat_idx / top_k;
    uint expert_slot = flat_idx % top_k;
    int n_tokens = N_TOKENS;
    int expert_id = indices[token_idx * top_k + expert_slot];

    uint n_act = N_ACTIVATED_EXPERTS;
    uint a_slot = min(expert_slot, n_act - 1);
    const global half* x = input_ptr + (a_slot * n_tokens + token_idx) * INPUT_STRIDE;

    // u2: 4 values per byte; EXPERT_STRIDE is the per-expert byte count (elements / 4).
    const global uchar* w = weight_ptr + (long)expert_id * EXPERT_STRIDE + n * (k / 4);

    const global WEIGHT_SCALE_DT* s = weight_scales + (long)expert_id * NUM_GROUPS * M_GEMM;
#ifdef WEIGHT_ZP_DT
#    ifdef WEIGHT_ZP_SCALAR
    // Scalar (per-tensor) zp: one element shared by all experts/groups/channels.
    const global WEIGHT_ZP_DT* z = weight_zps;
#    elif defined(WEIGHT_COMPRESSED_ZP_INT2)
    const global uchar* z = (const global uchar*)weight_zps + ((long)expert_id * NUM_GROUPS * M_GEMM) / 4;
#    elif defined(WEIGHT_COMPRESSED_ZP_INT4)
    const global uchar* z = (const global uchar*)weight_zps + ((long)expert_id * NUM_GROUPS * M_GEMM) / 2;
#    else
    const global WEIGHT_ZP_DT* z = weight_zps + (long)expert_id * NUM_GROUPS * M_GEMM;
#    endif
#endif

    const uint group_size = k / NUM_GROUPS;
    float acc = 0.0f;
    for (uint g = 0; g < NUM_GROUPS; ++g) {
        float scale = convert_float(s[g * M_GEMM + n]);
        float zp = 0.0f;
#ifdef WEIGHT_ZP_DT
#    ifdef WEIGHT_ZP_SCALAR
        zp = convert_float(z[0]);
#    else
        uint zp_flat = g * M_GEMM + n;
#        ifdef WEIGHT_COMPRESSED_ZP_INT2
        zp = convert_float((z[zp_flat / 4] >> ((zp_flat % 4) * 2)) & 0x3);
#        elif defined(WEIGHT_COMPRESSED_ZP_INT4)
        zp = convert_float((z[zp_flat / 2] >> ((zp_flat % 2) * 4)) & 0xF);
#        else
        zp = convert_float(z[zp_flat]);
#        endif
#    endif
#endif
        float partial = 0.0f;
        const uint k_begin = g * group_size;
        const uint k_end = k_begin + group_size;
        for (uint kk = k_begin; kk < k_end; ++kk) {
            float w_val = convert_float((w[kk / 4] >> ((kk % 4) * 2)) & 0x3);
            partial += convert_float(x[kk]) * (w_val - zp);
        }
        acc += partial * scale;
    }
#ifdef BIAS_DT
    acc += convert_float(bias_ptr[(long)expert_id * BIAS_STRIDE + n]);
#endif
    out_ptr[(expert_slot * n_tokens + token_idx) * OUTPUT_STRIDE + n] = convert_half(acc);
}
