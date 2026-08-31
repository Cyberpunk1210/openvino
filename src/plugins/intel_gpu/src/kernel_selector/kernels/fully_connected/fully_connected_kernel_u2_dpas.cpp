// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_u2_dpas.h"

#include "common_types.h"
#include "fully_connected_kernel_bf_tiled.h"
#include "kernel_selector_utils.h"

#include <algorithm>

using namespace kernel_selector::fc_kernel_bf_tiled_utils;

namespace kernel_selector {

static constexpr size_t mtile = FullyConnected_U2_DPAS::msub * 8;
static constexpr size_t n_per_wg = FullyConnected_U2_DPAS::n_sg * 16;

bool FullyConnected_U2_DPAS::shape_ok(size_t k, size_t n, size_t group_size) {
    // The block2d activation load wants a pitch of at least 64 bytes that is a multiple of 16, and
    // a quant group must not straddle a 16-deep DPAS step.
    if (k % 32 != 0 || group_size % 32 != 0 || k % group_size != 0) {
        return false;
    }
    // One sub-group owns 16 output channels and a work-group owns n_sg of them; there is no
    // channel-remainder path.
    return n % n_per_wg == 0;
}

ParamsKey FullyConnected_U2_DPAS::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::UINT2);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();
    k.EnableWeightsCompression();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

DeviceFeaturesKey FullyConnected_U2_DPAS::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_blocked_read_write();
    k.requires_blocked_read_write_short();
    k.requires_subgroup_broadcast();

    return k;
}

bool FullyConnected_U2_DPAS::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    const auto& input = fc_params.inputs[0];
    const auto& output = fc_params.outputs[0];
    const auto& weights = fc_params.weights;

    // XMX and the 2D block loads. Without them the OpenCL JIT fails at model load rather than
    // falling back, so this must be decided here and not left to the compiler. xe2 is also the
    // only architecture the kernel this was adapted from has been measured on.
    if (!params.engineInfo.supports_immad || params.engineInfo.arch < gpu_arch::xe2)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (!fc_params.compressed || weights.GetDType() != WeightsType::UINT2)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // SwiGLU fusion splits the output in half and combines the halves, so it needs a kernel that
    // owns both; this one produces a plain [M, N]. Rejecting it costs the gate projection, which
    // stays on bf_tiled. Activations and eltwise fusions are applied at the store instead.
    if (is_swiglu_fused(fc_params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // A fused op is only taken on a static shape. Measured reason: the projections that carry one
    // (o and down here) are exactly the ones OpenVINO does not rebuild a static implementation for
    // at decode, so accepting them shape-agnostically leaves this kernel running at M = 1, where
    // its 32-row tile yields 8-24 work-groups and decode drops from 30.3 to 15.8 tok/s. Serving
    // both needs the runtime sub-kernel switch bf_tiled uses; until then the unfused projections
    // are the ones that get the systolic path.
    if (params.is_shape_agnostic && !fc_params.fused_ops.empty())
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (input.GetDType() != Datatype::F16 || output.GetDType() != Datatype::F16)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // A zero point may be absent or a single per-tensor value. A per-channel or per-group zero
    // point would have to be read per output channel inside the group loop; not implemented.
    if (fc_params.has_decompression_zp && !fc_params.scalar_zp)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // The kernel indexes weights as a plain [N, K/4] byte matrix with K innermost, and
    // GetTunedKernelsDataByIndex always asks for oiyx to get exactly that. os_iyx_osv16 is accepted
    // here only because it is what the node already carries once GEMV has claimed the decode path
    // for the same weights: primitive_inst keeps a reordered copy per layout, so this impl still
    // receives oiyx. Any other packing would decode to garbage rather than fail, so it is rejected.
    if (weights.GetLayout() != WeightsLayout::oiyx && weights.GetLayout() != WeightsLayout::os_iyx_osv16)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // A static shape must have enough rows to be worth the 32-row tile; a dynamic one is judged at
    // execution time instead, where the base's dispatch-update hook re-runs SetDefault against the
    // real row count.
    const auto out_bf = get_output_aligned_bf_size(fc_params, false);
    const size_t m = out_bf.first;
    const size_t n = out_bf.second;
    if (!params.is_shape_agnostic && m < mtile)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Same derivation the base jit uses for DECOMPRESSION_SCALE_GROUP_SIZE, so the two can never
    // disagree about what the kernel is compiled for.
    const size_t scale_groups = fc_params.decompression_scale.Feature().v;
    if (scale_groups == 0 || weights.IFM().v % scale_groups != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    if (!shape_ok(weights.IFM().v, n, weights.IFM().v / scale_groups))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (weights.OFM().v != n)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // The block2d load addresses the activation base directly, so an offset or padded input would
    // silently read the wrong rows.
    if (input.GetFirstElementOffset() != 0 || input.X().pad.Total() != 0 || input.Y().pad.Total() != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    if (output.GetFirstElementOffset() != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    return true;
}

FullyConnected_U2_DPAS::DispatchData FullyConnected_U2_DPAS::SetDefault(const fully_connected_params& params,
                                                                        int,
                                                                        int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params);

    const auto out_bf = get_output_aligned_bf_size(params, false);
    const size_t m = out_bf.first;
    const size_t n = params.weights.OFM().v;  // always static, unlike the output tensor
    const size_t m_tiles = CeilDiv(m, mtile);

    // dim0 selects the N tile and is the fastest varying, so work-groups sharing an activation tile
    // stay co-resident. dim1 is the sub-group. dim2 carries both the sub-group id within the
    // work-group and the token tile.
    // m_tiles is 0 while the shape is still dynamic; the base re-runs this hook at execution time
    // with the real row count, so a placeholder of 1 here is replaced before anything launches.
    dispatchData.gws = {n / n_per_wg, dpas_sg, std::max<size_t>(m_tiles, 1) * n_sg};
    dispatchData.lws = {1, dpas_sg, n_sg};

    return dispatchData;
}

KernelsPriority FullyConnected_U2_DPAS::GetKernelsPriority(const Params& /*params*/) const {
    // Eligible only for u2 weights whose shape fits the tile, where it beats bf_tiled by ~2.7x at
    // prefill batch. A dynamic shape is accepted too, so the real model can reach it at all; the
    // row count is then only known at execution time.
    return FORCE_PRIORITY_1;
}

JitConstants FullyConnected_U2_DPAS::GetJitConstants(const fully_connected_params& params,
                                                     const FullyConnectedKernelBase::DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    // DECOMPRESSION_SCALE_GROUP_SIZE, DECOMPRESSION_ZP_SCALAR and DECOMPRESSION_ZP_VALUE all come
    // from the base for a compressed FC; redefining them here would risk the two disagreeing.
    jit.AddConstant(MakeJitConstant("U2_DPAS_SG", dpas_sg));
    jit.AddConstant(MakeJitConstant("U2_MSUB", msub));
    jit.AddConstant(MakeJitConstant("U2_N_SG", n_sg));
    jit.AddConstant(MakeJitConstant("WEIGHTS_K", params.weights.IFM().v));
    jit.AddConstant(MakeJitConstant("WEIGHTS_N", params.weights.OFM().v));
    // As an expression, not a number: under a dynamic shape OUTPUT_BATCH_NUM resolves from the
    // shape-info argument, and a baked-in row count would silently be the compile-time one. A 3D
    // output carries the tokens across batch and feature, matching how bf_tiled reads it.
    const bool out_3d = params.outputs[0].GetLayout() == DataLayout::bfyx;
    jit.AddConstant(MakeJitConstant("BATCH_SIZE",
                                    out_3d ? "(OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM)" : "(OUTPUT_BATCH_NUM)"));

    // The base does not emit these for this kernel type, and the store needs them even with no
    // fused op at all.
    auto activation_dt = GetActivationType(params);
    if (activation_dt == Datatype::F16) {
        activation_dt = Datatype::F32;
    }
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));

    if (!params.fused_ops.empty()) {
        // A 3D output carries the token index in the feature dimension and the channel in y; a 2D
        // one carries them in batch and feature. Getting this backwards reads the fused operand at
        // the wrong offset, which is silently wrong rather than a build error.
        std::vector<std::string> idx_order =
            out_3d ? std::vector<std::string>{"0", "out_row", "out_ch", "0"}
                   : std::vector<std::string>{"out_row", "out_ch", "0", "0"};
        FusedOpsConfiguration conf = {"", idx_order, "v", activation_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData FullyConnected_U2_DPAS::GetTunedKernelsDataByIndex(const Params& params, const int autoTuneIndex) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);

    return GetCommonKernelsData(params,
                                fc_params.inputs[0].GetLayout(),
                                WeightsLayout::oiyx,
                                EXE_MODE_DEFAULT,
                                autoTuneIndex,
                                0);
}

KernelsData FullyConnected_U2_DPAS::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    KernelsData kds = GetTunedKernelsDataByIndex(params, -1);
    if (!kds.empty()) {
        res.emplace_back(kds[0]);
    }

    return res;
}

}  // namespace kernel_selector
