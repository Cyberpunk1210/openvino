// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

// Prefill-only u2 fully connected on the XMX systolic array. See the header comment in
// fully_connected_gpu_u2_dpas.cl for why oneDNN cannot serve this case at all.
class FullyConnected_U2_DPAS : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;

    FullyConnected_U2_DPAS() : Parent("fully_connected_gpu_u2_dpas") {}

    using FullyConnectedKernelBase::GetTunedKernelsDataByIndex;
    KernelsData GetTunedKernelsDataByIndex(const Params& params, const int autoTuneIndex = -1) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

    // Token rows per work-group and output channels per work-group. Both are shape constraints the
    // caller has to test before this kernel can be selected, so they are public.
    static constexpr size_t msub = 4;   // MTILE = msub * 8 = 32 rows.
    static constexpr size_t n_sg = 16;  // N_PER_WG = n_sg * 16 = 256 channels.
    static constexpr size_t dpas_sg = 16;
    static bool shape_ok(size_t k, size_t n, size_t group_size);

protected:
    DispatchData SetDefault(const fully_connected_params& params,
                            int autoTuneIndex = -1,
                            int kernel_number = 0) const override;
    // SWIGLU is deliberately absent: it combines both halves of the output, which this kernel does
    // not own -- Validate rejects it so the gate projection stays on bf_tiled.
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::ACTIVATION, FusedOpType::ELTWISE};
    }
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
};
}  // namespace kernel_selector
