// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reinterleave/reinterleave.hpp"
#include "ttnn/run_operation.hpp"
#include "device/reinterleave_device_operation.hpp"

namespace ttnn::operations::experimental::reinterleave {

Tensor ReinterleaveFromBatch::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto t = ttnn::prim::reinterleave_from_batch(
        input, input_height, input_width, stride_hw, barrier_threshold, compute_kernel_config);
    return t;
}

OptionalTensors ReinterleaveLocal::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // TODO: Replace with actual implementation when available
    TT_THROW("ReinterleaveLocal::invoke is not yet implemented");
}

OptionalTensors ReinterleaveLocal::create_async_optional_output_tensors(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_THROW("ReinterleaveLocal::create_async_optional_output_tensors is not yet implemented");
}

}  // namespace ttnn::operations::experimental::reinterleave
