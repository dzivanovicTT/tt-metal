// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::reinterleave {
struct ReinterleaveFromBatch {
    static Tensor invoke(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

struct ReinterleaveLocal {
    static OptionalTensors invoke(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);

    static OptionalTensors create_async_optional_output_tensors(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::experimental::reinterleave

namespace ttnn {
namespace experimental {
constexpr auto reinterleave_from_batch = ttnn::register_operation<
    "ttnn::experimental::reinterleave_from_batch",
    ttnn::operations::experimental::reinterleave::ReinterleaveFromBatch>();

constexpr auto reinterleave_local = ttnn::register_operation<
    "ttnn::experimental::reinterleave_local",
    ttnn::operations::experimental::reinterleave::ReinterleaveLocal>();

}  // namespace experimental
}  // namespace ttnn
