// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstdint>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::reinterleave {

struct ReinterleaveFromBatchOperation {
    struct operation_attributes_t {
        const uint32_t input_height;
        const uint32_t input_width;
        const std::array<uint32_t, 2> stride_hw;
        const uint32_t barrier_threshold;
        const bool split_work = false;  // Whether to split work between DM0 and DM1
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const ttnn::Tensor& input;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    struct ProgramFactoryFromBatch {
        struct shared_variables_t {
            tt::tt_metal::CBHandle src_cb_id;
            tt::tt_metal::CBHandle dst_cb_id;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<ProgramFactoryFromBatch>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

// (ReinterleaveLocalOperation would be implemented similarly)

}  // namespace ttnn::operations::experimental::reinterleave

namespace ttnn::prim {
constexpr auto reinterleave_from_batch = ttnn::register_operation<
    "ttnn::prim::reinterleave_from_batch",
    ttnn::operations::experimental::reinterleave::ReinterleaveFromBatchOperation>();

// TODO: Register local operation when implemented
// constexpr auto reinterleave_local = ttnn::register_operation<
//     "ttnn::prim::reinterleave_local",
//     ttnn::operations::experimental::reinterleave::ReinterleaveLocalOperation>();

}  // namespace ttnn::prim
