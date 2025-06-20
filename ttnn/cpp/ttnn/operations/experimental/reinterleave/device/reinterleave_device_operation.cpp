// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reinterleave_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

// Device operation implementation for reinterleave.
namespace ttnn::operations::experimental::reinterleave {

void ReinterleaveFromBatchOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Reinterleave: input must be on device");
    TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Reinterleave: input must be BFLOAT16");
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Reinterleave: input must be ROW_MAJOR");
    TT_FATAL(input.buffer() != nullptr, "Reinterleave: input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Reinterleave: input must be HEIGHT_SHARDED");
    TT_FATAL(input.memory_config().shard_spec().has_value(), "Reinterleave: input must have shard_spec");
    TT_FATAL(
        input.memory_config().shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Reinterleave: input must have ROW_MAJOR orientation");
    auto per_core_height = input.memory_config().shard_spec().value().shape[0] / operation_attributes.input_width;
    TT_FATAL(
        per_core_height >= operation_attributes.stride_hw[0],
        "Reinterleave: per_core_height {} must be larger than {}",
        per_core_height,
        operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height % (operation_attributes.stride_hw[0]) == 0,
        "Reinterleave: per_core_height {} must be divisible by {}",
        per_core_height,
        operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height * operation_attributes.input_width == input.memory_config().shard_spec().value().shape[0],
        "Reinterleave: per_core_height {} * input_width {} must be equal to input shard_spec shape {}",
        per_core_height,
        operation_attributes.input_width,
        input.memory_config().shard_spec().value().shape[0]);
}

ReinterleaveFromBatchOperation::program_factory_t ReinterleaveFromBatchOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactoryFromBatch{};
}

void ReinterleaveFromBatchOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void ReinterleaveFromBatchOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

ReinterleaveFromBatchOperation::spec_return_value_t ReinterleaveFromBatchOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    log_debug(
        tt::LogOp,
        "ReinterleaveFromBatch::compute_output_specs logical_shape: {}: padded_shape: {}",
        input.get_logical_shape(),
        input.get_padded_shape());

    // Untilize returns bad shard spec padded to tile width
    // We expect shard witdh to match the actual number of channels in the tensor
    // auto output_memory_config = input.memory_config();
    // output_memory_config.shard_spec->shape[1] = input.get_logical_shape()[3];

    // TensorLayout::fromPaddedShape(
    //     input_tensor.get_dtype(),
    //     PageConfig(Layout::TILE),
    //     mem_config,
    //     input_tensor.get_logical_shape(),
    //     input_tensor.get_padded_shape()))

    // return {TensorSpec(
    //     input_tensor.get_logical_shape(),
    //     TensorLayout::fromPaddedShape(
    //         input_tensor.get_dtype(),
    //         PageConfig(Layout::TILE),
    //         mem_config,
    //         input_tensor.get_logical_shape(),
    //         input_tensor.get_padded_shape()))};

    auto tensor_spec = TensorSpec(
        input.get_logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input.get_dtype(),
            tt::tt_metal::PageConfig(input.get_layout()),
            input.memory_config(),
            input.get_logical_shape(),
            input.get_padded_shape()));
    // tt::tt_metal::TensorLayout(
    //     input.get_dtype(), tt::tt_metal::PageConfig(input.get_layout()), input.memory_config()));
    return tensor_spec;
};

ReinterleaveFromBatchOperation::tensor_return_value_t ReinterleaveFromBatchOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    log_debug(tt::LogOp, "ReinterleaveFromBatch::create_output_tensors");
    return create_device_tensor(spec, tensor_args.input.device());
}

std::tuple<ReinterleaveFromBatchOperation::operation_attributes_t, ReinterleaveFromBatchOperation::tensor_args_t>
ReinterleaveFromBatchOperation::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            input_height,
            input_width,
            stride_hw,
            barrier_threshold,
            false,
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{input},
    };
}

}  // namespace ttnn::operations::experimental::reinterleave
