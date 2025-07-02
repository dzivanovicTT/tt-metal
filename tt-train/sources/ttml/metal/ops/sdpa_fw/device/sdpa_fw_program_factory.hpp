// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "sdpa_fw_device_operation_types.hpp"

namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle sdpa_fw_reader_kernel_id;
        tt::tt_metal::KernelHandle sdpa_fw_writer_kernel_id;
        // TODO[improve]: think about how to split work around kernels more efficiently
        tt::tt_metal::KernelHandle sdpa_fw_kernel_group_1_id;
        tt::tt_metal::KernelHandle sdpa_fw_kernel_group_2_id;
        CoreRangeSet core_group_1;
        CoreRangeSet core_group_2;
        uint32_t num_cores{};
        uint32_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::sdpa_fw::device
