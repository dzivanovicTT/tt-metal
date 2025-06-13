#include "deinterleave_device_operation.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::experimental::deinterleave {
DeinterleaveLocalOperation::ProgramFactoryLocal::cached_program_t
DeinterleaveLocalOperation::ProgramFactoryLocal::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // TODO: implement
    Program program;

    // temporary
    tt::tt_metal::CBHandle src_cb_id = 0;
    tt::tt_metal::CBHandle dst_cb_id = 0;

    return {std::move(program), {src_cb_id, dst_cb_id}};
}

void DeinterleaveLocalOperation::ProgramFactoryLocal::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& src_cb_id = cached_program.shared_variables.src_cb_id;
    const auto& dst_cb_id = cached_program.shared_variables.dst_cb_id;
}
}  // namespace ttnn::operations::experimental::deinterleave
