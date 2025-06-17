// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reinterleave_device_operation.hpp"

// Program factory for ReinterleaveFromBatch.
namespace ttnn::operations::experimental::reinterleave {
ReinterleaveFromBatchOperation::ProgramFactoryFromBatch::cached_program_t
ReinterleaveFromBatchOperation::ProgramFactoryFromBatch::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Reinterleave mostly follows the same patern as Deinterleave
    // core allocation logic remains the same, but the source and destination swap places
    // Deinterleave runs on the destination cores and uses the NoC to read from the source cores
    // For Reinterleave, those same destination cores are now the source cores
    // and we will use the NoC to write to the former source/new destination cores
    // Addresses calculated should be the same in both cases

    std::map<int, int> batch_mapping = {
        {0, 0},   {1, 1},   {2, 8},   {3, 9},   {4, 2},   {5, 3},   {6, 10},  {7, 11},  {8, 16},  {9, 17},  {10, 24},
        {11, 25}, {12, 18}, {13, 19}, {14, 26}, {15, 27}, {16, 4},  {17, 5},  {18, 12}, {19, 13}, {20, 6},  {21, 7},
        {22, 14}, {23, 15}, {24, 20}, {25, 21}, {26, 28}, {27, 29}, {28, 22}, {29, 23}, {30, 30}, {31, 31}, {32, 32},
        {33, 33}, {34, 40}, {35, 41}, {36, 34}, {37, 35}, {38, 42}, {39, 43}, {40, 48}, {41, 49}, {42, 56}, {43, 57},
        {44, 50}, {45, 51}, {46, 58}, {47, 59}, {48, 36}, {49, 37}, {50, 44}, {51, 45}, {52, 38}, {53, 39}, {54, 46},
        {55, 47}, {56, 52}, {57, 53}, {58, 60}, {59, 61}, {60, 54}, {61, 55}, {62, 62}, {63, 63}};

    using namespace tt::constants;
    using namespace tt::tt_metal::detail;
    using namespace tt::tt_metal;
    using namespace tt;

    Program program;

    const auto& input = tensor_args.input;

    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tensor.get_padded_shape()[-1] * tensor.element_size();
    };

    uint32_t num_units = output.volume() / output.get_logical_shape()[-1];

    tt::tt_metal::CoreRangeSet worker_grid = input.memory_config().shard_spec().value().grid;
    auto num_units_per_core = input.memory_config().shard_spec().value().shape[0];

    // dst buffer here is same as deinterleave src buffer
    // vice versa for src buffer here
    uint32_t dst_cb_id = CBIndex::c_0;
    auto output_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    uint32_t aligned_output_unit_size = round_up_to_mul16(output_unit_size);
    uint32_t dst_total_size = output.get_logical_shape()[0] * aligned_output_unit_size;

    uint32_t src_cb_id = CBIndex::c_1;
    auto input_data_format = datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t aligned_input_unit_size = round_up_to_mul16(input_unit_size);
    uint32_t src_total_size = input.get_logical_shape()[0] * aligned_input_unit_size;

    tt::tt_metal::CircularBufferConfig dst_cb_config =
        tt::tt_metal::CircularBufferConfig(dst_total_size, {{dst_cb_id, output_data_format}})
            .set_page_size(dst_cb_id, aligned_output_unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto dst_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, dst_cb_config);

    tt::tt_metal::CircularBufferConfig src_cb_config =
        tt::tt_metal::CircularBufferConfig(src_total_size, {{src_cb_id, input_data_format}})
            .set_page_size(src_cb_id, aligned_input_unit_size)
            .set_globally_allocated_address(*input.buffer());
    auto src_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, src_cb_config);

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;

    TT_FATAL(input_unit_size == output_unit_size, "Deinterleave: input and output unit size must be equal");

    // after deinterleave, tensor is narrower by width stride and taller by height stride
    auto output_per_core_width = operation_attributes.input_width * operation_attributes.stride_hw[1];
    auto output_per_core_height = input.memory_config().shard_spec().value().shape[0] /
                                  operation_attributes.input_width / operation_attributes.stride_hw[0];
    log_debug(
        tt::LogOp,
        "DeinterleaveToBatchOperation::ProgramFactoryToBatch::create; stride_hw: {}; per core height {} per_core_width "
        "{}",
        operation_attributes.stride_hw,
        output_per_core_height,
        output_per_core_width);
    auto stick_size_bytes = aligned_output_unit_size;
    reader_compile_time_args = {
        (uint32_t)src_cb_id,
        (uint32_t)dst_cb_id,
        (uint32_t)output_per_core_width,
        (uint32_t)output_per_core_height,
        (uint32_t)stick_size_bytes,
        (uint32_t)operation_attributes.stride_hw[0],
        (uint32_t)operation_attributes.stride_hw[1],
        (uint32_t)operation_attributes.barrier_threshold,
    };

    writer_compile_time_args = {
        (uint32_t)src_cb_id,
        (uint32_t)dst_cb_id,
        (uint32_t)output_per_core_width,
        (uint32_t)output_per_core_height,
        (uint32_t)stick_size_bytes,
        (uint32_t)operation_attributes.stride_hw[0],
        (uint32_t)operation_attributes.stride_hw[1],
        (uint32_t)operation_attributes.barrier_threshold,
    };

    auto read_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reinterleave/device/kernels/reinterleave_kernel_rm.cpp",
        worker_grid,
        ReaderDataMovementConfig(reader_compile_time_args, {}));

    auto write_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reinterleave/device/kernels/reinterleave_kernel_rm.cpp",
        worker_grid,
        WriterDataMovementConfig(writer_compile_time_args, {}));

    TT_FATAL(worker_grid.size() == 1, "Deinterleave: shard spec CoreRangeSet must have single range");
    tt::tt_metal::CoreCoord device_grid =
        worker_grid.bounding_box().grid_size();  // input.device()->logical_grid_size();

    uint32_t num_of_shards = worker_grid.num_cores();
    auto cores = corerange_to_cores(worker_grid, std::nullopt, true);

    uint32_t in_batches = operation_attributes.stride_hw[0] * operation_attributes.stride_hw[1];

    // a single core cannot contain data belonging to multiple batches, same constraint as on deinterleave
    TT_FATAL(in_batches <= num_of_shards, "Deinterleave: out_batches {} > num_of_shards {}", in_batches, num_of_shards);

    log_debug(tt::LogOp, "Output buffer address {:#x}", output.buffer()->address());
    log_debug(tt::LogOp, "Input buffer address {:#x}", input.buffer()->address());

    using CoreCoord = tt::tt_metal::CoreCoord;

    // apply offset to the start location
    auto core_coord_offset = [](const CoreCoord start_loc, const CoreCoord grid, int offset) -> CoreCoord {
        auto x = start_loc.x + offset;
        auto y = start_loc.y;
        if (x >= grid.x) {
            x = x % grid.x;
            y += (start_loc.x + offset) / grid.x;
        }
        return {x, y};
    };

    // calculate the end location of the core
    auto core_coord_get_end_loc = [](const CoreCoord start_loc, const CoreCoord grid, int core_count) -> CoreCoord {
        auto x = start_loc.x + core_count;
        auto y = start_loc.y + 1;
        if (x >= grid.x) {
            x = grid.x;
            y = start_loc.y + (start_loc.x + core_count) / grid.x;
        }
        return {x, y};
    };

    // calculate the offset location of the core in start_loc-end_loc range.
    auto core_range_offset =
        [](const CoreCoord start_loc, const CoreCoord end_loc, const CoreCoord grid, int offset) -> CoreCoord {
        auto x = start_loc.x + offset;
        auto y = start_loc.y;
        if (x >= end_loc.x) {
            x = x % grid.x;
            y += (start_loc.x + offset) / grid.x;
        }
        return {x, y};
    };

    for (const auto& core : cores) {
        auto this_core = core.x + core.y * device_grid.x;

        // number of input batches,for ABAB;CDCD => 4 batches AAAA;BBBB;CCCD;DDDD
        // also turns out this is the number of src core one writes to
        uint32_t num_src_cores = in_batches;
        // number of cores containing one input batch
        uint32_t cores_in_batch = num_of_shards / in_batches;
        // batch this core is processing [0-3]
        uint32_t src_batch = this_core / cores_in_batch;
        // chainted deinterleave messes up order, use mapping
        src_batch = batch_mapping[src_batch];

        // id of this core in batch
        uint32_t id_in_batch = this_core % cores_in_batch;
        uint32_t start_id = id_in_batch * num_src_cores;
        CoreCoord start = {start_id % device_grid.x, start_id / device_grid.x};
        CoreCoord end = core_coord_get_end_loc(start, device_grid, num_src_cores);

        // core should proccess data from start_xy to end_xy, but we dont want every core writing to the same dest
        // to start from the same point but offset and process the data in a round robin fashion. cores that write same
        // dests have same id_in_batch, but different dst_batch
        CoreCoord offset_dm0 = core_range_offset(start, end, device_grid, src_batch);
        CoreCoord offset_dm1 = core_range_offset(start, end, device_grid, (src_batch + 1) % in_batches);

        // dest offset are not affected by offset change, because we always write all data to one dest and ordering
        // here is not important.
        uint32_t dst_width_stride = operation_attributes.stride_hw[1] * stick_size_bytes;
        uint32_t dst_height_offset_to_next =
            (operation_attributes.stride_hw[0] - 1) * output_per_core_width * stick_size_bytes;

        uint32_t dst_datum_width_offset = (src_batch % operation_attributes.stride_hw[0]) * stick_size_bytes;
        uint32_t dst_datum_height_offset =
            (src_batch / operation_attributes.stride_hw[1]) * output_per_core_width * stick_size_bytes;

        uint32_t dst_offset = dst_datum_width_offset + dst_datum_height_offset;

        // stride to move for one dst_core output in the src buffer
        uint32_t src_height = output_per_core_height / operation_attributes.stride_hw[0];
        uint32_t src_width = output_per_core_width / operation_attributes.stride_hw[1];

        uint32_t src_b1_size_bytes = src_height * src_width * stick_size_bytes;
        uint32_t src_offset_dm0 = src_batch * src_b1_size_bytes;
        uint32_t src_offset_dm1 = (src_offset_dm0 + src_b1_size_bytes) % (in_batches * src_b1_size_bytes);

        uint32_t dst_rollover_offset_dm0 =
            (src_batch % 2 == 0) ? 0 : src_b1_size_bytes;  // div by 2 for two data movement processors
        uint32_t dst_rollover_offset_dm1 =
            (src_batch % 2 == 0) ? src_b1_size_bytes : 0;  // div by 2 for two data movement processors

        log_debug(
            tt::LogOp,
            "DeinterleaveToBatchOperation::ProgramFactoryToBatch::create; core: {} myid {}, start {}-{}, end {}-{}, "
            "dst_batch "
            "{}, "
            "id_in_batch {} offset_dm0 {}-{} offset_dm1 {}-{}",
            core,
            this_core,
            start.y,
            start.x,
            end.y,
            end.x,
            src_batch,
            id_in_batch,
            offset_dm0.y,
            offset_dm0.x,
            offset_dm1.y,
            offset_dm1.x);
        TT_FATAL(end.x > 0, "Deinterleave: end.x {} == 0 | ", end.x);
        TT_FATAL(end.y > 0, "Deinterleave: end.y {} == 0 | start={} num_src_cores={}", end.y, start, num_src_cores);

        TT_FATAL(
            end.x <= device_grid.x,
            "Deinterleave: unsupported configuration. {} end.x {} cannot be larger than device_grid.x {}",
            core,
            end.x,
            device_grid.x);

        TT_FATAL(
            end.y <= device_grid.y,
            "Deinterleave: unsupported configuration. {} end.y {} cannot be larger than device_grid.y {}",
            core,
            end.y,
            device_grid.y);

        log_debug(
            tt::LogOp,
            "src_width_stride {}, src_height_offset_to_next {}",
            dst_width_stride,
            dst_height_offset_to_next);
        log_debug(
            tt::LogOp,
            "dst_batch {}, src_offset_dm0 {}, src_offset_dm1 {}, dst_b1_size_bytes {}, dst_offset_dm0 {}, "
            "dst_offset_dm1 {}",
            src_batch,
            dst_offset,
            dst_offset,
            src_b1_size_bytes,
            src_offset_dm0,
            src_offset_dm1);
        SetRuntimeArgs(
            program,
            read_kernel_id,
            core,
            {(uint32_t)start.x,
             (uint32_t)end.x,
             (uint32_t)start.y,
             (uint32_t)end.y,
             (uint32_t)dst_width_stride,
             (uint32_t)dst_height_offset_to_next,
             (uint32_t)dst_offset,
             (uint32_t)src_b1_size_bytes,
             (uint32_t)src_offset_dm0,
             (uint32_t)offset_dm0.x,
             (uint32_t)offset_dm0.y,
             (uint32_t)num_src_cores,
             (uint32_t)dst_rollover_offset_dm0});
        SetRuntimeArgs(
            program,
            write_kernel_id,
            core,
            {(uint32_t)start.x,
             (uint32_t)end.x,
             (uint32_t)start.y,
             (uint32_t)end.y,
             (uint32_t)dst_width_stride,
             (uint32_t)dst_height_offset_to_next,
             (uint32_t)dst_offset,
             (uint32_t)src_b1_size_bytes,
             (uint32_t)src_offset_dm1,
             (uint32_t)offset_dm1.x,
             (uint32_t)offset_dm1.y,
             (uint32_t)num_src_cores,
             (uint32_t)dst_rollover_offset_dm1});
    }

    return {std::move(program), {src_cb, dst_cb}};
}

void ReinterleaveFromBatchOperation::ProgramFactoryFromBatch::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& src_cb_id = cached_program.shared_variables.src_cb_id;
    const auto& dst_cb_id = cached_program.shared_variables.dst_cb_id;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, src_cb_id, *src_buffer);
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, dst_cb_id, *dst_buffer);
}
}  // namespace ttnn::operations::experimental::reinterleave
