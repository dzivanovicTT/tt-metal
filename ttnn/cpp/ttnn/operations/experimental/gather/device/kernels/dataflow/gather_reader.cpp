// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "gather_common.hpp"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t coordinator_core_physical_coord_x = get_arg_val<uint32_t>(0);
    const uint32_t coordinator_core_physical_coord_y = get_arg_val<uint32_t>(1);
    const uint32_t coordinator_to_cores_semaphore_id = get_semaphore(get_arg_val<uint32_t>(2));
    const uint32_t cores_to_coordinator_semaphore_id = get_semaphore(get_arg_val<uint32_t>(3));
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(4);
    const uint32_t index_loop_count = get_arg_val<uint32_t>(5);
    const bool additional_index_loop = get_arg_val<uint32_t>(6) == 1;

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(5);
    constexpr uint32_t tile_width = get_compile_time_arg_val(6);
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(8);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(9);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(10);
    constexpr uint32_t index_tiles_per_core = get_compile_time_arg_val(11);

    constexpr uint32_t one_tile = 1;

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_cb_index);
    const InterleavedAddrGenFast<index_tensor_is_dram> index_tensor_addr_gen = {
        .bank_base_address = index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

    // Semaphore setup
    volatile tt_l1_ptr uint32_t* semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(coordinator_to_cores_semaphore_id);
    noc_semaphore_set(semaphore_ptr, VALID);  // Reset the semaphore (Valid - we wait for 0)
    const uint64_t coordinator_core_addr = get_noc_addr(
        coordinator_core_physical_coord_x, coordinator_core_physical_coord_y, cores_to_coordinator_semaphore_id);

    // Copy input data to output and generate index tiles
    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        for (uint32_t index_loop = 0; index_loop < index_loop_count; index_loop++) {
            // Read index tiles
            uint32_t currently_read_index_tile =
                core_start * index_tiles_per_core + index_loop * number_of_available_cores * index_tiles_per_core;

            uint32_t current_iteration_tile_count = 0;

            for (uint32_t widx = 0; widx < index_tiles_per_core; widx++) {
                cb_reserve_back(index_tensor_cb_index, one_tile);
                const uint32_t l1_index_write_addr = get_write_ptr(index_tensor_cb_index);
                noc_async_read_tile(
                    h * Wt_index + currently_read_index_tile, index_tensor_addr_gen, l1_index_write_addr);
                noc_async_read_barrier();
                current_iteration_tile_count++;

                cb_push_back(index_tensor_cb_index, one_tile);

                currently_read_index_tile++;
                if (currently_read_index_tile >= Wt_index) {
                    break;  // No more index tiles to read
                }
            }  // widx loop

            // Get output tile
            cb_reserve_back(
                output_tensor_cb_index, current_iteration_tile_count);  // index_tiles_per_core = output_tiles_per_core

            cb_wait_front(index_tensor_cb_index, current_iteration_tile_count);

            for (uint32_t wi = 0; wi < Wt_input; wi++) {
                // Reserve space for input tile
                cb_reserve_back(input_tensor_cb_index, one_tile);

                // Indicate to the coordinator that the core is ready
                noc_semaphore_inc(coordinator_core_addr, 1);
                noc_semaphore_wait(semaphore_ptr, INVALID);  // Wait for coordinator to signal to start
                noc_semaphore_set(semaphore_ptr, VALID);  // Reset the semaphore

                // Process tile
                for (uint32_t index_tile_offset = 0; index_tile_offset < current_iteration_tile_count;
                     index_tile_offset++) {
                    process_input_tile(
                        input_tensor_cb_index,
                        index_tensor_cb_index,
                        output_tensor_cb_index,
                        wi,
                        tile_width,
                        index_tile_offset);
                }

                // Reset input buffer
                cb_push_back(input_tensor_cb_index, one_tile);   // Push tile to the writer
                cb_wait_front(input_tensor_cb_index, one_tile);  // Wait for the writer to finish
                cb_pop_front(input_tensor_cb_index, one_tile);   // Remove data from local buffer
            }  // wi loop

            cb_push_back(output_tensor_cb_index, current_iteration_tile_count);  // Push tile to the writer

            cb_pop_front(index_tensor_cb_index, current_iteration_tile_count);  // Remove data from local buffer
        }  // index_loop loop

        if (additional_index_loop) {
            for (uint32_t wi = 0; wi < Wt_input; wi++) {
                // Reserve space for input tile
                cb_reserve_back(input_tensor_cb_index, one_tile);

                // Indicate to the coordinator that the core is ready
                noc_semaphore_inc(coordinator_core_addr, 1);
                noc_semaphore_wait(semaphore_ptr, INVALID);  // Wait for coordinator to signal to start
                noc_semaphore_set(semaphore_ptr, VALID);  // Reset the semaphore

                // Reset input buffer
                cb_push_back(input_tensor_cb_index, one_tile);   // Push tile to the writer
                cb_wait_front(input_tensor_cb_index, one_tile);  // Wait for the writer to finish
                cb_pop_front(input_tensor_cb_index, one_tile);   // Remove data from local buffer
            }  // wi loop
        }  // if additional_index_loop
    }  // h loop
}
