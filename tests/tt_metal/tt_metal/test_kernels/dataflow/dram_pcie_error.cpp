// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t bank_base_address = get_arg_val<uint32_t>(0);
    uint32_t page_size = get_arg_val<uint32_t>(1);
    uint32_t dst_l1_addr = get_arg_val<uint32_t>(2);
    uint32_t num_pages_to_read = get_arg_val<uint32_t>(3);
    uint32_t num_iterations = 100000;  // get_arg_val<uint32_t>(4);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = bank_base_address, .page_size = page_size, .data_format = DataFormat::Float16_b};

    // uint32_t id = 0;
    // uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<true>(id);
    // uint32_t bank_index = interleaved_addr_gen::get_bank_index<true>(id, bank_offset_index);
    // uint32_t src_addr = s.get_addr(id, bank_offset_index, bank_index, 0);
    // uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank_index, noc_index);

    // uint64_t full_addr = get_noc_addr(src_noc_xy, src_addr, noc_index);

    for (uint32_t iter_idx = 0; iter_idx < num_iterations; iter_idx++) {
        for (uint32_t page_id = 0; page_id < num_pages_to_read; page_id++) {
            // noc_async_read_with_state(src_addr, dst_l1_addr, 8192, noc_index);
            noc_async_read_tile(page_id, s, dst_l1_addr);
        }
    }

    noc_async_read_barrier();
}
