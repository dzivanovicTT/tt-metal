//
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    cb_push_back(cb_id_in0, num_tiles_per_core);

    constexpr uint32_t cb_id_untilize_out = 16;
    constexpr uint32_t cb_id_out = 17;
    ;

    // cb_reserve_back(cb_id_out, num_unpadded_output_rows);
    uint32_t l1_write_addr = get_write_ptr(cb_id_out);

    // uint32_t act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
    // for (uint32_t inner = 0; inner < weight_size_w; inner++) {
    //     noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
    //     l1_write_addr_act += conv_act_c_read_bytes;
    //     act_l1_offset += stride_w_bytes;
    // }
    // l1_write_addr_act += act_block_w_extra_align_bytes;

    // act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
    // for (uint32_t inner = 0; inner < weight_size_w; inner++) {
    //     noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
    //     l1_write_addr_act += conv_act_c_read_bytes;
    //     act_l1_offset += stride_w_bytes;
    // }
    // l1_write_addr_act += act_block_w_extra_align_bytes;

    // DPRINT << "bytes; " << unpadded_block_row_size_bytes << " batch: " << batch << " num_unpadded_rows_per_batch: "
    // << num_unpadded_rows_per_batch << ENDL();

    constexpr uint32_t unpadded_block_row_size_bytes = 96;
    constexpr uint32_t batch = 1;
    constexpr uint32_t num_unpadded_rows_per_batch = 2048;
    constexpr uint32_t padded_block_row_size_bytes = 128;
    constexpr uint32_t num_padded_tiles_per_batch = 128;

    // DPRINT << "cb_id_untilize_out: " << cb_id_untilize_out << ENDL();
    // DPRINT << "cb_id_out: " << cb_id_out << ENDL();

    // DPRINT << "padded_block_row_size_bytes: " << padded_block_row_size_bytes << ENDL();

    // noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
    for (uint32_t b = 0; b < batch; ++b) {
        uint64_t noc_l1_read_addr = get_noc_addr(get_read_ptr(cb_id_untilize_out));
        noc_async_read_one_packet_set_state(noc_l1_read_addr, unpadded_block_row_size_bytes);
        cb_wait_front(cb_id_untilize_out, num_padded_tiles_per_batch / 2);

        for (uint32_t row = 0; row < num_unpadded_rows_per_batch / 2; ++row) {
            noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            // noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            noc_l1_read_addr += padded_block_row_size_bytes;
            l1_write_addr += unpadded_block_row_size_bytes;

            // noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            // // noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            // noc_l1_read_addr += padded_block_row_size_bytes;
            // l1_write_addr += unpadded_block_row_size_bytes;

            // noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            // // noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            // noc_l1_read_addr += padded_block_row_size_bytes;
            // l1_write_addr += unpadded_block_row_size_bytes;

            // noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            // // noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            // noc_l1_read_addr += padded_block_row_size_bytes;
            // l1_write_addr += unpadded_block_row_size_bytes;

            // noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            // // noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            // noc_l1_read_addr += padded_block_row_size_bytes;
            // l1_write_addr += unpadded_block_row_size_bytes;

            // noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            // // noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            // noc_l1_read_addr += padded_block_row_size_bytes;
            // l1_write_addr += unpadded_block_row_size_bytes;
        }

        // cb_pop_front(cb_id_untilize_out, num_padded_tiles_per_batch);
    }
    noc_async_read_barrier();
    // cb_push_back(cb_id_out, num_unpadded_output_rows);
}
