#include <cstdint>

#include "dataflow_api.h"
#include "risc_common.h"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t width = get_compile_time_arg_val(2);
    constexpr uint32_t height = get_compile_time_arg_val(3);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);

    const uint32_t sticks_to_transfer = get_arg_val<uint32_t>(0);
    const uint32_t input_offset = get_arg_val<uint32_t>(1);
    const uint32_t start_col = get_arg_val<uint32_t>(2);
    const uint32_t output_stride_width = get_arg_val<uint32_t>(3);
    const uint32_t output_stride_height = get_arg_val<uint32_t>(4);
    const uint32_t output_offset = get_arg_val<uint32_t>(5);

    uint32_t stick_size = stick_size_bytes / 2;

    // We will iterate over all cores and write the output to their L1 memory
    auto src_address = get_write_ptr(src_cb_id) + input_offset;

    uint32_t dst_noc_x = VIRTUAL_TENSIX_START_X;
    uint32_t dst_noc_y = VIRTUAL_TENSIX_START_Y;

    for (uint32_t src_core = 0; src_core < 64; src_core++) {
        auto dst_noc_address = get_noc_addr(dst_noc_x, dst_noc_y, get_read_ptr(dst_cb_id)) + output_offset;
        // noc_async_write(dst_noc_address, stick_size_bytes);

        // Copy sticks assigned to this core from src to dst
        uint32_t col = start_col;
        for (uint32_t stick_no = 0; stick_no < sticks_to_transfer; stick_no++) {
            noc_async_write(src_address, dst_noc_address, stick_size_bytes);
            src_address += stick_size_bytes;
            col++;
            // wrap rows
            if (col >= width) {
                col = 0;
                // adjust dst address to skip rows on output where we don't need to write
                dst_noc_address += output_stride_height;
            } else {
                // skip elements other cores will write
                dst_noc_address += output_stride_width;
            }
        }

        // We've transferred all sticks so we need to move to the next core
        // make sure to wrap the core grid
        dst_noc_x++;
        if (dst_noc_x >= VIRTUAL_TENSIX_START_X + 8) {
            dst_noc_x = VIRTUAL_TENSIX_START_X;
            dst_noc_y++;
        }
    }

    noc_async_write_barrier();
}
