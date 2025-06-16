// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"

inline void wait() {
    for (uint32_t i = 0; i < 2; i++) {
        asm volatile("nop");
        asm volatile("fence");
        invalidate_l1_cache();
    }
}

uint32_t cb_srca = 256;  // outside of any cbs
uint32_t cb_srcb = 256;  // outside of any cbs

constexpr std::uint32_t onetile = 1;

template <bool to_from_int8 = false>
ALWI void test_func(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    // DPRINT << "AAAAA 1\n";
    //  DPRINT << "A" << ENDL();
    asm volatile("add t1,zero,zero\n add t1,t1,zero" ::: "t1");
}

inline bool dummy_compare(std::uint32_t old_operand, std::uint32_t new_operand) {
    // return true;
    return (unpack_src_format[old_operand] != unpack_src_format[new_operand]) ||
           (unpack_dst_format[old_operand] != unpack_dst_format[new_operand]);
}

namespace NAMESPACE {
void MAIN {
    // Circular Buffers
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_other = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Variables
    constexpr uint32_t input_dst_reg = 0;

    binary_op_init_common(cb_in, cb_other, cb_out);

    tile_regs_acquire();
    cb_wait_front(cb_in, onetile);
    cb_wait_front(cb_other, onetile);
    volatile uint32_t i = 0;

    {
        volatile bool x = dummy_compare(cb_srca, cb_in);
        // DPRINT << "DST_ACCUM_MODE " << static_cast<uint32_t>(DST_ACCUM_MODE) << "\n";
        if (cb_srca == 256 || cb_srcb == 256) {
            test_func<DST_ACCUM_MODE>(cb_in, cb_other);
        } else {
            UNPACK(if (dummy_compare(cb_srca, cb_in)) { TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0); })
        }
        i = 0xdedbef01;
        cb_srca = cb_in;
        cb_srcb = cb_other;
    }

    // DPRINT << "CBS " << (uint32_t)cb_in << " " << (uint32_t)cb_other << " " << (uint32_t)cb_srca <<" " <<
    // (uint32_t)cb_srcb << ENDL();
    mul_tiles_init(cb_in, cb_other);
    mul_tiles(cb_in, cb_other, 0, 0, 0);
    // tile_regs_commit();
    cb_pop_front(cb_in, onetile);
    cb_pop_front(cb_other, onetile);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, onetile);
    // tile_regs_wait();
    pack_tile(0, cb_out);
    // tile_regs_release();
    cb_push_back(cb_out, onetile);
    tile_regs_release();

    // cb_wait_front(cb_out, 1);
    // UNPACK(tt::compute::common::print_full_tile(cb_out, 0, false);)
    // wait();
}
}  // namespace NAMESPACE
