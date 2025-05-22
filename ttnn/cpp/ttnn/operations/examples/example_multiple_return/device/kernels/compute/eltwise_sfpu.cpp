// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"

static uint32_t cb_srca = 256;  // outside of any cbs
static uint32_t cb_srcb = 256;  // outside of any cbs

namespace NAMESPACE {
void MAIN {
    // Circular Buffers
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_other = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    binary_op_init_common(cb_in, cb_other, cb_out);

    tile_regs_acquire();
    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_other, 1);

    // These DPRINTS are necessary
    UNPACK(DPRINT << "DST_ACCUM_MODE " << static_cast<uint32_t>(DST_ACCUM_MODE) << "\n");
    if (cb_srca == 256 || cb_srcb == 256) {
        UNPACK(DPRINT << "UNPACK 1\n");
        MATH(DPRINT << "  MATH 1\n");
        PACK(DPRINT << "  PACK 1\n");
    } else {
        UNPACK(
            if (unpack_src_format[cb_srca] != unpack_src_format[cb_in] ||
                unpack_dst_format[cb_srca] != unpack_dst_format[cb_in]) {
                TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
            });
    }

    cb_srca = cb_in;
    cb_srcb = cb_other;

    mul_tiles_init(cb_in, cb_other);
    mul_tiles(cb_in, cb_other, 0, 0, 0);
    //    PACK(tt::compute::common::print_full_tile(cb_other, 0, false));

    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_other, 1);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    cb_wait_front(cb_out, 1);
    // UNPACK(tt::compute::common::print_full_tile(cb_out, 0, false));
}
}  // namespace NAMESPACE
