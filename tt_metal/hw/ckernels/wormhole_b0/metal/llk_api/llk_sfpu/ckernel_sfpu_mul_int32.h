// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;
        // operand A - int32
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        // operand B - int32
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_offset * dst_tile_size);

        TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0xffff);

        // // copy
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);

        // upper bits
        TTI_SFPSHFT((-16), p_sfpu::LREG0, p_sfpu::LREG2, 1);  // LREG2 = A[31:16]
        TTI_SFPSHFT((-16), p_sfpu::LREG1, p_sfpu::LREG3, 1);  // LREG3 = B[31:16]

        // lower bits
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);  // LREG0 = A[15:0]
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);  // LREG1 = B[15:0]

        // int16 -> fp32
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);  // a_low
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);  // b_low
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);  // a_hi
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);  // b_hi

        // multiply in fp32
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // L4 = A_lo * B_lo
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // L5 = A_hi * B_hi
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // L0 = A_lo + A_hi
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // L1 = B_lo + B_hi
        TTI_SFPMUL(
            p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);  // L6 = (A_lo + A_hi) * (B_lo + B_hi)

        // cast back
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG4, p_sfpu::LREG4, 7);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG5, p_sfpu::LREG5, 7);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        // TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG7, p_sfpu::LREG7, 7);

        // Karatsuba Formula: A × B = P1 * 2^32 * (P2 - P1 - P0) * 2^16 + P0
        // L5 * 2^32 * (L6 - L5 - L4) * 2^16 + L4
        // L4 + (L6 - L5 - L4) << 16
        // shift back
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 6);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 6);
        TTI_SFPSHFT(16, 0, p_sfpu::LREG4, 1);
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, SFPIADD_MOD1_CC_NONE);

        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
