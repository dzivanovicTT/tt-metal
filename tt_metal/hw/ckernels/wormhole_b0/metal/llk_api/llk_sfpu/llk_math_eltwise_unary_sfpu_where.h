// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_where(
    uint dst_index, uint param0, uint param1, uint param2, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_ternary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::_calculate_where_<APPROXIMATE>, dst_index, vector_mode, param0, param1, param2);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_where_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::unused>();
}
}  // namespace ckernel
