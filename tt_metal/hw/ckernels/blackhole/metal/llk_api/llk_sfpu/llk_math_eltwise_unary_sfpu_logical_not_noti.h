// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_logical_not_noti.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_LOGICAL_NOT_NOTI_KERNEL(logical_not_unary, sfpi::vFloat, float, sfpi::vInt, int16_t)
