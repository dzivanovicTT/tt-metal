// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_identity.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_SIMPLE_TWO_PARAM_KERNEL_SUFFIX(identity, calculate_identity, 8)
SFPU_SIMPLE_TWO_PARAM_KERNEL_SUFFIX(identity_uint32, calculate_identity_uint, 8)
SFPU_IDENTITY_INIT()
