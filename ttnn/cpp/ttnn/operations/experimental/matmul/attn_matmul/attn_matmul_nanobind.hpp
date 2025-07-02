// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::matmul::detail {

namespace nb = nanobind;

void bind_attn_matmul(nb::module_& mod);
void bind_attn_matmul_from_cache(nb::module_& mod);

}  // namespace ttnn::operations::experimental::matmul::detail
