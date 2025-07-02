// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::reduction::detail {

namespace nb = nanobind;

void bind_argmax_operation(nb::module_& mod);

void bind_argmin_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::reduction::detail
