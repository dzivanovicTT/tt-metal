// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::data_movement {

namespace nb = nanobind;
void bind_sharded_to_interleaved_partial(nb::module_& mod);

}  // namespace ttnn::operations::data_movement
