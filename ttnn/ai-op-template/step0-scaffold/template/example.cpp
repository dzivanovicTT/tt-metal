
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "example.hpp"

namespace ttnn::operations::examples {

Tensor ExampleOperation::invoke(const Tensor& input_tensor) {
    return input_tensor;
}

}  // namespace ttnn::operations::examples