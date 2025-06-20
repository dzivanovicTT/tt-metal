
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::examples {

// The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op is registered
struct ExampleOperation {    
    static Tensor invoke(const Tensor& input_tensor);
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto example =
    ttnn::register_operation<"ttnn::example", operations::examples::ExampleOperation>();
}  // namespace ttnn
