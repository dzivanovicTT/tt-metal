// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_backward_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {
void bind_moreh_linear_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_linear_backward,
        "Moreh Linear Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::arg("input"),
            nb::arg("weight"),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true, true},

            nb::arg("bias") = std::nullopt,
            nb::arg("input_grad") = std::nullopt,
            nb::arg("weight_grad") = std::nullopt,
            nb::arg("bias_grad") = std::nullopt,

            nb::arg("input_grad_memory_config") = std::nullopt,
            nb::arg("weight_grad_memory_config") = std::nullopt,
            nb::arg("bias_grad_memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_linear_backward
