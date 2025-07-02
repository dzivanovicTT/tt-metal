// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "prefix_scan.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace py = nanobind;

void bind_prefix_scan(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::prefix_scan);

    const auto doc =
        R"doc(Performs a prefix scan to produce the SSM hidden states across an entire sequence. All input and output tensors are expected to be shape [1, 1, L, 2EN]. Values of 2EN and L can be any multiple of 32.)doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::prefix_scan,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& a,
               const ttnn::Tensor& bx,
               const ttnn::Tensor& h_prev,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType> dtype,
               const std::optional<MathFidelity> math_fidelity,
               QueueId queue_id) { return self(queue_id, a, bx, h_prev, memory_config, dtype, math_fidelity); },
            nb::arg("a"),
            nb::arg("bx"),
            nb::arg("h_prev"),
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("math_fidelity") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::ssm::detail
