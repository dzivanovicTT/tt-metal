// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "clone.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::data_movement::clone {
void bind_clone_operation(nb::module_& mod) {
    auto doc = R"doc(clone(input: Tensor, dtype: DataType, memory_config: MemoryConfig) -> Tensor

    Clones the input, creating a copy with the specified `memory_config` and converting its data type to `dtype`.
    This operation does not alter the tensor's layout.
    - ROW_MAJOR_LAYOUT: Returns the tensor unpadded in the last two dimensions.
    - TILE_LAYOUT: Pads the tensor to ensure its width and height are multiples of 32.
    If the input's current layout matches the specified layout, padding adjustments are applied to the last two dimensions as necessary.

    Args:
        * :attr:`input`: The tensor to be cloned.
        * :attr:`dtype`: The target data type of the cloned tensor.
        * :attr:`memory_config`: The memory configuration for the clone, options include DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG.
        * :attr:`compute_kernel_config`: The configuration for the compute kernel.
    )doc";

    bind_registered_operation(
        mod,
        ttnn::clone,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::data_movement::clone
