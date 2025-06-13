// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reinterleave_pybind.hpp"
#include "reinterleave.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::reinterleave {
void bind_reinterleave_operation(pybind11::module& module) {
    auto doc = R"doc(reinterleave(input: Tensor, dtype: DataType, memory_config: MemoryConfig) -> Tensor

    Reinterleaves the input, creating a copy with the specified `memory_config` and converting its data type to `dtype`.
    This operation does not alter the tensor's layout.
    - ROW_MAJOR_LAYOUT: Returns the tensor unpadded in the last two dimensions.
    - TILE_LAYOUT: Pads the tensor to ensure its width and height are multiples of 32.
    If the input's current layout matches the specified layout, padding adjustments are applied to the last two dimensions as necessary.

    Args:
        * :attr:`input`: The tensor to be Reinterleaved.
        * :attr:`dtype`: The target data type of the Reinterleaved tensor.
        * :attr:`memory_config`: The memory configuration for the Reinterleave, options include DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG.
        * :attr:`compute_kernel_config`: The configuration for the compute kernel.
    )doc";

    bind_registered_operation(
        module,
        ttnn::experimental::reinterleave_from_batch,
        doc,
        ttnn::pybind_arguments_t{
            pybind11::arg("input"),
            pybind11::kw_only(),
            pybind11::arg("input_height"),
            pybind11::arg("input_width"),
            pybind11::arg("stride_hw") = std::array<uint32_t, 2>{2, 2},
            pybind11::arg("barrier_threshold") = 0,
            pybind11::arg("compute_kernel_config") = std::nullopt,
        });

    bind_registered_operation(
        module,
        ttnn::experimental::reinterleave_local,
        doc,
        ttnn::pybind_arguments_t{
            pybind11::arg("input"),
            pybind11::kw_only(),
            pybind11::arg("input_height"),
            pybind11::arg("input_width"),
            pybind11::arg("stride_hw") = std::array<uint32_t, 2>{2, 2},
            pybind11::arg("barrier_threshold") = 0,
            pybind11::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::experimental::reinterleave
