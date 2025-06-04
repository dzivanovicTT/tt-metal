// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

// Functions to pack/unpack the data of ND sharded tensors into a vector of bytes, in such a way that all data going to
// a single memory bank is contiguous.
template <typename T>
std::vector<std::vector<T>> pack_nd_sharded_data(tt::stl::Span<const T> data, const TensorSpec& tensor_spec);
template <typename T>
std::vector<T> unpack_nd_sharded_data(tt::stl::Span<const T> sharded_data, const TensorSpec& tensor_spec);

}  // namespace tt::tt_metal
