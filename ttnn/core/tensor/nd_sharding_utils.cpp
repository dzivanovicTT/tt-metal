// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nd_sharding_utils.hpp"

#include <tt-metalium/buffer_distribution_spec.hpp>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T, typename U, bool pack>
void pack_unpack_nd_sharded_data_impl(
    tt::stl::Span<T> data, tt::stl::Span<U> sharded_data, const TensorSpec& tensor_spec) {
    if (tensor_spec.padded_shape().volume() == 0) {
        return;
    }

    const auto& memory_config = tensor_spec.memory_config();
    const auto& shape = tensor_spec.padded_shape();
    auto shape_span = shape.view();
    const auto& physical_shape = tensor_spec.physical_shape();
    const auto& strides = tensor_spec.compute_strides();
    auto strides_span = tt::stl::make_const_span(strides);
    const auto& shard_spec = memory_config.nd_shard_spec().value();
    const auto& shard_shape = shard_spec.shard_shape;
    auto shard_shape_span = shard_shape.view();
    auto shard_size = shard_shape.volume();
    size_t shard_width = shard_shape[-1];

    size_t num_shards = 1;
    tt::stl::SmallVector<size_t> num_shards_per_dim(shape_span.size());
    for (size_t i = 0; i < shape_span.size(); i++) {
        num_shards_per_dim[i] = (shape_span[i] + shard_shape_span[i] - 1) / shard_shape_span[i];
        num_shards *= num_shards_per_dim[i];
    }
    auto num_shards_per_dim_span = tt::stl::make_const_span(num_shards_per_dim);
    size_t num_cores = shard_spec.grid.num_cores();
    size_t num_shards_per_core = (num_shards + num_cores - 1) / num_cores;

    tt::stl::SmallVector<size_t> shard_strides(shape_span.size());
    shard_strides.back() = 1;
    for (int i = static_cast<int>(shape_span.size()) - 2; i >= 0; i--) {
        shard_strides[i] = shard_strides[i + 1] * shard_shape_span[i + 1];
    }
    auto shard_strides_span = tt::stl::make_const_span(shard_strides);

    tt::stl::SmallVector<size_t> shard_index_strides(shape_span.size());
    shard_index_strides.back() = 1;
    for (int i = static_cast<int>(num_shards_per_dim_span.size()) - 2; i >= 0; i--) {
        shard_index_strides[i] = shard_index_strides[i + 1] * num_shards_per_dim_span[i + 1];
    }
    auto shard_index_strides_span = tt::stl::make_const_span(shard_index_strides);

    // static tf::Executor executor;
    // tf::Taskflow taskflow;

    // taskflow.for_each_index(static_cast<size_t>(0), physical_shape.height(), static_cast<size_t>(1), [&](size_t
    // row_idx) {
    for (size_t row_idx = 0; row_idx < physical_shape.height(); row_idx++) {
        for (size_t col_block_idx = 0; col_block_idx < num_shards_per_dim_span.back(); col_block_idx++) {
            size_t element_idx = row_idx * physical_shape.width() + col_block_idx * shard_width;

            size_t src_offset = 0;
            size_t offset_within_shard = 0;
            size_t shard_idx = 0;
            size_t element_idx_tmp = element_idx;
            for (int i = static_cast<int>(shape_span.size()) - 1; i >= 0; i--) {
                size_t element_coord = element_idx_tmp % shape_span[i];
                src_offset += element_coord * strides_span[i];
                size_t shard_coord = element_coord / shard_shape_span[i];
                shard_idx += shard_coord * shard_index_strides_span[i];
                size_t coord_within_shard = element_coord % shard_shape_span[i];
                offset_within_shard += coord_within_shard * shard_strides_span[i];
                element_idx_tmp /= shape_span[i];
            }

            size_t core_idx = shard_idx % num_cores;
            size_t shard_idx_within_core = shard_idx / num_cores;
            size_t shard_offset = (num_shards_per_core * core_idx + shard_idx_within_core) * shard_size;
            size_t dst_offset = shard_offset + offset_within_shard;

            size_t num_bytes_to_copy = shard_width * sizeof(T);
            bool last_shard_in_row = col_block_idx == num_shards_per_dim_span.back() - 1;
            if (last_shard_in_row && shape_span.back() % shard_width != 0) {
                num_bytes_to_copy = (shape_span.back() % shard_width) * sizeof(T);
            }

            if constexpr (pack) {
                std::memcpy(sharded_data.data() + dst_offset, data.data() + src_offset, num_bytes_to_copy);
            } else {
                std::memcpy(data.data() + src_offset, sharded_data.data() + dst_offset, num_bytes_to_copy);
            }
        }
    }

    // executor.run(taskflow).wait();
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
std::vector<std::vector<T>> pack_nd_sharded_data(tt::stl::Span<const T> data, const TensorSpec& tensor_spec) {
    const auto& memory_config = tensor_spec.memory_config();
    const auto& shape = tensor_spec.padded_shape();
    const auto& shard_spec = memory_config.nd_shard_spec().value();
    const auto& shard_shape = shard_spec.shard_shape;
    auto shard_size = shard_shape.volume();

    size_t num_shards = 1;
    for (size_t i = 0; i < shape.rank(); i++) {
        num_shards *= (shape[i] + shard_shape[i] - 1) / shard_shape[i];
    }
    size_t num_cores = shard_spec.grid.num_cores();
    size_t num_shards_per_core = (num_shards + num_cores - 1) / num_cores;

    /*
    CMAKE_UNIQUE_NAMESPACE::pack_unpack_nd_sharded_data_impl<const T, T, true>(
        data, sharded_data, tensor_spec);*/

    auto dist_spec = BufferDistributionSpec::from_shard_spec(
        tensor_spec.padded_shape(),
        tensor_spec.memory_config().nd_shard_spec()->shard_shape,
        tensor_spec.tensor_layout().compute_page_shape(tensor_spec.physical_shape()),
        tensor_spec.memory_config().nd_shard_spec()->grid,
        tensor_spec.memory_config().nd_shard_spec()->orientation);
    auto page_size = tensor_spec.compute_page_size_bytes();
    auto aligned_page_size = page_size;  // tensor_spec.compute_aligned_page_size_bytes();
    auto mode = page_size == aligned_page_size ? DistributionSpec::MappingMode::COALESCED
                                               : DistributionSpec::MappingMode::NONCOALESCED;
    auto page_mapping = dist_spec.get_page_mapping(mode);

    std::vector<std::vector<T>> sharded_data(num_cores);

    static tf::Executor executor;
    tf::Taskflow taskflow;

    taskflow.for_each_index(0, static_cast<int>(page_mapping.size()), 1, [&](int target_idx) {
        const auto& target_data = page_mapping[target_idx];
        sharded_data[target_idx].resize(num_shards_per_core * shard_size);
        for (const auto& chunk_mapping : target_data) {
            std::memcpy(
                sharded_data[target_idx].data() + chunk_mapping.dst * aligned_page_size,
                data.data() + chunk_mapping.src * page_size,
                chunk_mapping.size * page_size);
        }
    });
    executor.run(taskflow).wait();

    return sharded_data;
}

template <typename T>
std::vector<T> unpack_nd_sharded_data(tt::stl::Span<const T> sharded_data, const TensorSpec& tensor_spec) {
    std::vector<T> data(tensor_spec.padded_shape().volume());
    CMAKE_UNIQUE_NAMESPACE::pack_unpack_nd_sharded_data_impl<T, const T, false>(data, sharded_data, tensor_spec);
    return data;
}

template std::vector<std::vector<bfloat16>> pack_nd_sharded_data(
    tt::stl::Span<const bfloat16> data, const TensorSpec& tensor_spec);
template std::vector<std::vector<float>> pack_nd_sharded_data(
    tt::stl::Span<const float> data, const TensorSpec& tensor_spec);
template std::vector<std::vector<double>> pack_nd_sharded_data(
    tt::stl::Span<const double> data, const TensorSpec& tensor_spec);
template std::vector<std::vector<int32_t>> pack_nd_sharded_data(
    tt::stl::Span<const int32_t> data, const TensorSpec& tensor_spec);
template std::vector<std::vector<uint32_t>> pack_nd_sharded_data(
    tt::stl::Span<const uint32_t> data, const TensorSpec& tensor_spec);
template std::vector<std::vector<uint16_t>> pack_nd_sharded_data(
    tt::stl::Span<const uint16_t> data, const TensorSpec& tensor_spec);
template std::vector<std::vector<uint8_t>> pack_nd_sharded_data(
    tt::stl::Span<const uint8_t> data, const TensorSpec& tensor_spec);

template std::vector<bfloat16> unpack_nd_sharded_data(
    tt::stl::Span<const bfloat16> data, const TensorSpec& tensor_spec);
template std::vector<float> unpack_nd_sharded_data(tt::stl::Span<const float> data, const TensorSpec& tensor_spec);
template std::vector<double> unpack_nd_sharded_data(tt::stl::Span<const double> data, const TensorSpec& tensor_spec);
template std::vector<int32_t> unpack_nd_sharded_data(tt::stl::Span<const int32_t> data, const TensorSpec& tensor_spec);
template std::vector<uint32_t> unpack_nd_sharded_data(
    tt::stl::Span<const uint32_t> data, const TensorSpec& tensor_spec);
template std::vector<uint16_t> unpack_nd_sharded_data(
    tt::stl::Span<const uint16_t> data, const TensorSpec& tensor_spec);
template std::vector<uint8_t> unpack_nd_sharded_data(tt::stl::Span<const uint8_t> data, const TensorSpec& tensor_spec);

}  // namespace tt::tt_metal
