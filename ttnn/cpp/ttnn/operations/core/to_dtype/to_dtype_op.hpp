// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/api.hpp"

namespace ttnn {

namespace operations {

namespace core {

namespace detail {

inline Tensor convert_to_cpp_supported_dtype(const Tensor& input_tensor) {
    TT_FATAL(
        input_tensor.storage_type() == StorageType::HOST, "convert_to_cpp_supported_dtype only supports host tensors");
    const auto& host_storage = std::get<tt::tt_metal::HostStorage>(input_tensor.storage());
    auto input_dtype = input_tensor.dtype();

    if (input_dtype != DataType::BFLOAT8_B && input_dtype != DataType::BFLOAT4_B) {
        return input_tensor;
    }

    return Tensor(
        host_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<uint32_t>(buffer);
            auto float_unpacked_data =
                input_dtype == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false)
                    : unpack_bfp4_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        }),
        TensorSpec(
            input_tensor.logical_shape(),
            tt::tt_metal::TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                tt::tt_metal::PageConfig(input_tensor.layout()),
                MemoryConfig{},
                input_tensor.logical_shape(),
                input_tensor.padded_shape())),
        input_tensor.distributed_tensor_config());
}

template <typename NewT, typename OldT>
std::vector<NewT> cast(tt::stl::Span<const OldT> input_buffer) {
    std::vector<NewT> output_vector(input_buffer.size());
    for (auto index = 0; index < input_buffer.size(); ++index) {
        auto convert_value = [](auto&& value) {
            if constexpr (std::is_same_v<OldT, bfloat16>) {
                return value.to_float();
            } else if constexpr (std::is_same_v<NewT, bfloat16>) {
                return static_cast<float>(value);
            } else {
                return value;
            }
        };
        auto value = input_buffer[index];
        output_vector[index] = static_cast<NewT>(convert_value(value));
    }
    return output_vector;
}

template <typename T>
Tensor create_tensor_from_span(
    tt::stl::Span<const T> input_buffer,
    const Shape& logical_shape,
    const Shape& padded_shape,
    const Layout& input_layout,
    const DataType& dtype) {
    switch (dtype) {
        case DataType::UINT16: {
            auto data = cast<uint16_t, T>(input_buffer);
            return Tensor(
                       tt::tt_metal::HostBuffer(std::move(data)), logical_shape, padded_shape, dtype, Layout::ROW_MAJOR)
                .to_layout(input_layout);
        }
        case DataType::INT32: {
            auto data = cast<int32_t, T>(input_buffer);
            return Tensor(
                       tt::tt_metal::HostBuffer(std::move(data)), logical_shape, padded_shape, dtype, Layout::ROW_MAJOR)
                .to_layout(input_layout);
        }
        case DataType::UINT32: {
            auto data = cast<uint32_t, T>(input_buffer);
            return Tensor(
                       tt::tt_metal::HostBuffer(std::move(data)), logical_shape, padded_shape, dtype, Layout::ROW_MAJOR)
                .to_layout(input_layout);
        }
        case DataType::FLOAT32: {
            auto data = cast<float, T>(input_buffer);
            return Tensor(
                       tt::tt_metal::HostBuffer(std::move(data)), logical_shape, padded_shape, dtype, Layout::ROW_MAJOR)
                .to_layout(input_layout);
        }
        case DataType::BFLOAT16: {
            auto data = cast<::bfloat16, T>(input_buffer);
            return Tensor(
                       tt::tt_metal::HostBuffer(std::move(data)), logical_shape, padded_shape, dtype, Layout::ROW_MAJOR)
                .to_layout(input_layout);
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            auto data = cast<float, T>(input_buffer);
            auto buffer = tt::tt_metal::HostBuffer(std::move(data));
            auto tensor = Tensor(std::move(buffer), logical_shape, padded_shape, DataType::FLOAT32, Layout::ROW_MAJOR)
                              .to_layout(Layout::TILE);
            tt::stl::Span<const float> output_float_data = tt::tt_metal::host_buffer::get_as<float>(tensor);
            auto output_packed_data =
                dtype == DataType::BFLOAT8_B
                    ? pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false)
                    : pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return Tensor(
                tt::tt_metal::HostBuffer(std::move(output_packed_data)),
                logical_shape,
                padded_shape,
                dtype,
                Layout::TILE);  // has to be in tile layout
        }
        default: {
            TT_THROW("Unsupported DataType: {}", dtype);
            break;
        }
    }
}

inline Tensor convert_to_dtype(const Tensor& input_tensor, const Layout& input_layout, const DataType& dtype) {
    auto input_dtype = input_tensor.dtype();
    const auto& logical_shape = input_tensor.logical_shape();
    const auto& padded_shape = input_tensor.padded_shape();

    auto convert_dtype = [&input_layout, &input_dtype, &dtype, &logical_shape, &padded_shape](
                             const tt::tt_metal::HostBuffer& input_tensor) {
        switch (input_dtype) {
            case DataType::UINT16: {
                auto buffer = tt::tt_metal::host_buffer::get_as<uint16_t>(input_tensor);
                return tt::tt_metal::host_buffer::get_host_buffer(
                    create_tensor_from_span(buffer, logical_shape, padded_shape, input_layout, dtype));
            }
            case DataType::INT32: {
                auto buffer = tt::tt_metal::host_buffer::get_as<int32_t>(input_tensor);
                return tt::tt_metal::host_buffer::get_host_buffer(
                    create_tensor_from_span(buffer, logical_shape, padded_shape, input_layout, dtype));
            }
            case DataType::UINT32: {
                auto buffer = tt::tt_metal::host_buffer::get_as<uint32_t>(input_tensor);
                return tt::tt_metal::host_buffer::get_host_buffer(
                    create_tensor_from_span(buffer, logical_shape, padded_shape, input_layout, dtype));
            }
            case DataType::FLOAT32: {
                auto buffer = tt::tt_metal::host_buffer::get_as<float>(input_tensor);
                return tt::tt_metal::host_buffer::get_host_buffer(
                    create_tensor_from_span(buffer, logical_shape, padded_shape, input_layout, dtype));
            }
            case DataType::BFLOAT16: {
                auto buffer = tt::tt_metal::host_buffer::get_as<::bfloat16>(input_tensor);
                return tt::tt_metal::host_buffer::get_host_buffer(
                    create_tensor_from_span(buffer, logical_shape, padded_shape, input_layout, dtype));
            }
            default: TT_THROW("Unsupported DataType: {}", input_dtype); break;
        }
    };

    const Layout output_layout =
        (dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B) ? Layout::TILE : input_layout;

    return Tensor(
        std::get<tt::tt_metal::HostStorage>(input_tensor.storage()).transform(convert_dtype),
        TensorSpec(
            logical_shape,
            tt::tt_metal::TensorLayout::fromPaddedShape(
                dtype, tt::tt_metal::PageConfig(output_layout), MemoryConfig{}, logical_shape, padded_shape)),
        input_tensor.distributed_tensor_config());
}

}  // namespace detail

struct ToDtype {
    // TODO: Move to cpp once we merge with tt_eager
    static Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype) {
        auto input_layout = input_tensor.layout();
        auto input_dtype = input_tensor.dtype();

        if (input_dtype == dtype) {
            return input_tensor;
        }

        TT_FATAL(is_cpu_tensor(input_tensor), "to_dtype only supports host tensors");

        auto row_major_input_tensor = input_tensor.to_layout(ttnn::ROW_MAJOR_LAYOUT);

        auto intermediate_tensor = detail::convert_to_cpp_supported_dtype(row_major_input_tensor);
        return detail::convert_to_dtype(intermediate_tensor, input_layout, dtype);
    };
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
