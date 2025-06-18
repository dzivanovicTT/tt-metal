// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emitc.hpp"

namespace ttnn {
namespace test {

template <typename... T>
std::vector<ttnn::Tensor> util_create_vec(T &&...t) {
  return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}

::std::vector<::ttnn::Tensor> create_inputs_for_prepare_conv2d_weights() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({16, 8, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> prepare_conv2d_weights(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 8, 16, 3, 32, 32, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 2, 1, 3}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

TEST(EmitC, PrepareConv2dWeights) {
  ::std::vector<::ttnn::Tensor> v1 = create_inputs_for_prepare_conv2d_weights();
  ::std::vector<::ttnn::Tensor> v2 = prepare_conv2d_weights(v1);
  std::cout << v2[0].write_to_string() << std::endl;
  ::ttnn::Tensor hostTensor = ::ttnn::to_layout(v2[0], ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt, static_cast<::ttnn::MeshDevice *>(nullptr));
}

}  // namespace test
}  // namespace ttnn
