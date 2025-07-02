// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>

#include "ttnn-nanobind/json_class.hpp"
#include "ttnn-nanobind/export_enum.hpp"

#include <tt-metalium/host_buffer.hpp>
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

namespace nanobind::detail {
template <>
struct dtype_traits<::bfloat16> {
    static constexpr dlpack::dtype value{
        static_cast<uint8_t>(nanobind::dlpack::dtype_code::Bfloat),  // type code
        16,                                                          // size in bits
        1                                                            // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("bfloat16");
};
}  // namespace nanobind::detail

using namespace tt::tt_metal;

namespace ttnn::tensor {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

namespace detail {

template <class T>
struct DataTypeToFormatType {
    using type = T;
};

// https://github.com/wjakob/nanobind/blob/master/docs/ndarray.rst#nonstandard-arithmetic-types

// template <>
// struct DataTypeToFormatType<::bfloat16> {
//     using type = uint16_t;
// };

}  // namespace detail

void tensor_mem_config_module_types(nb::module_& m_tensor) {
    export_enum<Layout>(m_tensor);
    export_enum<DataType>(m_tensor);
    export_enum<StorageType>(m_tensor);
    // export_enum<MathFidelity>(m_tensor);

    // for whatever reason using magic_enum for this in particular just threw
    // std::bad_cast errors in the binding code when trying to import ttnn
    // in python. It threw the error in ttnn-nanobind/operations/core.cpp
    // when setting a default arg to MathFidelity::Invalid.
    // The problem went away locally when I did this manually.
    // Why? I have no idea. There might be some UB buried in export_enum
    // or magic_enum.
    nb::enum_<MathFidelity>(m_tensor, "MathFidelity")
        .value("LoFi", MathFidelity::LoFi)
        .value("HiFi2", MathFidelity::HiFi2)
        .value("HiFi3", MathFidelity::HiFi3)
        .value("HiFi4", MathFidelity::HiFi4)
        .value("Invalid", MathFidelity::Invalid)
        .export_values();

    export_enum<TensorMemoryLayout>(m_tensor);
    export_enum<ShardOrientation>(m_tensor);
    export_enum<ShardMode>(m_tensor);

    nb::enum_<tt::tt_metal::BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1)
        .value("L1_SMALL", BufferType::L1_SMALL)
        .value("TRACE", BufferType::TRACE);

    tt_serializable_class<tt::tt_metal::CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    nb::class_<Tile>(m_tensor, "Tile", R"doc(
        Class defining tile dims
    )doc");

    nb::class_<ttnn::TensorSpec>(m_tensor, "TensorSpec", R"doc(
        Class defining the specification of Tensor
    )doc");

    tt_serializable_class<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    tt_serializable_class<tt::tt_metal::ShardSpec>(m_tensor, "ShardSpec", R"doc(
        Class defining the specs required for sharding.
    )doc");

    tt_serializable_class<tt::tt_metal::NdShardSpec>(m_tensor, "NdShardSpec", R"doc(
        Class defining the specs required for ND sharding.
        Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.
    )doc");

    tt_serializable_class<tt::tt_metal::CoreRange>(m_tensor, "CoreRange", R"doc(
        Class defining a range of cores)doc");

    tt_serializable_class<tt::tt_metal::CoreRangeSet>(m_tensor, "CoreRangeSet", R"doc(
        Class defining a set of CoreRanges required for sharding)doc");

    // the buffer_protocol was dropped in nanobind in favor of ndarray.
    // So either have to write a caster or figure out how to map HostBuffer to ndarray

    // nb::ndarray<uint8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>

    // def_buffer removed in nanobind. use nb::ndarray instead
    // note: ndarray has several gotchas. See for more information:
    // https://github.com/wjakob/nanobind/blob/master/docs/ndarray.rst
    nb::class_<tt::tt_metal::HostBuffer>(m_tensor, "HostBuffer")
        .def("__getitem__", [](const HostBuffer& self, std::size_t index) { return self.view_bytes()[index]; })
        .def("__len__", [](const HostBuffer& self) { return self.view_bytes().size(); })
        .def(
            "__iter__",
            [](const HostBuffer& self) {
                return nb::make_iterator(
                    nb::type<tt::tt_metal::HostBuffer>(),
                    "iterator",
                    self.view_bytes().begin(),
                    self.view_bytes().end());
            },
            nb::keep_alive<0, 1>());
}

void tensor_mem_config_module(nb::module_& m_tensor) {
    auto py_core_coord = static_cast<nb::class_<CoreCoord>>(m_tensor.attr("CoreCoord"));

    py_core_coord.def(nb::init<std::size_t, std::size_t>())
        .def(
            "__init__",
            [](CoreCoord* t, std::tuple<std::size_t, std::size_t> core_coord) {
                new (t) CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
            })
        .def("__repr__", [](const CoreCoord& self) -> std::string { return self.str(); })
        .def_ro("x", &CoreCoord::x)
        .def_ro("y", &CoreCoord::y);
    nb::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();

    auto py_tile = static_cast<nb::class_<Tile>>(m_tensor.attr("Tile"));
    py_tile
        .def(nb::init<const std::array<uint32_t, 2>&, bool>(), nb::arg("tile_shape"), nb::arg("transpose_tile") = false)
        .def(
            "__init__",
            [](Tile* t, const std::array<uint32_t, 2>& tile_shape, bool transpose_tile = false) {
                new (t) Tile{tile_shape, transpose_tile};
            })
        .def(
            "__repr__",
            [](const Tile& self) {
                return fmt::format("Tile with shape: [{}, {}]", self.get_tile_shape()[0], self.get_tile_shape()[1]);
            })
        .def_ro("tile_shape", &Tile::tile_shape)
        .def_ro("face_shape", &Tile::face_shape)
        .def_ro("num_faces", &Tile::num_faces)
        .def_ro("partial_face", &Tile::partial_face)
        .def_ro("narrow_tile", &Tile::narrow_tile)
        .def_ro("transpose_within_face", &Tile::transpose_within_face)
        .def_ro("transpose_of_faces", &Tile::transpose_of_faces);

    auto pyTensorSpec = static_cast<nb::class_<TensorSpec>>(m_tensor.attr("TensorSpec"));
    pyTensorSpec.def("shape", &TensorSpec::logical_shape, "Logical shape of a tensor")
        .def("layout", &TensorSpec::layout, "Layout of a tensor")
        .def("dtype", &TensorSpec::data_type, "Dtype of a tensor");

    auto pyMemoryConfig = static_cast<nb::class_<MemoryConfig>>(m_tensor.attr("MemoryConfig"));
    pyMemoryConfig
        .def(
            "__init__",
            [](MemoryConfig* t,
               TensorMemoryLayout memory_layout,
               BufferType buffer_type,
               std::optional<ShardSpec> shard_spec) {
                new (t) MemoryConfig(memory_layout, buffer_type, std::move(shard_spec));
            },
            nb::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
            nb::arg("buffer_type") = BufferType::DRAM,
            nb::arg("shard_spec") = std::nullopt,
            R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.SINGLE_BANK)
            )doc")
        .def(
            "__init__",
            [](MemoryConfig* t, BufferType buffer_type, NdShardSpec nd_shard_spec) {
                new (t) MemoryConfig(buffer_type, std::move(nd_shard_spec));
            },
            nb::arg("buffer_type"),
            nb::arg("nd_shard_spec"),
            R"doc(
                Create MemoryConfig class.
                This constructor is used to create MemoryConfig for ND sharded tensors.
                Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.

                Example of creating MemoryConfig for ND sharded tensors.

                .. code-block:: python

                    mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([1, 1, 1, 1]), ttnn.CoreRangeSet([ttnn.CoreCoord(0, 0)])))
            )doc")
        .def(
            "__hash__",
            [](const MemoryConfig& memory_config) -> tt::stl::hash::hash_t {
                return tt::stl::hash::detail::hash_object(memory_config);
            })
        .def("is_sharded", &MemoryConfig::is_sharded, "Whether tensor data is sharded across multiple cores in L1")
        .def(
            "with_shard_spec",
            &MemoryConfig::with_shard_spec,
            "Returns a new MemoryConfig with the shard spec set to the given value")
        .def_prop_ro(
            "interleaved",
            [](const MemoryConfig& memory_config) {
                return memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED;
            },
            "Whether tensor data is interleaved across multiple DRAM channels")
        .def_prop_ro("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1")
        .def_prop_ro("memory_layout", &MemoryConfig::memory_layout, "Memory layout of tensor data.")
        .def_prop_ro("shard_spec", &MemoryConfig::shard_spec, "Memory layout of tensor data.")
        .def_prop_ro("nd_shard_spec", &MemoryConfig::nd_shard_spec, "ND shard spec of tensor data.")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    m_tensor.def(
        "dump_memory_config",
        nb::overload_cast<const std::string&, const MemoryConfig&>(&dump_memory_config),
        R"doc(
            Dump memory config to file
        )doc");
    m_tensor.def(
        "load_memory_config",
        nb::overload_cast<const std::string&>(&load_memory_config),
        R"doc(
            Load memory config to file
        )doc");

    auto pyCoreRange = static_cast<nb::class_<CoreRange>>(m_tensor.attr("CoreRange"));
    pyCoreRange
        .def(
            "__init__",
            [](CoreRange* t, const CoreCoord& start, const CoreCoord& end) { new (t) CoreRange{start, end}; })
        .def_ro("start", &CoreRange::start_coord)
        .def_ro("end", &CoreRange::end_coord)
        .def("grid_size", &CoreRange::grid_size);

    auto pyCoreRangeSet = static_cast<nb::class_<CoreRangeSet>>(m_tensor.attr("CoreRangeSet"));
    pyCoreRangeSet
        .def(
            "__init__",
            [](CoreRangeSet* t, const std::set<CoreRange>& core_ranges) { new (t) CoreRangeSet(core_ranges); })
        .def(
            "__init__",
            [](CoreRangeSet* t, const std::vector<CoreRange>& core_ranges) {
                new (t) CoreRangeSet(tt::stl::Span<const CoreRange>(core_ranges));
            })
        .def(
            "bounding_box",
            &CoreRangeSet::bounding_box,
            "Returns a CoreRange i.e. bounding box covering all the core ranges in the CoreRangeSet")
        .def("num_cores", &CoreRangeSet::num_cores, "Returns total number of cores in the CoreRangeSet")
        .def("subtract", &CoreRangeSet::subtract, "Subtract common CoreRanges from current i.e. it returns A - (AnB)")
        .def("ranges", &CoreRangeSet::ranges, "Returns the core ranges in the CoreRangeSet");

    auto pyShardSpec = static_cast<nb::class_<ShardSpec>>(m_tensor.attr("ShardSpec"));
    pyShardSpec
        .def(
            "__init__",
            [](ShardSpec* t,
               const CoreRangeSet& core_sets,
               const std::array<uint32_t, 2>& shard_shape,
               const ShardOrientation& shard_orientation,
               const ShardMode& shard_mode) {
                new (t) ShardSpec(core_sets, shard_shape, shard_orientation, shard_mode);
            },
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("shard_orientation"),
            nb::arg("shard_mode") = ShardMode::PHYSICAL)
        .def(
            "__init__",
            [](ShardSpec* t,
               const CoreRangeSet& core_sets,
               const std::array<uint32_t, 2>& shard_shape,
               const std::array<uint32_t, 2>& physical_shard_shape,
               const ShardOrientation& shard_orientation) {
                new (t) ShardSpec(core_sets, shard_shape, physical_shard_shape, shard_orientation);
            },
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("physical_shard_shape"),
            nb::arg("shard_orientation"))
        .def_rw("shape", &ShardSpec::shape, "Shape of shard.")
        .def_rw("grid", &ShardSpec::grid, "Grid to layout shards.")
        .def_rw("orientation", &ShardSpec::orientation, "Orientation of cores to read shards")
        .def_rw("mode", &ShardSpec::mode, "Treat shard shape as physical (default) or logical")
        .def("num_cores", &ShardSpec::num_cores, "Number of cores")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    auto pyNdShardSpec = static_cast<nb::class_<NdShardSpec>>(m_tensor.attr("NdShardSpec"));
    pyNdShardSpec
        .def(
            "__init__",
            [](NdShardSpec* t,
               const ttnn::Shape& shard_shape,
               const CoreRangeSet& grid,
               const ShardOrientation& orientation) { new (t) NdShardSpec(shard_shape, grid, orientation); },
            nb::arg("shard_shape"),
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR)
        .def_rw("shard_shape", &NdShardSpec::shard_shape, "Shape of shard.")
        .def_rw("grid", &NdShardSpec::grid, "Grid to layout shards.")
        .def_rw("orientation", &NdShardSpec::orientation, "Orientation of cores to distribute shards")
        .def(
            "num_cores", [](const NdShardSpec& self) { return self.grid.num_cores(); }, "Number of cores")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    m_tensor.def(
        "dump_tensor",
        &dump_tensor,
        nb::arg("filename"),
        nb::arg("tensor"),
        nb::arg("strategy") = std::unordered_map<std::string, std::string>{},
        R"doc(
            Dump tensor to file
        )doc");

    m_tensor.def(
        "load_tensor",
        nb::overload_cast<const std::string&, MeshDevice*>(&load_tensor),
        nb::arg("file_name"),
        nb::arg("device") = nullptr,
        R"doc(Load tensor to file)doc");
}

}  // namespace ttnn::tensor
