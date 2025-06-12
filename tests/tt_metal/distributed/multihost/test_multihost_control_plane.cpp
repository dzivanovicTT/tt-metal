// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/allocator_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "gmock/gmock.h"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt-metalium/system_mesh.hpp>

#include <impl/context/metal_context.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/fabric_types.hpp>

#ifdef __linux__
#include <execinfo.h>
#include <cstdlib>
#endif
namespace tt::tt_metal::distributed {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::tt::tt_fabric::ControlPlaneMode;
using ::tt::tt_fabric::HostRankId;
using ::tt::tt_fabric::MeshId;
using ::tt::tt_metal::distributed::MeshContainer;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshShape;

// Test fixture for ControlPlane tests

std::map<tt_fabric::FabricNodeId, chip_id_t> get_physical_chip_mapping() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto dctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    auto mpi_rank = dctx->rank();
    auto mpi_size = dctx->size();

    fmt::print("MPI rank: {}\n", *mpi_rank);
    fmt::print("MPI size: {}\n", *mpi_size);

    std::map<tt_fabric::FabricNodeId, chip_id_t> physical_chip_ids_mapping;



    // This should match the mesh descriptor: Mesh 0 has 8 chips (2x4 layout)
    // The dual host configuration has host rank 0 controlling chips 0-3 and host rank 1 controlling chips 4-7

    auto rank0_eth_coords = std::vector<eth_coord_t>{
        eth_coord_t{0, 0, 0, 0, 0},
        eth_coord_t{0, 1, 0, 0, 0},
        eth_coord_t{0, 0, 1, 0, 0},
        eth_coord_t{0, 1, 1, 0, 0},
    };
    // TODO: NEED TO VERIFY THESE COORDINATES
    auto rank1_eth_coords = std::vector<eth_coord_t>{
        eth_coord_t{0, 1, 0, 0, 0},
        eth_coord_t{0, 0, 0, 0, 0},
        eth_coord_t{0, 1, 1, 0, 0},
        eth_coord_t{0, 0, 1, 0, 0}
    };
    auto chip_ids_rank0 = std::vector<chip_id_t>{
        0, 1, 4, 5
    };
    auto chip_ids_rank1 = std::vector<chip_id_t>{
        2, 3, 6, 7
    };
    auto eth_coords = *mpi_rank == 0 ? rank0_eth_coords : rank1_eth_coords;
    auto fabric_chip_ids = *mpi_rank == 0 ? chip_ids_rank0 : chip_ids_rank1;

    std::uint32_t mesh_id = 0;
    for (std::uint32_t chip_id = 0; chip_id < eth_coords.size(); chip_id++) {
        const auto& eth_coord = eth_coords[chip_id];
        auto fabric_chip_id = fabric_chip_ids.at(chip_id);
        physical_chip_ids_mapping.insert(
            {tt_fabric::FabricNodeId(tt_fabric::MeshId{mesh_id}, fabric_chip_id),
             cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
    }

    return physical_chip_ids_mapping;
}

class ControlPlaneMultihostTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set environment variables for different test scenarios
        if (!multihost::DistributedContext::is_initialized()) {
            multihost::DistributedContext::create(0, nullptr);
        }

        fmt::print("Setting up test with mesh graph descriptor...\n");
        auto physical_chip_mapping = get_physical_chip_mapping();
        fmt::print("Physical chip mapping size: {}\n", physical_chip_mapping.size());
        for (const auto& [fabric_node_id, chip_id] : physical_chip_mapping) {
            fmt::print(
                "  FabricNodeId(mesh={}, chip={}) -> physical chip {}\n",
                *fabric_node_id.mesh_id,
                fabric_node_id.chip_id,
                chip_id);
        }

        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml",
            physical_chip_mapping);

        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        mesh_id_ = control_plane.get_local_mesh_id();
        auto host_rank_opt = control_plane.get_local_host_rank_id();
        ASSERT_TRUE(host_rank_opt.has_value()) << "Local host rank ID not available";
        host_rank_id_ = *host_rank_opt;
        if (fabric_config_ == tt::tt_metal::FabricConfig::DISABLED) {
            tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::FABRIC_1D);
            fabric_config_ = tt::tt_metal::FabricConfig::FABRIC_1D;
        }

        fmt::print("SetUp complete: mesh_id={}, host_rank_id={}\n", *mesh_id_, *host_rank_id_);
    }

    void TearDown() override {
        if (fabric_config_ != tt::tt_metal::FabricConfig::DISABLED) {
            tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
            fabric_config_ = tt::tt_metal::FabricConfig::DISABLED;
        }
    }

    tt::tt_metal::FabricConfig fabric_config_ = tt::tt_metal::FabricConfig::DISABLED;
    MeshId mesh_id_;
    HostRankId host_rank_id_;
};

TEST_F(ControlPlaneMultihostTest, ValidateMPIWorldSize) {
    auto context = multihost::DistributedContext::get_current_world();
    auto mpi_size = context->size();
    fmt::print("MPI size: {}\n", *mpi_size);
    EXPECT_EQ(*mpi_size, 2);
}


TEST_F(ControlPlaneMultihostTest, ValidateMPIRankToMeshIdHostRankId) {
    // Validate MPI rank maps to correct mesh and host rank
    auto context = multihost::DistributedContext::get_current_world();
    auto mpi_rank = context->rank();

    if (*mpi_rank == 0) {
        EXPECT_EQ(mesh_id_, MeshId{0});
        EXPECT_EQ(host_rank_id_, HostRankId{0});
    } else {
        EXPECT_EQ(mesh_id_, MeshId{0});
        EXPECT_EQ(host_rank_id_, HostRankId{1});
    }
}

TEST_F(ControlPlaneMultihostTest, GetMeshShapeLocalMode) {
    // Get mesh shape in local mode
    auto local_shape =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_shape(ControlPlaneMode::LOCAL_MESH);

    auto context = multihost::DistributedContext::get_current_world();
    auto mpi_rank = context->rank();

    fmt::print(
        "MPI Rank {}: Local mesh shape is [{}, {}], expected [2, 2]\n", *mpi_rank, local_shape[0], local_shape[1]);

    EXPECT_EQ(local_shape, MeshShape(2, 2));
}

TEST_F(ControlPlaneMultihostTest, GetMeshShapeGlobalMode) {
    auto global_shape =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_shape(ControlPlaneMode::GLOBAL_MESH);
    fmt::print("Global mesh shape is [{}, {}], expected [2, 4]\n", global_shape[0], global_shape[1]);
    EXPECT_EQ(global_shape, MeshShape(2, 4));
}


TEST_F(ControlPlaneMultihostTest, SystemMeshShape) {
    auto distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    auto& system_mesh = tt::tt_metal::distributed::SystemMesh::instance();
    auto shape = system_mesh.get_shape();
    fmt::print("System mesh shape is [{}, {}], expected [2, 4]\n", shape[0], shape[1]);

    EXPECT_EQ(shape, MeshShape(2, 4));

    if (*distributed_context->rank() == 0) {
        EXPECT_EQ(system_mesh.local_shape(), MeshShape(2, 2));
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(0, 0)), 2);
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(0, 1)), 0);
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(1, 0)), 3);
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(1, 1)), 1);
    } else {
        std::cout << "**************Rank 1**************" << std::endl;
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        auto physical_chip_id = control_plane.get_physical_chip_id(tt_fabric::FabricNodeId(MeshId{0}, 2));
        fmt::print("FabricNodeId(mesh={}, chip={}) -> physical chip {}\n", 0, 2, physical_chip_id.value());
        physical_chip_id = control_plane.get_physical_chip_id(tt_fabric::FabricNodeId(MeshId{0}, 3));
        fmt::print("FabricNodeId(mesh={}, chip={}) -> physical chip {}\n", 0, 3, physical_chip_id.value());
        physical_chip_id = control_plane.get_physical_chip_id(tt_fabric::FabricNodeId(MeshId{0}, 6));
        fmt::print("FabricNodeId(mesh={}, chip={}) -> physical chip {}\n", 0, 6, physical_chip_id.value());
        physical_chip_id = control_plane.get_physical_chip_id(tt_fabric::FabricNodeId(MeshId{0}, 7));
        fmt::print("FabricNodeId(mesh={}, chip={}) -> physical chip {}\n", 0, 7, physical_chip_id.value());

        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto chip_ids = cluster.all_chip_ids();
        for (auto chip_id : chip_ids) {
            fmt::print("CLUSTER -> Chip ID: {}\n", chip_id);
        }




        EXPECT_EQ(system_mesh.local_shape(), MeshShape(2, 2));
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(0, 2)), 0);
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(0, 3)), 2);
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(1, 2)), 1);
        EXPECT_EQ(system_mesh.get_physical_device_id(MeshCoordinate(1, 3)), 3);
    }
}


TEST_F(ControlPlaneMultihostTest, MeshDevice2x4) {
    {
        auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    }
    std::cout << "MeshDevice2x4" << std::endl;
}

}  // namespace
}  // namespace tt::tt_metal::distributed
