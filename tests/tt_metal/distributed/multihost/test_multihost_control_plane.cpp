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
    std::map<tt_fabric::FabricNodeId, chip_id_t> physical_chip_ids_mapping;

    // This should match the mesh descriptor: Mesh 0 has 8 chips (2x4 layout)
    // The dual host configuration has host rank 0 controlling chips 0-3 and host rank 1 controlling chips 4-7
    std::vector<eth_coord_t> mesh_0_eth_coords = {
        eth_coord_t{0, 0, 0, 0, 0},
        eth_coord_t{0, 1, 0, 0, 0},
        eth_coord_t{0, 2, 0, 0, 0},
        eth_coord_t{0, 3, 0, 0, 0},  // Row 0
        eth_coord_t{0, 0, 1, 0, 0},
        eth_coord_t{0, 1, 1, 0, 0},
        eth_coord_t{0, 2, 1, 0, 0},
        eth_coord_t{0, 3, 1, 0, 0}  // Row 1
    };

    std::uint32_t mesh_id = 0;
    for (std::uint32_t chip_id = 0; chip_id < mesh_0_eth_coords.size(); chip_id++) {
        const auto& eth_coord = mesh_0_eth_coords[chip_id];
        physical_chip_ids_mapping.insert(
            {tt_fabric::FabricNodeId(tt_fabric::MeshId{mesh_id}, chip_id),
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

        try {
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
            host_rank_id_ = control_plane.get_local_host_rank_id();

            fmt::print("SetUp complete: mesh_id={}, host_rank_id={}\n", *mesh_id_, *host_rank_id_);
        } catch (const std::exception& e) {
            fmt::print("Exception in SetUp: {}\n", e.what());
            throw;
        }
    }

    MeshId mesh_id_;
    HostRankId host_rank_id_;
};

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

}  // namespace
}  // namespace tt::tt_metal::distributed
