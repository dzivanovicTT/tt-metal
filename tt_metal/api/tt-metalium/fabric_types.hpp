// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt_stl/strong_type.hpp>
#include <umd/device/types/cluster_descriptor_types.h>
#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {
enum class FabricConfig : uint32_t {
    DISABLED = 0,
    FABRIC_1D = 1,          // Instatiates fabric with 1D routing and no deadlock avoidance
    FABRIC_1D_RING = 2,     // Instatiates fabric with 1D routing and with deadlock avoidance using datelines
    FABRIC_2D = 3,          // Instatiates fabric with 2D routing
    FABRIC_2D_TORUS = 4,    // Instatiates fabric with 2D routing and with deadlock avoidance using datelines
    FABRIC_2D_DYNAMIC = 5,  // Instatiates fabric with 2D routing with dynamic routing
    CUSTOM = 6
};

}  // namespace tt::tt_metal

namespace tt::tt_fabric {

using MeshId = tt::stl::StrongType<uint32_t, struct MeshIdTag>;
using HostRankId = tt::stl::StrongType<uint32_t, struct HostRankTag>;

struct EthernetLinkDescriptor { // RemoteEthLinkDescriptor
    eth_coord_t local_chip_eth_coord;  // 
    eth_coord_t remote_chip_eth_coord; // chip x y
    CoreCoord local_eth_core_coord;
    CoreCoord remote_eth_core_coord;
};

struct IntermeshLinkDescriptor {
    MeshId local_mesh_id = MeshId{0};
    std::vector<EthernetLinkDescriptor> intermesh_links;
};

}  // namespace tt::tt_fabric
