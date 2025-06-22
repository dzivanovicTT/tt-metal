// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <unordered_map>

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

struct EthChanDescriptor {
    uint64_t board_id = 0;  // Unique board identifier
    uint32_t chan_id = 0;   // Eth Channel ID

    bool operator==(const EthChanDescriptor& other) const {
        return board_id == other.board_id && chan_id == other.chan_id;
    }
};

}  // namespace tt::tt_fabric

namespace std {

template<>
struct hash<tt::tt_fabric::EthChanDescriptor> {
    std::size_t operator()(const tt::tt_fabric::EthChanDescriptor& desc) const {
        return std::hash<uint64_t>{}(desc.board_id) ^ 
               (std::hash<uint32_t>{}(desc.chan_id) << 1);
    }
};

}  // namespace std

namespace tt::tt_fabric {

struct IntermeshLinkTable {
    // Local mesh ID
    MeshId local_mesh_id = MeshId{0};
    // Maps local eth channel to remote eth channel
    std::unordered_map<EthChanDescriptor, EthChanDescriptor> intermesh_links;
};

}  // namespace tt::tt_fabric
