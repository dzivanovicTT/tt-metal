// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/cluster_descriptor_types.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/fabric_types.hpp>

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const IntermeshLinkDescriptor& intermesh_link_descrptor);
IntermeshLinkDescriptor deserialize_from_bytes(const std::vector<uint8_t>& data);

} // namespace tt::tt_fabric
