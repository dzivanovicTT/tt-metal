// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "flatbuffers/flatbuffers.h"

namespace tt::tt_metal::distributed {

std::vector<std::byte> serialize_to_bytes(const SocketPeerDescriptor& socket_md);
SocketPeerDescriptor deserialize_from_bytes(const std::vector<std::byte>& data);

}  // namespace tt::tt_metal::distributed
