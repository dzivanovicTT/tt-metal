// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/multi_mesh_utils.hpp"
#include "intermesh_link_descriptor_generated.h"

namespace tt::tt_fabric {

// Helper function to create EthCoord FlatBuffer object
flatbuffers::Offset<flatbuffer::EthCoord> create_eth_coord(flatbuffers::FlatBufferBuilder& builder, const eth_coord_t& coord) {
    return flatbuffer::CreateEthCoord(builder, coord.cluster_id, coord.x, coord.y, coord.rack, coord.shelf);
}

// Helper function to create CoreCoord FlatBuffer object
flatbuffers::Offset<flatbuffer::CoreCoord> create_core_coord(flatbuffers::FlatBufferBuilder& builder, const CoreCoord& coord) {
    return flatbuffer::CreateCoreCoord(builder, coord.x, coord.y);
}

// Helper function to create MeshId FlatBuffer object
flatbuffers::Offset<flatbuffer::MeshId> create_mesh_id(flatbuffers::FlatBufferBuilder& builder, const MeshId& mesh_id) {
    return flatbuffer::CreateMeshId(builder, *mesh_id);
}

// Helper function to create EthernetLinkDescriptor FlatBuffer object
flatbuffers::Offset<flatbuffer::EthernetLinkDescriptor> create_ethernet_link_descriptor(
    flatbuffers::FlatBufferBuilder& builder, 
    const EthernetLinkDescriptor& link_desc) {
    
    auto local_chip_eth_coord = create_eth_coord(builder, link_desc.local_chip_eth_coord);
    auto remote_chip_eth_coord = create_eth_coord(builder, link_desc.remote_chip_eth_coord);
    auto local_eth_core_coord = create_core_coord(builder, link_desc.local_eth_core_coord);
    auto remote_eth_core_coord = create_core_coord(builder, link_desc.remote_eth_core_coord);
    
    return flatbuffer::CreateEthernetLinkDescriptor(builder, 
                                      local_chip_eth_coord,
                                      remote_chip_eth_coord,
                                      local_eth_core_coord,
                                      remote_eth_core_coord);
}

std::vector<uint8_t> serialize_to_bytes(const IntermeshLinkDescriptor& intermesh_link_descriptor) {
    flatbuffers::FlatBufferBuilder builder;
    
    // Create MeshId
    auto mesh_id = create_mesh_id(builder, intermesh_link_descriptor.local_mesh_id);
    
    // Create vector of EthernetLinkDescriptor objects
    std::vector<flatbuffers::Offset<flatbuffer::EthernetLinkDescriptor>> ethernet_links;
    ethernet_links.reserve(intermesh_link_descriptor.intermesh_links.size());
    
    for (const auto& link_desc : intermesh_link_descriptor.intermesh_links) {
        ethernet_links.push_back(create_ethernet_link_descriptor(builder, link_desc));
    }
    
    // Create the vector in FlatBuffer
    auto intermesh_links_vector = builder.CreateVector(ethernet_links);
    
    // Create the root IntermeshLinkDescriptor
    auto intermesh_link_desc = CreateIntermeshLinkDescriptor(builder, mesh_id, intermesh_links_vector);
    
    // Finish the buffer
    builder.Finish(intermesh_link_desc);
    
    // Return the serialized data
    return std::vector<uint8_t>(builder.GetBufferPointer(), 
                               builder.GetBufferPointer() + builder.GetSize());
}

eth_coord_t convert_eth_coord(const flatbuffer::EthCoord* fb_coord) {
    eth_coord_t coord;
    coord.cluster_id = fb_coord->cluster_id();
    coord.x = fb_coord->x();
    coord.y = fb_coord->y();
    coord.rack = fb_coord->rack();
    coord.shelf = fb_coord->shelf();
    return coord;
}

// Helper function to convert FlatBuffer CoreCoord to C++ CoreCoord
CoreCoord convert_core_coord(const flatbuffer::CoreCoord* fb_coord) {
    return CoreCoord(fb_coord->x(), fb_coord->y());
}

// Helper function to convert FlatBuffer MeshId to C++ MeshId
MeshId convert_mesh_id(const flatbuffer::MeshId* fb_mesh_id) {
    return MeshId(fb_mesh_id->value());
}

// Helper function to convert FlatBuffer EthernetLinkDescriptor to C++ EthernetLinkDescriptor
EthernetLinkDescriptor convert_ethernet_link_descriptor(const flatbuffer::EthernetLinkDescriptor* fb_link_desc) {
    EthernetLinkDescriptor link_desc;
    link_desc.local_chip_eth_coord = convert_eth_coord(fb_link_desc->local_chip_eth_coord());
    link_desc.remote_chip_eth_coord = convert_eth_coord(fb_link_desc->remote_chip_eth_coord());
    link_desc.local_eth_core_coord = convert_core_coord(fb_link_desc->local_eth_core_coord());
    link_desc.remote_eth_core_coord = convert_core_coord(fb_link_desc->remote_eth_core_coord());
    return link_desc;
}

IntermeshLinkDescriptor deserialize_from_bytes(const std::vector<uint8_t>& data) {
    // Verify the buffer
    flatbuffers::Verifier verifier(data.data(), data.size());
    if (!flatbuffer::VerifyIntermeshLinkDescriptorBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data");
    }
    
    // Get the root object
    const flatbuffer::IntermeshLinkDescriptor* fb_intermesh_desc = flatbuffer::GetIntermeshLinkDescriptor(data.data());
    
    // Convert to C++ structure
    IntermeshLinkDescriptor intermesh_desc;
    intermesh_desc.local_mesh_id = convert_mesh_id(fb_intermesh_desc->local_mesh_id());
    
    // Convert the vector of ethernet links
    const auto* fb_links = fb_intermesh_desc->intermesh_links();
    if (fb_links) {
        intermesh_desc.intermesh_links.reserve(fb_links->size());
        for (const auto* fb_link : *fb_links) {
            intermesh_desc.intermesh_links.push_back(convert_ethernet_link_descriptor(fb_link));
        }
    }
    
    return intermesh_desc;
}

} // namespace tt::tt_fabric
