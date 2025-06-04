// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fmt/base.h>
#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/logger.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

// Test parameters structure
struct FabricBandwidthTestParams {
    std::string test_name;
    bool is_unicast;
    std::string noc_message_type;
    size_t num_messages;
    size_t num_links;
    size_t num_op_invocations;
    bool line_sync;
    size_t line_size;
    size_t packet_size;
    FabricTestMode fabric_mode;
    bool disable_sends_for_interior_workers;
    bool unidirectional;
    bool senders_are_unidirectional;
    std::string test_mode;
    size_t num_cluster_rows;
    size_t num_cluster_cols;

    // For gtest output formatting
    friend std::ostream& operator<<(std::ostream& os, const FabricBandwidthTestParams& params) {
        return os << params.test_name << "_" << params.packet_size << "B_" << params.num_links << "L";
    }
};

// Convert FabricTestMode to the common header enum (no conversion needed since they're the same now)
FabricTestMode convertFabricTestMode(FabricTestMode mode) { return mode; }

// noc_send_type, flush
static std::tuple<tt::tt_fabric::NocSendType, bool> get_noc_send_type(const std::string& message_noc_type) {
    tt::tt_fabric::NocSendType noc_send_type;
    bool flush = true;
    if (message_noc_type == "noc_unicast_write") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE;
    } else if (message_noc_type == "noc_multicast_write") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE;
    } else if (message_noc_type == "noc_unicast_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC;
        flush = true;
    } else if (message_noc_type == "noc_unicast_no_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC;
        flush = false;
    } else if (message_noc_type == "noc_fused_unicast_write_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC;
        flush = true;
    } else if (message_noc_type == "noc_fused_unicast_write_no_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC;
        flush = false;
    } else {
        TT_THROW("Invalid message type: {}", message_noc_type.c_str());
    }

    return std::make_tuple(noc_send_type, flush);
}

static int baseline_validate_test_environment(const WriteThroughputStabilityTestWithPersistentFabricParams& params) {
    uint32_t min_test_num_devices = 8;
    if (tt::tt_metal::GetNumAvailableDevices() < min_test_num_devices) {
        tt::log_warning("This test can only be run on T3000 or TG devices");
        return 1;
    }

    uint32_t galaxy_num_devices = 32;
    if (params.num_links > 2 && tt::tt_metal::GetNumAvailableDevices() < galaxy_num_devices) {
        tt::log_warning("This test with {} links can only be run on Galaxy systems", params.num_links);
        return 1;
    }

    if (tt::tt_metal::GetNumAvailableDevices() == min_test_num_devices && params.num_links > 1 &&
        params.line_size > 4) {
        tt::log_warning("T3000 cannot run multi-link with more than 4 devices");
        return 1;
    }

    return 0;
}

// Main test execution function that replaces the subprocess call
void run_fabric_edm_test(const FabricBandwidthTestParams& test_params) {
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = test_params.line_sync;
    params.line_size = test_params.line_size;
    params.num_links = test_params.num_links;
    params.num_op_invocations = test_params.num_op_invocations;
    params.fabric_mode = convertFabricTestMode(test_params.fabric_mode);
    params.disable_sends_for_interior_workers = test_params.disable_sends_for_interior_workers;
    params.disable_end_workers_in_backward_direction = test_params.unidirectional;
    params.senders_are_unidirectional = test_params.senders_are_unidirectional;
    params.num_fabric_rows = test_params.num_cluster_rows;
    params.num_fabric_cols = test_params.num_cluster_cols;
    params.first_link_offset = 0;

    auto rc = baseline_validate_test_environment(params);
    if (rc != 0) {
        GTEST_SKIP() << "Test environment validation failed";
    }

    TT_FATAL(test_params.packet_size > 0, "packet_payload_size_bytes must be greater than 0");
    TT_FATAL(test_params.num_links > 0, "num_links must be greater than 0");
    TT_FATAL(test_params.num_op_invocations > 0, "num_op_invocations must be greater than 0");
    TT_FATAL(test_params.line_size > 0, "line_size must be greater than 0");

    auto chip_send_type = test_params.is_unicast ? tt::tt_fabric::CHIP_UNICAST : tt::tt_fabric::CHIP_MULTICAST;
    auto [noc_send_type, flush] = get_noc_send_type(test_params.noc_message_type);

    std::vector<Fabric1DPacketSendTestSpec> test_specs{
        {.chip_send_type = chip_send_type,
         .noc_send_type = noc_send_type,
         .num_messages = test_params.num_messages,
         .packet_payload_size_bytes = test_params.packet_size,
         .flush = flush}};

    if (test_params.test_mode == "1_fabric_instance") {
        Run1DFabricPacketSendTest(test_specs, params);
    } else if (test_params.test_mode == "1D_fabric_on_mesh") {
        TT_FATAL(
            test_params.num_cluster_rows > 0 ^ test_params.num_cluster_cols > 0,
            "Either num_rows or num_cols (but not both) must be greater than 0 when running 1D fabric on mesh BW test");

        if (params.fabric_mode == ::FabricTestMode::Linear) {
            Run1DFabricPacketSendTest<Fabric1DLineDeviceInitFixture>(test_specs, params);
        } else if (params.fabric_mode == ::FabricTestMode::FullRing) {
            Run1DFabricPacketSendTest<Fabric1DRingDeviceInitFixture>(test_specs, params);
        } else {
            TT_THROW(
                "Invalid fabric mode when using device init fabric in 1D fabric on mesh BW test: {}",
                params.fabric_mode);
        }
    } else {
        TT_THROW("Invalid test mode: {}", test_params.test_mode.c_str());
    }
}

// Parameterized test fixture
class FabricBandwidthParameterizedTest : public ::testing::TestWithParam<FabricBandwidthTestParams> {
protected:
    void SetUp() override {
        // Check environment variables and hardware requirements
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set";
        }

        auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();

        if (!(arch == tt::ARCH::WORMHOLE_B0 && num_devices >= 8)) {
            GTEST_SKIP() << "This suite can only be run on T3000 or TG Wormhole devices";
        }
    }
};

// Single parameterized test that covers all cases
TEST_P(FabricBandwidthParameterizedTest, FabricBandwidthTest) {
    const auto& params = GetParam();
    run_fabric_edm_test(params);
}

// Generate test parameters that mirror the pytest parametrize decorators
std::vector<FabricBandwidthTestParams> GenerateTestParams() {
    std::vector<FabricBandwidthTestParams> params;

    // Test: test_fabric_edm_mcast_half_ring_bw
    // @pytest.mark.parametrize("line_size, num_links", [(4, 1), (4, 2), (4, 3), (4, 4)])
    for (auto [line_size, num_links] : std::vector<std::pair<size_t, size_t>>{{4, 1}, {4, 2}, {4, 3}, {4, 4}}) {
        params.push_back(
            {.test_name = "FabricEdmMcastHalfRingBw",
             .is_unicast = false,
             .noc_message_type = "noc_unicast_write",
             .num_messages = 200000,
             .num_links = num_links,
             .num_op_invocations = 1,
             .line_sync = true,
             .line_size = line_size,
             .packet_size = 4096,
             .fabric_mode = FabricTestMode::HalfRing,
             .disable_sends_for_interior_workers = false,
             .unidirectional = false,
             .senders_are_unidirectional = false,
             .test_mode = "1_fabric_instance",
             .num_cluster_rows = 0,
             .num_cluster_cols = 0});
    }

    // // Test: test_fabric_4chip_one_link_mcast_full_ring_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     params.push_back({
    //         .test_name = "Fabric4ChipOneLinkMcastFullRingBw",
    //         .is_unicast = false,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = 1,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 4,
    //         .packet_size = packet_size,
    //         .fabric_mode = FabricTestMode::FullRing,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = false,
    //         .test_mode = "1_fabric_instance",
    //         .num_cluster_rows = 0,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_4chip_multi_link_mcast_full_ring_bw
    // // @pytest.mark.parametrize("num_links", [2, 3, 4])
    // for (auto num_links : std::vector<size_t>{2, 3, 4}) {
    //     params.push_back({
    //         .test_name = "Fabric4ChipMultiLinkMcastFullRingBw",
    //         .is_unicast = false,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = num_links,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 4,
    //         .packet_size = 4096,
    //         .fabric_mode = FabricTestMode::FullRing,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = false,
    //         .test_mode = "1_fabric_instance",
    //         .num_cluster_rows = 0,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_8chip_one_link_edm_mcast_full_ring_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     params.push_back({
    //         .test_name = "Fabric8ChipOneLinkEdmMcastFullRingBw",
    //         .is_unicast = false,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = 1,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 8,
    //         .packet_size = packet_size,
    //         .fabric_mode = FabricTestMode::FullRing,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = false,
    //         .test_mode = "1_fabric_instance",
    //         .num_cluster_rows = 0,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_8chip_multi_link_edm_mcast_full_ring_bw
    // // @pytest.mark.parametrize("num_links", [2, 3, 4])
    // for (auto num_links : std::vector<size_t>{2, 3, 4}) {
    //     params.push_back({
    //         .test_name = "Fabric8ChipMultiLinkEdmMcastFullRingBw",
    //         .is_unicast = false,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = num_links,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 8,
    //         .packet_size = 4096,
    //         .fabric_mode = FabricTestMode::FullRing,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = false,
    //         .test_mode = "1_fabric_instance",
    //         .num_cluster_rows = 0,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_4chip_multi_link_edm_unicast_full_ring_bw
    // // @pytest.mark.parametrize("num_links", [1, 2, 3, 4])
    // for (auto num_links : std::vector<size_t>{1, 2, 3, 4}) {
    //     params.push_back({
    //         .test_name = "Fabric4ChipMultiLinkEdmUnicastFullRingBw",
    //         .is_unicast = true,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = num_links,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 4,
    //         .packet_size = 4096,
    //         .fabric_mode = FabricTestMode::FullRing,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = false,
    //         .test_mode = "1_fabric_instance",
    //         .num_cluster_rows = 0,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_4_chip_one_link_mcast_saturate_chip_to_chip_ring_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     params.push_back({
    //         .test_name = "Fabric4ChipOneLinkMcastSaturateChipToChipRingBw",
    //         .is_unicast = false,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = 1,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 4,
    //         .packet_size = packet_size,
    //         .fabric_mode = FabricTestMode::SaturateChipToChipRing,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = false,
    //         .test_mode = "1_fabric_instance",
    //         .num_cluster_rows = 0,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_4_chip_multi_link_mcast_saturate_chip_to_chip_ring_bw
    // // @pytest.mark.parametrize("num_links", [2, 3, 4])
    // for (auto num_links : std::vector<size_t>{2, 3, 4}) {
    //     params.push_back({
    //         .test_name = "Fabric4ChipMultiLinkMcastSaturateChipToChipRingBw",
    //         .is_unicast = false,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = num_links,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 4,
    //         .packet_size = 4096,
    //         .fabric_mode = FabricTestMode::SaturateChipToChipRing,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = false,
    //         .test_mode = "1_fabric_instance",
    //         .num_cluster_rows = 0,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_t3k_4chip_cols_mcast_bw
    // // @pytest.mark.parametrize("num_links", [1, 2])
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // for (auto num_links : std::vector<size_t>{1, 2}) {
    //     for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //         params.push_back({
    //             .test_name = "FabricT3k4ChipColsMcastBw",
    //             .is_unicast = true, // Note: Original test has this as True despite being named mcast
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = num_links,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 2,
    //             .packet_size = packet_size,
    //             .fabric_mode = FabricTestMode::Linear,
    //             .disable_sends_for_interior_workers = false,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1D_fabric_on_mesh",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 4
    //         });
    //     }
    // }

    // // Test: test_fabric_t3k_4chip_rows_mcast_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     params.push_back({
    //         .test_name = "FabricT3k4ChipRowsMcastBw",
    //         .is_unicast = false,
    //         .noc_message_type = "noc_unicast_write",
    //         .num_messages = 200000,
    //         .num_links = 1,
    //         .num_op_invocations = 1,
    //         .line_sync = true,
    //         .line_size = 4,
    //         .packet_size = packet_size,
    //         .fabric_mode = FabricTestMode::Linear,
    //         .disable_sends_for_interior_workers = false,
    //         .unidirectional = false,
    //         .senders_are_unidirectional = true,
    //         .test_mode = "1D_fabric_on_mesh",
    //         .num_cluster_rows = 2,
    //         .num_cluster_cols = 0
    //     });
    // }

    // // Test: test_fabric_4chip_one_link_mcast_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "Fabric4ChipOneLinkMcastBw",
    //             .is_unicast = false,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 4,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = false,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_4chip_one_link_bidirectional_single_producer_mcast_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "Fabric4ChipOneLinkBidirectionalSingleProducerMcastBw",
    //             .is_unicast = false,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 4,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = true,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_4chip_one_link_unidirectional_single_producer_mcast_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "Fabric4ChipOneLinkUnidirectionalSingleProducerMcastBw",
    //             .is_unicast = false,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 4,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = true,
    //             .unidirectional = true,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_4chip_two_link_mcast_bw
    // // @pytest.mark.parametrize("num_links", [2, 3, 4])
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto num_links : std::vector<size_t>{2, 3, 4}) {
    //     for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //         for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear,
    //         FabricTestMode::RingAsLinear}) {
    //             params.push_back({
    //                 .test_name = "Fabric4ChipTwoLinkMcastBw",
    //                 .is_unicast = false,
    //                 .noc_message_type = "noc_unicast_write",
    //                 .num_messages = 200000,
    //                 .num_links = num_links,
    //                 .num_op_invocations = 1,
    //                 .line_sync = true,
    //                 .line_size = 4,
    //                 .packet_size = packet_size,
    //                 .fabric_mode = fabric_mode,
    //                 .disable_sends_for_interior_workers = false,
    //                 .unidirectional = false,
    //                 .senders_are_unidirectional = true,
    //                 .test_mode = "1_fabric_instance",
    //                 .num_cluster_rows = 0,
    //                 .num_cluster_cols = 0
    //             });
    //         }
    //     }
    // }

    // // Test: test_fabric_one_link_non_forwarding_unicast_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "FabricOneLinkNonForwardingUnicastBw",
    //             .is_unicast = true,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 2,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = false,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_two_link_non_forwarding_unicast_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "FabricTwoLinkNonForwardingUnicastBw",
    //             .is_unicast = true,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 2,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 2,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = false,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_one_link_forwarding_unicast_multiproducer_multihop_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "FabricOneLinkForwardingUnicastMultiproducerMultihopBw",
    //             .is_unicast = true,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 4,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = false,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_one_link_forwarding_unicast_single_producer_multihop_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "FabricOneLinkForwardingUnicastSingleProducerMultihopBw",
    //             .is_unicast = true,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 4,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = true,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_one_link_forwarding_unicast_unidirectional_single_producer_multihop_bw
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "FabricOneLinkForwardingUnicastUnidirectionalSingleProducerMultihopBw",
    //             .is_unicast = true,
    //             .noc_message_type = "noc_unicast_write",
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 4,
    //             .packet_size = packet_size,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = true,
    //             .unidirectional = true,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_one_link_forwarding_unicast_single_producer_multihop_atomic_inc_bw
    // // @pytest.mark.parametrize("noc_message_type", ["noc_unicast_flush_atomic_inc",
    // "noc_unicast_no_flush_atomic_inc"])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto noc_message_type : std::vector<std::string>{"noc_unicast_flush_atomic_inc",
    // "noc_unicast_no_flush_atomic_inc"}) {
    //     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear, FabricTestMode::RingAsLinear}) {
    //         params.push_back({
    //             .test_name = "FabricOneLinkForwardingUnicastSingleProducerMultihopAtomicIncBw",
    //             .is_unicast = true,
    //             .noc_message_type = noc_message_type,
    //             .num_messages = 200000,
    //             .num_links = 1,
    //             .num_op_invocations = 1,
    //             .line_sync = true,
    //             .line_size = 4,
    //             .packet_size = 16,
    //             .fabric_mode = fabric_mode,
    //             .disable_sends_for_interior_workers = true,
    //             .unidirectional = false,
    //             .senders_are_unidirectional = true,
    //             .test_mode = "1_fabric_instance",
    //             .num_cluster_rows = 0,
    //             .num_cluster_cols = 0
    //         });
    //     }
    // }

    // // Test: test_fabric_one_link_multihop_fused_write_atomic_inc_bw
    // // @pytest.mark.parametrize("is_unicast", [False, True])
    // // @pytest.mark.parametrize("disable_sends_for_interior_workers", [False, True])
    // // @pytest.mark.parametrize("packet_size", [16, 2048, 4096])
    // // @pytest.mark.parametrize("unidirectional", [False, True])
    // // @pytest.mark.parametrize("noc_message_type", ["noc_fused_unicast_write_flush_atomic_inc",
    // "noc_fused_unicast_write_no_flush_atomic_inc"])
    // // @pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
    // for (auto is_unicast : std::vector<bool>{false, true}) {
    //     for (auto disable_sends_for_interior_workers : std::vector<bool>{false, true}) {
    //         for (auto packet_size : std::vector<size_t>{16, 2048, 4096}) {
    //             for (auto unidirectional : std::vector<bool>{false, true}) {
    //                 for (auto noc_message_type : std::vector<std::string>{"noc_fused_unicast_write_flush_atomic_inc",
    //                 "noc_fused_unicast_write_no_flush_atomic_inc"}) {
    //                     for (auto fabric_mode : std::vector<FabricTestMode>{FabricTestMode::Linear,
    //                     FabricTestMode::RingAsLinear}) {
    //                         params.push_back({
    //                             .test_name = "FabricOneLinkMultihopFusedWriteAtomicIncBw",
    //                             .is_unicast = is_unicast,
    //                             .noc_message_type = noc_message_type,
    //                             .num_messages = 200000,
    //                             .num_links = 1,
    //                             .num_op_invocations = 1,
    //                             .line_sync = true,
    //                             .line_size = 4,
    //                             .packet_size = packet_size,
    //                             .fabric_mode = fabric_mode,
    //                             .disable_sends_for_interior_workers = disable_sends_for_interior_workers,
    //                             .unidirectional = unidirectional,
    //                             .senders_are_unidirectional = true,
    //                             .test_mode = "1_fabric_instance",
    //                             .num_cluster_rows = 0,
    //                             .num_cluster_cols = 0
    //                         });
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    return params;
}

// Instantiate the parameterized test with all generated parameters
INSTANTIATE_TEST_SUITE_P(
    FabricBandwidth,
    FabricBandwidthParameterizedTest,
    ::testing::ValuesIn(GenerateTestParams()),
    [](const ::testing::TestParamInfo<FabricBandwidthTestParams>& info) {
        std::string name = info.param.test_name + "_" + std::to_string(info.param.packet_size) + "B_" +
                           std::to_string(info.param.num_links) + "L";

        // Add distinguishing parameters to avoid duplicates
        if (info.param.line_size != 4) {
            name += "_LineSize" + std::to_string(info.param.line_size);
        }
        if (info.param.is_unicast) {
            name += "_Unicast";
        } else {
            name += "_Mcast";
        }
        if (info.param.unidirectional) {
            name += "_Unidirectional";
        }
        if (info.param.senders_are_unidirectional) {
            name += "_SendersUnidir";
        }
        if (info.param.disable_sends_for_interior_workers) {
            name += "_DisableInterior";
        }
        if (info.param.fabric_mode != FabricTestMode::HalfRing) {
            name += "_Mode" + std::to_string(static_cast<int>(info.param.fabric_mode));
        }
        if (info.param.test_mode != "1_fabric_instance") {
            name += "_" + info.param.test_mode;
        }
        if (info.param.num_cluster_rows > 0) {
            name += "_Rows" + std::to_string(info.param.num_cluster_rows);
        }
        if (info.param.num_cluster_cols > 0) {
            name += "_Cols" + std::to_string(info.param.num_cluster_cols);
        }

        // Add NOC message type to distinguish atomic inc tests
        if (info.param.noc_message_type != "noc_unicast_write") {
            std::string noc_suffix = info.param.noc_message_type;
            // Replace special characters for valid test names (must be valid C++ identifiers)
            // Convert to a more readable format
            if (noc_suffix == "noc_unicast_flush_atomic_inc") {
                noc_suffix = "FlushAtomicInc";
            } else if (noc_suffix == "noc_unicast_no_flush_atomic_inc") {
                noc_suffix = "NoFlushAtomicInc";
            } else if (noc_suffix == "noc_fused_unicast_write_flush_atomic_inc") {
                noc_suffix = "FusedFlushAtomicInc";
            } else if (noc_suffix == "noc_fused_unicast_write_no_flush_atomic_inc") {
                noc_suffix = "FusedNoFlushAtomicInc";
            } else if (noc_suffix == "noc_multicast_write") {
                noc_suffix = "MulticastWrite";
            } else {
                // Fallback: replace underscores and other special chars
                std::replace(noc_suffix.begin(), noc_suffix.end(), '_', '-');
                std::replace(noc_suffix.begin(), noc_suffix.end(), '-', '_');
            }
            name += "_" + noc_suffix;
        }

        return name;
    });
