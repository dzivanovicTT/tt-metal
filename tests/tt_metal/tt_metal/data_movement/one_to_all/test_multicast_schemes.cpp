// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include "test_one_to_all.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_all::multicast_schemes {

enum MulticastSchemeType {
    SenderInGridMiddle = 1,
    SenderInGridBottomLeftCorner,
    SenderInGridStartingColumnNotRow,
    SenderNotInGridStartingColumnOrRow,
    num_multicast_scheme_types = SenderNotInGridStartingColumnOrRow
};

uint32_t determine_max_grid_dimension(std::vector<IDevice*>& devices_) {
    uint32_t smaller_dimension = std::min(
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y);
    return (smaller_dimension - 1);
}

void test(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    uint32_t sub_grid_dimension_size,
    MulticastSchemeType multicast_scheme_type) {  // Use enum type here
    bool is_multicast = true;
    bool is_linked = true;
    CoreCoord sub_grid_size = {sub_grid_dimension_size, sub_grid_dimension_size};

    CoreCoord mst_core_coord;
    CoreCoord sub_start_core_coord;
    switch (multicast_scheme_type) {
        case SenderInGridMiddle:
            // Scheme 1: Sender in the middle of the grid
            mst_core_coord = {sub_grid_dimension_size - 1, sub_grid_dimension_size - 1};
            sub_start_core_coord = {0, 0};
            break;
        case SenderInGridBottomLeftCorner:
            // Scheme 2: Sender in the leftmost corner of the grid
            mst_core_coord = {0, 0};
            sub_start_core_coord = {0, 0};
            break;
        case SenderInGridStartingColumnNotRow:
            // Scheme 3: Sender in starting grid column, not in starting grid row
            mst_core_coord = {0, 0};
            sub_start_core_coord = {1, 0};
            break;
        case SenderNotInGridStartingColumnOrRow:
            // Scheme 4: Sender not in starting grid column, not in starting grid row
            mst_core_coord = {0, 0};
            sub_start_core_coord = {1, 1};
            break;
        default: throw std::invalid_argument("Invalid multicast scheme type");
    }

    // Run the directed ideal test
    tt::tt_metal::unit_tests::dm::core_to_all::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        multicast_scheme_type);
}

}  // namespace unit_tests::dm::core_to_all::multicast_schemes

/* ============================================================= */
/* ======================== ALL SCHEMES ======================== */
/* ============================================================= */

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastSchemes) {
    // Parameters
    uint32_t test_case_id = 100;  // Arbitrary test id

    uint32_t sub_grid_dimension_limit =
        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::determine_max_grid_dimension(devices_);

    for (uint32_t multicast_scheme_type =
             unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType::SenderInGridMiddle;
         multicast_scheme_type <=
         unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType::num_multicast_scheme_types;
         multicast_scheme_type++) {  // Use enum values in the loop
        for (uint32_t sub_grid_dimension_size = 2; sub_grid_dimension_size <= sub_grid_dimension_limit;
             sub_grid_dimension_size++) {
            tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
                arch_,
                devices_,
                num_devices_,
                test_case_id,
                sub_grid_dimension_size,
                static_cast<unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType>(
                    multicast_scheme_type));  // Cast to enum
        }
    }
}

}  // namespace tt::tt_metal

// CONSIDER: Maybe have all these in a single test case with a nested loop so that the python script can plot it
// directly
