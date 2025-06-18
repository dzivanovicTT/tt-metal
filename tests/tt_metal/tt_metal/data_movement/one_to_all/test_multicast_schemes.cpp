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

void test(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size) {
    bool is_multicast = true;
    bool is_linked = true;

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
        sub_grid_size);
}

uint32_t determine_max_grid_dimension(std::vector<IDevice*>& devices_) {
    uint32_t smaller_dimension = std::min(
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y);
    return (smaller_dimension - 1);
}

}  // namespace unit_tests::dm::core_to_all::multicast_schemes

/* ============================================================================================ */
/* ======================== SCHEME 1: SENDER IN THE MIDDLE OF THE GRID ======================== */
/* ============================================================================================ */

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastScheme1) {
    // Parameters
    uint32_t test_case_id = 100;

    uint32_t sub_grid_dimension_limit =
        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::determine_max_grid_dimension(devices_);

    CoreCoord mst_core_coord;
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size;

    for (uint32_t sub_grid_dimension = 2; sub_grid_dimension <= sub_grid_dimension_limit; sub_grid_dimension++) {
        sub_grid_size = {sub_grid_dimension, sub_grid_dimension};
        mst_core_coord = {sub_grid_dimension - 1, sub_grid_dimension - 1};

        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
            arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
    }

    tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
        arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ============================================================================================ */
/* ==================== SCHEME 2: SENDER IN THE LEFTMOST CORNER OF THE GRID =================== */
/* ============================================================================================ */

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastScheme2) {
    // Parameters
    uint32_t test_case_id = 100;

    uint32_t sub_grid_dimension_limit =
        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::determine_max_grid_dimension(devices_);

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size;

    for (uint32_t sub_grid_dimension = 2; sub_grid_dimension <= sub_grid_dimension_limit; sub_grid_dimension++) {
        sub_grid_size = {sub_grid_dimension, sub_grid_dimension};

        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
            arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
    }

    tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
        arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* =========================================================================================================== */
/* ==================== SCHEME 3: SENDER IN STARTING GRID COLUMN, NOT IN STARTING GRID ROW =================== */
/* =========================================================================================================== */

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastScheme3) {
    // Parameters
    uint32_t test_case_id = 100;

    uint32_t sub_grid_dimension_limit =
        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::determine_max_grid_dimension(devices_);

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {1, 0};
    CoreCoord sub_grid_size;

    for (uint32_t sub_grid_dimension = 2; sub_grid_dimension <= sub_grid_dimension_limit; sub_grid_dimension++) {
        sub_grid_size = {sub_grid_dimension, sub_grid_dimension};

        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
            arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
    }

    tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
        arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* =============================================================================================================== */
/* ==================== SCHEME 3: SENDER NOT IN STARTING GRID COLUMN, NOT IN STARTING GRID ROW =================== */
/* =============================================================================================================== */

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastScheme4) {
    // Parameters
    uint32_t test_case_id = 100;

    uint32_t sub_grid_dimension_limit =
        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::determine_max_grid_dimension(devices_);

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {1, 1};
    CoreCoord sub_grid_size;

    for (uint32_t sub_grid_dimension = 2; sub_grid_dimension <= sub_grid_dimension_limit; sub_grid_dimension++) {
        sub_grid_size = {sub_grid_dimension, sub_grid_dimension};

        tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
            arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
    }

    tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
        arch_, devices_, num_devices_, test_case_id, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

}  // namespace tt::tt_metal
