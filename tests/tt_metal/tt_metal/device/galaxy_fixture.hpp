// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device_pool.hpp>
#include "dispatch_fixture.hpp"

namespace tt::tt_metal {

class GalaxyFixture : public DispatchFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    bool SkipTestSuiteIfNotGalaxyMotherboard() {
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (!(this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices >= 32)) {
            return true;
        }
        return false;
    }

    void SetUp() override {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            GTEST_SKIP() << "Not a galaxy mobo";
        }
        DispatchFixture::SetUp();
    }

    void TearDown() override {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            return;
        }
        DispatchFixture::TearDownTestSuite();
    }

private:
    std::map<chip_id_t, IDevice*> device_ids_to_devices_;
};

class TGFixture : public GalaxyFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SkipTestSuiteIfNotTG() {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            GTEST_SKIP() << "Not a galaxy mobo";
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 32 && num_pcie_devices == 4)) {
            GTEST_SKIP() << "This test can only run on TG";
        }
    }
};

class TGGFixture : public GalaxyFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SkipTestSuiteIfNotTGG() {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            GTEST_SKIP() << "Not a galaxy mobo";
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 64 && num_pcie_devices == 8)) {
            GTEST_SKIP() << "This test can only run on TGG";
        }
    }
};

}  // namespace tt::tt_metal
