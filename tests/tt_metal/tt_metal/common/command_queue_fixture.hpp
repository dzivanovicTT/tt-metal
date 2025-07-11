// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.h>
#include <cstdint>
#include "fabric_types.hpp"
#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"

namespace tt::tt_metal {

class CommandQueueFixture : public DispatchFixture {
protected:
    tt::tt_metal::IDevice* device_;

    // This test fixture closes/opens devices on each test
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_device();
    }

    void TearDown() override {
        if (!this->IsSlowDispatch()) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            return false;
        }
        return true;
    }

    void create_device(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const chip_id_t device_id = *tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().begin();
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        this->device_ =
            tt::tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_config);
    }
};

class CommandQueueEventFixture : public CommandQueueFixture {};

class CommandQueueBufferFixture : public CommandQueueFixture {};

class CommandQueueProgramFixture : public CommandQueueFixture {};

class CommandQueueTraceFixture : public CommandQueueFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevice(const size_t trace_region_size) { this->create_device(trace_region_size); }
};

class UnitMeshCommandQueueFixture : public DispatchFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device.reset();
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }

    void create_devices(std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const chip_id_t mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<chip_id_t> chip_ids;
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_ids, DEFAULT_L1_SMALL_SIZE, trace_region_size, 1, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            this->devices_.push_back(device);
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
};

class CommandQueueSingleCardFixture : virtual public DispatchFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override {
        if (!reserved_devices_.empty()) {
            tt::tt_metal::detail::CloseDevices(reserved_devices_);
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = false;
            return false;
        }
        return true;
    }

    void create_devices(const std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const chip_id_t mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<chip_id_t> chip_ids;
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        this->reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            chip_ids, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_config);
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip) {
            for (const auto& [id, device] : this->reserved_devices_) {
                this->devices_.push_back(device);
            }
        } else {
            for (const auto& chip_id : chip_ids) {
                this->devices_.push_back(this->reserved_devices_.at(chip_id));
            }
        }
    }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
};

class CommandQueueSingleCardBufferFixture : public CommandQueueSingleCardFixture {};

class CommandQueueSingleCardTraceFixture : virtual public CommandQueueSingleCardFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices(90000000);
    }
};

class CommandQueueSingleCardProgramFixture : virtual public CommandQueueSingleCardFixture {};

// Multi device command queue fixture. This fixture keeps the device open between test cases.
// If the device should open closed/reopned for each test case then override the SetUpTestSuite and TearDownTestSuite
// methods
class CommandQueueMultiDeviceFixture : public DispatchFixture {
private:
    inline static std::vector<tt::tt_metal::IDevice*> devices_internal;
    inline static std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_internal;
    inline static size_t num_devices_internal;

protected:
    static bool ShouldSkip() {
        if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
            return true;
        }
        if (tt::tt_metal::GetNumAvailableDevices() < 2) {
            return true;
        }

        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires fast dispatch and at least 2 devices"; }

    static void DoSetUpTestSuite(
        uint32_t num_cqs = 1,
        uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        uint32_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        if (ShouldSkip()) {
            return;
        }

        std::vector<chip_id_t> chip_ids;
        for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            chip_ids.push_back(id);
        }

        auto dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        if (num_cqs > 1 && arch == tt::ARCH::WORMHOLE_B0 && tt::tt_metal::GetNumAvailableDevices() != 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                log_warning(
                    tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
                dispatch_core_config = DispatchCoreType::ETH;
            }
        }

        reserved_devices_internal = tt::tt_metal::detail::CreateDevices(
            chip_ids, num_cqs, l1_small_size, trace_region_size, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices_internal) {
            devices_internal.push_back(device);
        }
    }

    static void DoTearDownTestSuite() {
        if (ShouldSkip()) {
            return;
        }
        tt::tt_metal::detail::CloseDevices(reserved_devices_internal);
    }

    static void SetUpTestSuite() { CommandQueueMultiDeviceFixture::DoSetUpTestSuite(); }

    static void TearDownTestSuite() { CommandQueueMultiDeviceFixture::DoTearDownTestSuite(); }

    void SetUp() override {
        if (ShouldSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }

        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = devices_internal.size();
        devices_ = devices_internal;
        reserved_devices_ = reserved_devices_internal;
    }

    void TearDown() override {
        slow_dispatch_ = false;
        devices_.clear();
        reserved_devices_.clear();
        num_devices_ = 0;
        arch_ = tt::ARCH::Invalid;
    }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
    size_t num_devices_;
};

class CommandQueueMultiDeviceProgramFixture : public CommandQueueMultiDeviceFixture {
public:
    static void SetUpTestSuite() {
        if (ShouldSkip()) {
            return;
        }
        CommandQueueMultiDeviceFixture::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (ShouldSkip()) {
            return;
        }
        CommandQueueMultiDeviceFixture::DoTearDownTestSuite();
    }
};

class CommandQueueMultiDeviceBufferFixture : public CommandQueueMultiDeviceFixture {
public:
    static void SetUpTestSuite() {
        if (ShouldSkip()) {
            return;
        }
        CommandQueueMultiDeviceFixture::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (ShouldSkip()) {
            return;
        }
        CommandQueueMultiDeviceFixture::DoTearDownTestSuite();
    }
};

class CommandQueueMultiDeviceOnFabricFixture : public CommandQueueMultiDeviceFixture,
                                               public ::testing::WithParamInterface<tt::tt_metal::FabricConfig> {
protected:
    // Multiple fabric configs so need to reset the devices for each test
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (CommandQueueMultiDeviceFixture::ShouldSkip()) {
            GTEST_SKIP() << CommandQueueMultiDeviceFixture::GetSkipMessage();
        }
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Dispatch on Fabric tests only applicable on Wormhole B0";
        }
        // Skip for TG as it's still being implemented
        if (tt::tt_metal::IsGalaxyCluster()) {
            GTEST_SKIP();
        }
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(true);
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_metal::detail::SetFabricConfig(GetParam(), FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        CommandQueueMultiDeviceFixture::DoSetUpTestSuite();
        CommandQueueMultiDeviceFixture::SetUp();

        if (::testing::Test::IsSkipped()) {
            tt::tt_metal::detail::SetFabricConfig(FabricConfig::DISABLED);
        }
    }

    void TearDown() override {
        CommandQueueMultiDeviceFixture::TearDown();
        CommandQueueMultiDeviceFixture::DoTearDownTestSuite();
        tt::tt_metal::detail::SetFabricConfig(FabricConfig::DISABLED);
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(false);
    }
};

}  // namespace tt::tt_metal
