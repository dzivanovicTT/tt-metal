#pragma once

#include <cstddef>
#include <cstdint>

#include <hostdevcommon/sharded_accessor/arg_config.hpp>
#include "const.hpp"

namespace nd_sharding {
namespace detail {
using std::size_t;

template <size_t CTA_OFFSET, size_t CRTA_OFFSET = UNKNOWN>
struct ArgsOffsets {
    static constexpr auto args_config =
        ArgsConfig(static_cast<ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_OFFSET)));

    static constexpr bool rank_is_crta = args_config.test(ArgConfig::RankCRTA);
    static constexpr bool num_banks_is_crta = args_config.test(ArgConfig::NumBanksCRTA);
    static constexpr bool tensor_shape_is_crta = args_config.test(ArgConfig::TensorShapeCRTA);
    static constexpr bool shard_shape_is_crta = args_config.test(ArgConfig::ShardShapeCRTA);
    static constexpr bool bank_coords_is_crta = args_config.test(ArgConfig::BankCoordsCRTA);

    static_assert(
        !rank_is_crta || (rank_is_crta and tensor_shape_is_crta and shard_shape_is_crta),
        "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    static_assert(
        !num_banks_is_crta || (num_banks_is_crta and bank_coords_is_crta),
        "If num_banks is runtime, bank_coords must also be runtime");

    static constexpr uint32_t RankCTAOffset = CTA_OFFSET + 1;
    static constexpr uint32_t NumBanksCTAOffset = RankCTAOffset + (rank_is_crta ? 0 : 1);

    static constexpr uint32_t RankCT = [] {
        if constexpr (rank_is_crta) {
            return 0;
        } else {
            return get_compile_time_arg_val(RankCTAOffset);
        }
    }();
    static constexpr uint32_t NumBanksCT = [] {
        if constexpr (num_banks_is_crta) {
            return 0;
        } else {
            return get_compile_time_arg_val(NumBanksCTAOffset);
        }
    }();
    static constexpr uint32_t PhysicalNumBanksCT = (NumBanksCT + 1) / 2;

    static_assert(rank_is_crta || RankCT > 0, "Rank must be greater than 0!");
    static_assert(num_banks_is_crta || NumBanksCT > 0, "Number of banks must be greater than 0!");

    static constexpr uint32_t TensorShapeCTAOffset = NumBanksCTAOffset + (num_banks_is_crta ? 0 : 1);
    static constexpr uint32_t ShardShapeCTAOffset = TensorShapeCTAOffset + (tensor_shape_is_crta ? 0 : RankCT);
    static constexpr uint32_t BankCoordsCTAOffset = ShardShapeCTAOffset + (shard_shape_is_crta ? 0 : RankCT);

    static constexpr uint32_t NumArgsCT = [] {
        if constexpr (bank_coords_is_crta) {
            return BankCoordsCTAOffset - CTA_OFFSET;
        } else {
            return BankCoordsCTAOffset + PhysicalNumBanksCT - CTA_OFFSET;
        }
    }();

private:
    [[no_unique_address]] uint32_t crta_offset_rt_;

public:
    constexpr ArgsOffsets() {
        if constexpr (CRTA_OFFSET == UNKNOWN) {
            crta_offset_rt_ = 0;
        }
    }
    constexpr explicit ArgsOffsets(uint32_t crta_offset) {
        static_assert(CRTA_OFFSET == UNKNOWN, "Do not pass crta_offset when CRTA_OFFSET is known");
        crta_offset_rt_ = crta_offset;
    }
    constexpr uint32_t crta_offset() const {
        if constexpr (CRTA_OFFSET != UNKNOWN) {
            return CRTA_OFFSET;
        } else {
            return crta_offset_rt_;
        }
    }

    constexpr uint32_t rank_crta_offset() const { return crta_offset(); }
    constexpr uint32_t num_banks_crta_offset() const { return crta_offset() + rank_is_crta; }

    constexpr uint32_t get_rank() const {
        if constexpr (!rank_is_crta) {
            return RankCT;
        } else {
            return get_common_arg_val<uint32_t>(rank_crta_offset());
        }
    }

    constexpr uint32_t get_num_banks() const {
        if constexpr (!num_banks_is_crta) {
            return NumBanksCT;
        } else {
            return get_common_arg_val<uint32_t>(num_banks_crta_offset());
        }
    }

    constexpr uint32_t get_physical_num_banks() const { return (get_num_banks() + 1) / 2; }

    constexpr uint32_t tensor_shape_crta_offset() const { return num_banks_crta_offset() + num_banks_is_crta; }

    constexpr uint32_t shard_shape_crta_offset() const {
        return tensor_shape_crta_offset() + (tensor_shape_is_crta ? get_rank() : 0);
    }

    constexpr uint32_t bank_coords_crta_offset() const {
        return shard_shape_crta_offset() + (shard_shape_is_crta ? get_rank() : 0);
    }

    static constexpr uint32_t compile_time_args_skip() { return NumArgsCT; }
    constexpr uint32_t runtime_args_skip() const {
        return bank_coords_crta_offset() + (bank_coords_is_crta ? get_physical_num_banks() : 0) - crta_offset();
    }
};

}  // namespace detail
}  // namespace nd_sharding
