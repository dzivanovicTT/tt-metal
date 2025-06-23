#pragma once

#include <type_traits>
#include "accessor/detail/array_wrapper.hpp"
#include "detail/dspec.h"
#include "detail/helpers.hpp"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "dataflow_api.h"
#endif

namespace nd_sharding {
using detail::ArrayDynamicWrapper;
using detail::ArrayStaticWrapperU16;
using detail::ArrayStaticWrapperU32;
template <size_t StartIdx, uint32_t Size>
using array_u32_cta_sequence_wrapper_t = detail::struct_cta_sequence_wrapper_t<ArrayStaticWrapperU32, StartIdx, Size>;
template <size_t StartIdx, uint32_t Size>
using array_packed_u16_cta_sequence_wrapper_t =
    detail::struct_cta_sequence_wrapper_packed_u16_from_u32_t<StartIdx, Size>;

template <typename DSpec>
struct ShardedAccessor {
private:
    using StaticDspec = detail::ConditionalStaticInstance<DSpec, DSpec::is_static>;
    detail::ConditionalField<!DSpec::is_static, DSpec> dspec_instance;

    mutable detail::ConditionalField<!DSpec::has_static_rank, uint32_t[detail::MAX_RANK]> _page_coord;
    const size_t bank_base_address;

    const uint32_t page_size;

public:
    template <typename DSpec_ = DSpec, std::enable_if_t<std::is_same_v<std::decay_t<DSpec_>, DSpec>, int> = 0>
    constexpr explicit ShardedAccessor(
        DSpec_&& dspec, const size_t bank_base_address_in, const uint32_t page_size_in = 0) :
        dspec_instance(std::forward<DSpec_>(dspec)), bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    template <typename DSpec_ = DSpec, std::enable_if_t<DSpec_::is_static, int> = 0>
    ShardedAccessor(const size_t bank_base_address_in = 0, uint32_t page_size_in = 0) :
        bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    constexpr auto& dspec() const {
        if constexpr (DSpec::is_static) {
            return StaticDspec::instance;
        } else {
            return dspec_instance.value;
        }
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t page_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        const auto [bank_id, bank_offset] = this->get_bank_and_offset(page_id);
        const auto& packed_xy_coords = dspec().packed_xy_coords();
        return NOC_XY_ADDR(
            DYNAMIC_NOC_X(noc, (packed_xy_coords[bank_id] >> 8) & 0xFF),
            DYNAMIC_NOC_Y(noc, packed_xy_coords[bank_id] & 0xFF),
            bank_base_address + bank_offset * page_size + offset);
    }

    FORCE_INLINE
    void noc_async_read_page(
        const uint32_t page_id, const uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_read(get_noc_addr(page_id, offset, noc), dest_addr, page_size, noc);
    }

    FORCE_INLINE
    void noc_async_write_page(
        const uint32_t page_id, const uint32_t src_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_write(src_addr, get_noc_addr(page_id, offset, noc), page_size, noc);
    }

    struct PageMapping {
        size_t bank_id;
        size_t bank_page_offset;
    };

    PageMapping get_bank_and_offset(uint32_t page_id) const {
        const auto& d = dspec();
        const auto& shape = d.tensor_shape();
        const uint32_t rank = d.rank();
        ASSERT(page_id < d.tensor_volume());
        typename DSpec::Shape page_coord;
        if constexpr (!DSpec::has_static_rank) {
            page_coord = typename DSpec::Shape(_page_coord.value, rank);
        }
        for (int i = rank - 1; i >= 0; --i) {
            const uint32_t dim = shape[i];
            page_coord[i] = page_id % dim;
            page_id /= dim;
        }
        return get_bank_and_offset(page_coord);
    }

    template <typename ArrType, std::enable_if_t<detail::has_subscript_operator_v<ArrType>, int> = 0>
    PageMapping get_bank_and_offset(const ArrType& page_coord) const {
        const auto& d = dspec();

        const auto& tensor_shape = d.tensor_shape();
        const auto& shard_shape = d.shard_shape();
        const auto& shard_grid_strides = d.shard_grid_strides();
        const auto& shard_strides = d.shard_strides();
        const size_t rank = d.rank();

        size_t flattened_shard_id = 0;
        size_t page_offset_within_shard = 0;

        for (size_t i = 0; i < rank; ++i) {
            const uint32_t coord = page_coord[i];
            const uint32_t shape = tensor_shape[i];
            ASSERT(coord < shape);

            const uint32_t shard_dim = shard_shape[i];
            const uint32_t stride_grid = shard_grid_strides[i];
            const uint32_t stride_local = shard_strides[i];

            const uint32_t q = coord / shard_dim;
            const uint32_t r = coord % shard_dim;

            flattened_shard_id += static_cast<size_t>(q) * stride_grid;
            page_offset_within_shard += static_cast<size_t>(r) * stride_local;
        }

        const size_t num_banks = d.num_banks();
        const size_t bank_id = flattened_shard_id % num_banks;
        const size_t bank_shard_id = flattened_shard_id / num_banks;
        const size_t bank_page_offset = bank_shard_id * d.shard_volume() + page_offset_within_shard;

        return {bank_id, bank_page_offset};
    }
};

template <size_t CTA_BASE, size_t CRTA_BASE>
FORCE_INLINE auto make_args() {
    return detail::ArgsOffsets<CTA_BASE, CRTA_BASE>();
}

template <size_t CTA_BASE>
FORCE_INLINE auto make_args(const size_t crta_base) {
    return detail::ArgsOffsets<CTA_BASE>(crta_base);
}

template <typename ArgsOffsetsT>
FORCE_INLINE auto make_sharded_accessor_from_args(
    const ArgsOffsetsT& args, const size_t bank_base_address_in, const uint32_t page_size_in) {
    auto dspec = detail::make_dspec_from_args(args);
    return ShardedAccessor<decltype(dspec)>(std::move(dspec), bank_base_address_in, page_size_in);
}

template <
    uint32_t RankCT = 0,
    uint32_t NumBanksCT = 0,
    typename TensorShapeWrapper_ = ArrayDynamicWrapper,
    typename ShardShapeWrapper_ = ArrayDynamicWrapper,
    typename BankCoordsWrapper_ = ArrayDynamicWrapper>
FORCE_INLINE auto make_dspec(
    uint32_t rank_rt = 0,
    uint32_t num_banks_rt = 0,
    uint32_t* tensor_shape_ptr = nullptr,
    uint32_t* shard_shape_ptr = nullptr,
    uint16_t* bank_coords_ptr = nullptr) {
    return detail::make_dspec<RankCT, NumBanksCT, TensorShapeWrapper_, ShardShapeWrapper_, BankCoordsWrapper_>(
        rank_rt, num_banks_rt, tensor_shape_ptr, shard_shape_ptr, bank_coords_ptr);
}

template <typename DSpec>
FORCE_INLINE auto make_sharded_accessor_from_dspec(
    DSpec&& dspec, const size_t bank_base_address_in, const uint32_t page_size_in) {
    return ShardedAccessor<std::decay_t<decltype(dspec)>>(
        std::forward<DSpec>(dspec), bank_base_address_in, page_size_in);
}

}  // namespace nd_sharding
