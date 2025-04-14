// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>

#include "pool2d_utils.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include "tt-metalium/constants.hpp"
#include "tt-metalium/hal.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"

using namespace tt;
namespace ttnn {
namespace operations::pool {
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

uint32_t calculate_L1_usage(
    const Tensor& input,
    const uint32_t kernel_h,
    const uint32_t kernel_w,
    const uint32_t out_h,
    const uint32_t out_w,
    const MemoryConfig& input_memory,
    const MemoryConfig& output_memory) {
    const auto input_shape = input.get_padded_shape();

    tt::DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat out_df = in_df;
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);

    auto pconfig = input.memory_config();
    auto grid_size = input.shard_spec().value().grid.bounding_box().grid_size();
    uint32_t num_shards_c = 0;
    if (pconfig.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_shards_c = 1;
    } else if (pconfig.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        num_shards_c = input.shard_spec().value().grid.num_cores();
    } else if (input.shard_spec().value().orientation == ShardOrientation::COL_MAJOR) {
        num_shards_c = grid_size.y;
    } else {
        num_shards_c = grid_size.x;
    }

    uint32_t in_nbytes_c = input_shape[3] / num_shards_c * in_nbytes;  // row of input (channels)
    uint32_t out_nbytes_c = out_w / num_shards_c * out_nbytes;         // row of output (channels)

    tt::DataFormat indices_df =
        tt::DataFormat::RawUInt16;  // datatype_to_dataformat_converter(reader_indices.get_dtype());
    uint32_t indices_nbytes = datum_size(indices_df);

    uint32_t kernel_size_hw = kernel_h * kernel_w;  // number of valid rows, to read
    uint32_t kernel_size_hw_padded = tt::round_up(kernel_size_hw, tt::constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t)std::ceil((float)kernel_size_hw_padded / tt::constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);
    uint32_t out_ntiles_c = (uint32_t)std::ceil((float)out_w / num_shards_c / tt::constants::TILE_WIDTH);

    uint32_t max_rows_for_reduction = 16;
    // TODO #14588: temporarily disabling 32 row reductions due to issues in large kernels
    /* uint32_t max_rows_for_reduction = tt::constants::TILE_HEIGHT;
    // For GRAYSKULL, make reduction for 16 rows at a time.
    if (device->arch() == tt::ARCH::GRAYSKULL)
        max_rows_for_reduction /= 2; */

    // Hardware can do reduction of 8 tiles at a time.
    // CB sizes can be restricted to this in case input channels are more than 256 to perform reduction iteratively.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    const bool is_large_kernel = kernel_size_hw > max_rows_for_reduction;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    uint32_t nblocks = 1;
    // TT_FATAL(nblocks == 1, "Multiple blocks not yet supported");

    uint32_t tile_w = tt::constants::TILE_WIDTH;
    if (input_shape[3] < tt::constants::TILE_WIDTH) {
        TT_FATAL(input_shape[3] == 16, "Error");
        tile_w = tt::constants::FACE_WIDTH;
    }
    uint32_t out_w_loop_count = std::ceil((float)out_w / nblocks);

    // distributing out_hw across the grid
    auto all_cores = input_memory.shard_spec.value().grid;
    uint32_t ncores = all_cores.num_cores();
    auto core_range = all_cores;
    uint32_t in_nhw_per_core = input_memory.shard_spec.value().shape[0];
    uint32_t out_nhw_per_core = output_memory.shard_spec.value().shape[0];

    // TODO: support generic nblocks
    TT_FATAL(
        out_nhw_per_core % nblocks == 0,
        "number of sticks per core ({}) should be divisible by nblocks ({})",
        out_nhw_per_core,
        nblocks);

    // CBs
    uint32_t multi_buffering_factor = 2;

    uint32_t split_reader = 1;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    uint32_t in_scalar_cb_config_size = in_scalar_cb_npages /*1*/ * in_scalar_cb_pagesize /*2048*/;

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    auto raw_in_cb_id = tt::CBIndex::c_2;
    uint32_t raw_in_cb_npages = input_memory.shard_spec.value().shape[0];
    uint32_t raw_in_cb_pagesize = in_nbytes_c;
    uint32_t raw_in_cb_config_size = raw_in_cb_npages /*100*/ * raw_in_cb_pagesize /*192*/;

    // reader indices
    uint32_t in_reader_indices_cb_pagesize =
        tt::round_up(out_nhw_per_core * indices_nbytes, 4);  // pagesize needs to be multiple of 4
    uint32_t in_reader_indices_cb_npages = 1;
    uint32_t in_reader_indices_cb_config_size =
        in_reader_indices_cb_npages /*1*/ * in_reader_indices_cb_pagesize /*72*/;

    uint32_t in_cb_sz = 0;
    uint32_t in_nblocks_c = 1;
    if (is_wide_reduction) {
        in_cb_sz = MAX_TILES_PER_REDUCTION * tt::constants::TILE_HW;
        in_nblocks_c = std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);
    } else {
        in_cb_sz = in_ntiles_c * tt::constants::TILE_HW;
    }

    uint32_t in_cb_page_padded = tt::round_up(
        in_cb_sz,
        tt::constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_padded;
    uint32_t in_cb_npages = multi_buffering_factor * nblocks;
    uint32_t in_cb_config_0_size = in_cb_npages /*2*/ * in_cb_pagesize /*6144*/;
    uint32_t in_cb_config_1_size = 0;

    if (split_reader) {
        in_cb_config_1_size = in_cb_npages /*2*/ * in_cb_pagesize /*6144*/;
    }

    // after reduction
    uint32_t out_cb_pagesize = std::min(tt::constants::TILE_WIDTH, output_memory.shard_spec.value().shape[1]) *
                               out_nbytes;  // there is just one row of channels after each reduction (or 1 block
                                            // of c if its greater than 8 tiles)
    uint32_t out_cb_npages = output_memory.shard_spec.value().shape[0] * in_ntiles_c;
    uint32_t out_cb_config_size = out_cb_npages /*108*/ * out_cb_pagesize /*62*/;

    uint32_t max_pool_partials_cb_config_size = 0;
    if (is_large_kernel) {
        uint32_t max_pool_partials_cb_pagesize = out_cb_pagesize;
        uint32_t max_pool_partials_cb_npages = nblocks;
        max_pool_partials_cb_config_size = max_pool_partials_cb_npages /*1*/ * max_pool_partials_cb_pagesize /*64*/;
    }

    return in_scalar_cb_config_size
           // + raw_in_cb_config_size
           // + in_reader_indices_cb_config_size
           + in_cb_config_0_size +
           in_cb_config_1_size
           // + out_cb_config_size
           + max_pool_partials_cb_config_size;
}

sliding_window::ParallelConfig determine_pool_config_for_auto_shard(
    const Tensor& input_tensor, const sliding_window::SlidingWindowConfig& sliding_window_config, uint32_t channels) {
    auto batch_size = sliding_window_config.batch_size;
    auto output_shape = sliding_window_config.get_output_shape();
    auto compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    auto input_parallel_config_height = conv::determine_parallel_config(
        TensorMemoryLayout::HEIGHT_SHARDED,
        batch_size,
        channels,
        output_shape[1],
        output_shape[2],
        channels,
        compute_grid_size,
        ShardOrientation::ROW_MAJOR,
        false,
        false);

    auto input_parallel_config_width = conv::determine_parallel_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        batch_size,
        channels,
        output_shape[1],
        output_shape[2],
        channels,
        compute_grid_size,
        ShardOrientation::ROW_MAJOR,
        false,
        false);
    auto output_parallel_config_width =
        conv::determine_output_parallel_config(input_parallel_config_width, compute_grid_size, channels, false);

    auto input_parallel_config_block = conv::determine_parallel_config(
        TensorMemoryLayout::BLOCK_SHARDED,
        batch_size,
        channels,
        output_shape[1],
        output_shape[2],
        channels,
        compute_grid_size,
        ShardOrientation::COL_MAJOR,
        false,
        false);

    auto get_memconfig = [&](const ParallelConfig& parallel_config) {
        uint32_t nhw = batch_size * output_shape[1] * output_shape[2];
        uint32_t out_channeal_padded = tt::round_up(
            channels, conv::get_num_cores_channels_from_parallel_config(parallel_config) * tt::constants::TILE_WIDTH);
        return conv::create_sharded_memory_config_from_parallel_config(
            ttnn::Shape({1, 1, nhw, out_channeal_padded}), parallel_config, tt::constants::TILE_HEIGHT);
    };

    uint32_t l1_usage_height = calculate_L1_usage(
        input_tensor,
        sliding_window_config.window_hw.first,
        sliding_window_config.window_hw.second,
        sliding_window_config.get_output_shape()[1],
        sliding_window_config.get_output_shape()[2],
        get_memconfig(input_parallel_config_height),
        get_memconfig(input_parallel_config_height));

    uint32_t l1_usage_width = calculate_L1_usage(
        input_tensor,
        sliding_window_config.window_hw.first,
        sliding_window_config.window_hw.second,
        sliding_window_config.get_output_shape()[1],
        sliding_window_config.get_output_shape()[2],
        get_memconfig(input_parallel_config_width),
        get_memconfig(output_parallel_config_width));

    uint32_t l1_usage_block = calculate_L1_usage(
        input_tensor,
        sliding_window_config.window_hw.first,
        sliding_window_config.window_hw.second,
        sliding_window_config.get_output_shape()[1],
        sliding_window_config.get_output_shape()[2],
        get_memconfig(input_parallel_config_block),
        get_memconfig(input_parallel_config_block));

    uint32_t ncores_height = input_parallel_config_height.grid.num_cores();
    uint32_t ncores_width = input_parallel_config_width.grid.num_cores();
    uint32_t ncores_block = input_parallel_config_block.grid.num_cores();

    uint32_t winning_l1_usage = l1_usage_height;
    auto winning_config = input_parallel_config_height;
    // Make sure that BS not only has smaller size but provides at least some slicing along the channels.
    // In case we have BS that would slice the tensor only along the HS conv2d code would fail later on.
    if (l1_usage_block < l1_usage_height && ncores_block > compute_grid_size.x) {
        winning_l1_usage = l1_usage_block;
        winning_config = input_parallel_config_block;
    }
    if (l1_usage_width < winning_l1_usage) {
        winning_config = input_parallel_config_width;
    }

    return winning_config;
}

}  // namespace operations::pool
}  // namespace ttnn
