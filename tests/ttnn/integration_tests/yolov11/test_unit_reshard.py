# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_reshard(device):  # check dtype
    input_tensor = torch.randn((1, 1, 1600, 64), dtype=torch.bfloat16)
    tt_input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    shard_spec = ttnn.ShardSpec(shard_grid, (25, 64), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, input_mem_config)
    p(tt_input_tensor, "1st sahrd")
    # reshard
    shard_grid2 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(1, 6)),
        }
    )
    shard_spec2 = ttnn.ShardSpec(shard_grid2, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config2 = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec2
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, input_mem_config2)
    tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.TILE_LAYOUT)
    p(tt_input_tensor, "2nd sahrd")

    input_tensor = torch.randn((1, 1, 400, 128), dtype=torch.bfloat16)
    tt_input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    shard_spec = ttnn.ShardSpec(shard_grid, (7, 128), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, input_mem_config)
    p(tt_input_tensor, "1st sahrd")
    # reshard
    shard_grid2 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(4, 1)),
        }
    )
    shard_spec2 = ttnn.ShardSpec(shard_grid2, (32, 128), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config2 = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec2
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, input_mem_config2)
    tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.TILE_LAYOUT)
    a, b = ttnn.split()
    tt_input_tensor2 = ttnn.slice(tt_input_tensor, (0, 0, 0, 0), (1, 1, 400, 128))
    # p(tt_input_tensor,"2nd sahrd")
    torch_tens = ttnn.to_torch(tt_input_tensor)
    print("shape1", torch_tens.shape)
    torch_tens2 = ttnn.to_torch(tt_input_tensor2)
    print("shape2", torch_tens2.shape)

    # p(tt_input_tensor2,"2nd sahrd")
