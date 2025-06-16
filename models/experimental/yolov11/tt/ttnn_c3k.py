# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov11.tt.common import Conv, deallocate_tensors, sharded_concat
from models.experimental.yolov11.tt.ttnn_bottleneck import Bottleneck


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class C3K:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)
        self.cv3 = Conv(device, parameter.cv3, conv_pt.cv3, reshard=True)
        self.k1 = Bottleneck(device, parameter.m[0], conv_pt.m[0])
        self.k2 = Bottleneck(device, parameter.m[1], conv_pt.m[1])

    def __call__(self, device, x):
        x1 = self.cv1(device, x)
        x2 = self.cv2(device, x)

        k1 = self.k1(device, x1)
        k2 = self.k2(device, k1)
        # x2 = ttnn.sharded_to_interleaved(x2, ttnn.L1_MEMORY_CONFIG)
        # k2 = ttnn.sharded_to_interleaved(k2, ttnn.L1_MEMORY_CONFIG)
        use_shard_concat = True
        if use_shard_concat:
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            # x2 = ttnn.to_dtype(x2, ttnn.bfloat16)
            k2 = ttnn.to_layout(k2, ttnn.ROW_MAJOR_LAYOUT)
            # k2 = ttnn.to_dtype(k2, ttnn.bfloat16)
            x = sharded_concat([k2, x2], to_interleaved=False)
        else:
            x = ttnn.concat((k2, x2), 3, memory_config=ttnn.L1_MEMORY_CONFIG)

        p(x, "after concat")
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 0))})
        shard_spec = ttnn.ShardSpec(shard_grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        print("mode is ", input_mem_config.shard_spec.mode)
        # print("physica; SAHRD SHAPE is ",input_mem_config.shard_spec.physical_shard_shape)
        input_mem_config.shard_spec.mode = ttnn.ShardMode.LOGICAL
        # {'keep_l1_aligned': 'false'; 'output_dtype': 'DataType::BFLOAT16'; 'output_mem_config': 'MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED;buffer_type=BufferType::L1;shard_spec=ShardSpec(grid={[(x=0;y=0) - (x=6;y=0)]};shape={32; 64};orientation=ShardOrientation::ROW_MAJOR;mode=ShardMode::LOGICAL;physical_shard_shape={32; 64});nd_shard_spec=std::nullopt;created_with_nd_shard_spec=0)'
        # p(x, "before")

        # x = ttnn.to_memory_config(x,input_mem_config)
        # x = ttnn.to_layout(x,ttnn.TILE_LAYOUT)
        # p(x, "after")
        x = self.cv3(device, x)
        deallocate_tensors(x1, x2, k1, k2)
        return x
