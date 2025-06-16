# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.yolov11.tt.common import Conv, deallocate_tensors, sharded_concat


class SPPF:
    def __init__(self, device, parameter, conv_pt):
        self.parameter = parameter
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2, reshard=True)

    def __call__(self, device, x):
        x = self.cv1(device, x)
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x
        m1 = ttnn.max_pool2d(
            x,
            batch_size=self.parameter.cv2.conv.batch_size,
            input_h=self.parameter.cv2.conv.input_height,
            input_w=self.parameter.cv2.conv.input_width,
            channels=self.parameter.cv2.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m2 = ttnn.max_pool2d(
            m1,
            batch_size=self.parameter.cv2.conv.batch_size,
            input_h=self.parameter.cv2.conv.input_height,
            input_w=self.parameter.cv2.conv.input_width,
            channels=self.parameter.cv2.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m3 = ttnn.max_pool2d(
            m2,
            batch_size=self.parameter.cv2.conv.batch_size,
            input_h=self.parameter.cv2.conv.input_height,
            input_w=self.parameter.cv2.conv.input_width,
            channels=self.parameter.cv2.conv.in_channels,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        use_sharded_concat = True
        if use_sharded_concat:
            y = sharded_concat([x1, m1, m2, m3], to_interleaved=False)
        else:
            y = ttnn.concat([x1, m1, m2, m3], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # print(y.memory_config())
        # shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(4,1))})
        # shard_spec = ttnn.ShardSpec(
        #     shard_grid,(32,512) , ttnn.ShardOrientation.ROW_MAJOR
        # )
        # input_mem_config = ttnn.MemoryConfig(
        #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        # )
        # input_mem_config.shard_spec.mode = ttnn.ShardMode.LOGICAL
        # print("inn",input_mem_config)
        # x = ttnn.to_memory_config(x,input_mem_config)
        # x = ttnn.to_layout(x,ttnn.TILE_LAYOUT)
        x = self.cv2(device, y)

        deallocate_tensors(x1, m1, m2, m3)
        return x
