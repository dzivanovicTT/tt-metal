# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov11.tt.common import Conv, deallocate_tensors, sharded_concat
from models.experimental.yolov11.tt.ttnn_c3k import C3K
from models.experimental.yolov11.tt.ttnn_bottleneck import Bottleneck


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class C3k2:
    def __init__(self, device, parameter, conv_pt, is_bk_enabled=False, reshard=False):
        self.is_bk_enabled = is_bk_enabled
        self.parameter = parameter

        if is_bk_enabled:
            self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1, reshard=reshard)
            self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2, reshard=True)
            self.k = Bottleneck(device, parameter[0], conv_pt.m[0])
        else:
            self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1, reshard=True)
            self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2, reshard=True)
            self.c3k = C3K(device, parameter[0], conv_pt.m[0])

    def __call__(self, device, x):
        x = self.cv1(device, x)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)  # to support slice
        p(x, "1st conv out")
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        p(x, "1st conv out lchange")
        y1 = x[:, :, :, : x.shape[-1] // 2]
        y2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        if self.is_bk_enabled:
            p(y2, "inputttt")
            y3 = self.k(device, y2)
            # y3 = ttnn.sharded_to_interleaved(y3, ttnn.L1_MEMORY_CONFIG)  #
        else:
            y3 = self.c3k(device, y2)

        if y2.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            y2 = ttnn.to_layout(y2, ttnn.ROW_MAJOR_LAYOUT)
        if y3.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            y3 = ttnn.to_layout(y3, ttnn.ROW_MAJOR_LAYOUT)
        use_shard_concat = True
        if use_shard_concat:
            to_interleaved = True if y1.shape[2] == 25600 else False
            x = sharded_concat([y1, y2, y3], to_interleaved=to_interleaved)
        else:
            y3 = ttnn.sharded_to_interleaved(y3, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.concat((y1, y2, y3), 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        # reshard and check if padding happens, if it's the case, slice in interleaved

        p(x, "finallll")

        x = self.cv2(device, x)

        deallocate_tensors(y1, y2, y3)
        return x
