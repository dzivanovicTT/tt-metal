# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.yolov11.tt.common import Conv


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


import ttnn


class Bottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = Conv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = Conv(device, parameter.cv2, conv_pt.cv2)

    def __call__(self, device, x):
        input = x
        x = self.cv1(device, x)
        x = self.cv2(device, x)
        p(x, "11")
        p(input, "22")
        if x.shape[2] == 25600:  # tile inputs are needed
            input = ttnn.to_layout(input, layout=ttnn.TILE_LAYOUT)
            x = ttnn.add(input, x, memory_config=x.memory_config())
        else:
            x = ttnn.add(input, x, memory_config=x.memory_config(), use_legacy=False)
        p(x, "after addddd")
        return x
