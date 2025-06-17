# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import itertools
import pytest
import math
from typing import Optional, Tuple, List

from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d, mesh_device_fixture

import ttnn

parameters = {
    "max_pool2d_large_kernel": {
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 16, 128, 128, 16, 16, 1, 1, 0, 0, 1, 1, False],
            [1, 16, 128, 128, 18, 18, 1, 1, 0, 0, 1, 1, False],
            [1, 32, 128, 128, 32, 32, 1, 1, 0, 0, 1, 1, False],
            [1, 32, 128, 128, 34, 34, 1, 1, 0, 0, 1, 1, False],
        ],
    },
}


@pytest.mark.parametrize("input_spec", parameters["max_pool2d_large_kernel"]["input_specs"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )
