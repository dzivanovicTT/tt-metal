# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


def randomize_tensor(tensor_map, tensor_shape):
    tensor_shape = tuple(tensor_shape)
    if tensor_shape in tensor_map.keys():
        torch_tensor = tensor_map[tensor_shape]
    else:
        torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    return torch_tensor


def run_avg_pool2d(
    device,
    use_program_cache,
    tensor_map,
    input_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    divisor_override,
    count_include_pad,
    shard_scheme,
):
    ## Test setup for both.
    in_n, in_c, in_h, in_w = input_shape
    torch.manual_seed(0)
    torch_input = randomize_tensor(tensor_map, input_shape)

    ## Test setup for Actual.
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    ## Get Expected output.
    torch_output = torch.nn.functional.avg_pool2d(
        torch_input,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    ## Get Actual output
    ttnn_output = ttnn.avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        applied_shard_scheme=shard_scheme,
    )

    ## Test teardown for Actual.
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    )
    ttnn_output = ttnn.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W
    ttnn_output = ttnn.to_torch(ttnn_output)

    ## Assertion
    assert_with_pcc(torch_output, ttnn_output, 0.99)
    allclose = torch.allclose(ttnn_output, torch_output, rtol=0.02)
    assert allclose, " Reference and output tensor are not close"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # Normal reduction cases are when channels <= 8 * 32 and kernel_hw <= 16
        # Wide reduction cases channels > 8 * 32
        # Large reduction cases (channels < 32 and kernel_hw > 16) or (channels > 32 and kernel_hw > 32)
        [1, 288, 32, 32],
        [1, 320, 32, 32],
        [1, 544, 32, 32],
        [1, 576, 32, 32],
        [1, 800, 32, 32],
        [1, 832, 32, 32],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (3, 3),
        (9, 9),
    ),
)
@pytest.mark.parametrize(
    "use_program_cache",
    [False],
)
def test_run_avg_pool2d(
    device,
    use_program_cache,
    tensor_map,
    input_shape,
    kernel_size,
):
    run_avg_pool2d(
        device,
        use_program_cache,
        tensor_map,
        input_shape,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        ceil_mode=False,
        divisor_override=None,
        count_include_pad=False,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
