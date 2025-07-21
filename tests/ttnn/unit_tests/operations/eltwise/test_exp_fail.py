# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


# failing case : tests/tt_eager/python_api_testing/unit_testing/misc/test_flash_multi_latent_attention_prefill.py::test_flash_mla_prefill[q_dtype0-dtype0-2-1024-128-1-512-64]
def test_exp_fail(device):
    torch_input = torch.arange(-5.0, 5.01, 0.01, dtype=torch.float32).to(torch.bfloat16)
    torch.set_printoptions(threshold=float("inf"), linewidth=200, precision=4)

    print("Input shape : ", torch_input.shape)
    # print("Input tensor : ",torch_input)

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    print("Torch tensor : ", golden, "\nTTNN tensor : ", result)
    assert_with_ulp(golden, result, 0)
