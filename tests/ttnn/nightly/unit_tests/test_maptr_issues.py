# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_maptr_div(device):
    numerator = torch.randn(2, 3200, 8, 1, 4, 2)
    denominator = torch.randn(1, 1, 1, 1, 1, 2)

    result = numerator / denominator

    ttnn_numerator = ttnn.from_torch(numerator, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_denominator = ttnn.from_torch(denominator, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_result = ttnn.div(ttnn_numerator, ttnn_denominator)
    ttnn_result = ttnn.to_torch(ttnn_result)

    assert_with_pcc(result, ttnn_result, 0.99)


def test_maptr_add(device):
    input_a = torch.randn(2, 3200, 1, 1, 1, 2)
    input_b = torch.randn(2, 3200, 8, 1, 4, 2)

    result = input_a + input_b

    ttnn_input_a = ttnn.from_torch(input_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_b = ttnn.from_torch(input_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_result = ttnn.add(ttnn_input_a, ttnn_input_b)
    ttnn_result = ttnn.to_torch(ttnn_result)

    assert_with_pcc(result, ttnn_result, 0.99)
