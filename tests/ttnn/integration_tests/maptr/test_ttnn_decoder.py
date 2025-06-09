# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import torch.nn as nn
import copy
import ttnn
from models.experimental.maptr.reference import decoder
from models.experimental.maptr.ttnn import ttnn_decoder
from models.experimental.maptr.ttnn.model_preprocessing import (
    create_maptr_model_parameters_decoder,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_decoder(
    device,
    reset_seeds,
    use_program_cache,
):
    weights_path = "models/experimental/maptr/maptr_weights_sd.pth"
    torch_model = decoder.MapTRDecoder(num_layers=2, embed_dim=256, num_heads=4)

    torch_dict = torch.load(weights_path)

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("pts_bbox_head.transformer.decoder"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    reg_branch = []
    for _ in range(2):
        reg_branch.append(nn.Linear(256, 256))
        reg_branch.append(nn.ReLU())
    reg_branch.append(nn.Linear(256, 2))
    reg_branch = nn.Sequential(*reg_branch)
    reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for i in range(2)])

    query = torch.rand(2000, 1, 256)
    key = None
    value = torch.rand(3200, 1, 256)
    query_pos = torch.rand(2000, 1, 256)
    reference_points = torch.rand(1, 2000, 2)
    spatial_shapes = torch.tensor([[80, 40]], dtype=torch.long)

    parameter = create_maptr_model_parameters_decoder(
        torch_model, [query, key, value, query_pos, reference_points, spatial_shapes, reg_branches]
    )

    output, reference_points = torch_model(
        query,
        key=key,
        value=value,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        reg_branches=reg_branches,
    )
