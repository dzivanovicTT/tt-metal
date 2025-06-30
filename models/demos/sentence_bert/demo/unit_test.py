# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.sentence_bert.reference.sentence_bert import mean_pooling
from models.demos.sentence_bert.ttnn.ttnn_sentence_bert_model import ttnn_mean_pooling
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_torchsum(device):
    input_tensor = torch.randn((8, 384, 768), dtype=torch.float16)
    torch_out = torch.sum(input_tensor, 1)
    tt_in = ttnn.from_torch(
        input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_out = ttnn.sum(tt_in, 1)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(torch_out, ttnn_out, 1.0)
    # ttnn_out = ttnn.experimental.cumsum(tt_in, 1)[:, -1, :]


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


# working - emb(r 16 dram), att( t 16 dram)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_meanpool(device):
    token_embeddings = torch.randn((8, 384, 768), dtype=torch.bfloat16)
    attention_mask = torch.ones((8, 384))
    # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    tt_token_embeddings = ttnn.from_torch(
        token_embeddings,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_attention_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    l1 = mean_pooling(token_embeddings, attention_mask)
    l2 = ttnn_mean_pooling(tt_token_embeddings, tt_attention_mask)
    assert_with_pcc(l1, ttnn.to_torch(l2), 1.0)
    # p(tt_token_embeddings,"11")
    # p(tt_attention_mask,"22")
    # tt_attention_mask = ttnn.unsqueeze(tt_attention_mask,dim=-1)
    # tt_attention_mask_ex = ttnn.repeat(tt_attention_mask,[1, 1,768])
    # # tt_attention_mask_ex = ttnn.to_torch(tt_attention_mask_ex)
    # # assert_with_pcc(tt_attention_mask_ex,input_mask_expanded,1.0)

    # torch_mul =token_embeddings * input_mask_expanded#torch.sum(, 1)
    # ttnn_mul = tt_token_embeddings*tt_attention_mask_ex
    # assert_with_pcc(torch_mul,ttnn.to_torch(ttnn_mul),0.99)
    # torch_mul = torch.sum(torch_mul,1)
    # # torch op
    # print("input to sum",ttnn_mul.shape)
    # ttnn_mul = ttnn.experimental.cumsum(ttnn_mul, 1)[:,-1,:]
    # # ttnn_mul = torch.sum(ttnn.to_torch(ttnn_mul),1)
    # # ttnn_mul = ttnn.from_torch(ttnn_mul,dtype=ttnn.bfloat16,device=device)
    # assert_with_pcc(torch_mul,ttnn.to_torch(ttnn_mul),0.99)

    # # dn
    # # clamped = ttnn.sum(tt_attention_mask_ex,dim=1)
    # print("input to sum2",tt_attention_mask_ex.shape)
    # clamped= ttnn.experimental.cumsum(tt_attention_mask_ex, 1)[:,-1,:]
    # # clamped = torch.sum(ttnn.to_torch(tt_attention_mask_ex),1)
    # # clamped = ttnn.from_torch(clamped,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device)
    # clamped = ttnn.clamp(clamped, min=1e-9)
    # input_mask_expanded = input_mask_expanded.sum(1)
    # den = torch.clamp(input_mask_expanded, min=1e-9)
    # assert_with_pcc(den,ttnn.to_torch(clamped),1.0)

    # # res
    # torch_res = torch_mul/den
    # # den = ttnn.from_torch(den,device=device)
    # ttnn_mul = ttnn.to_torch(ttnn_mul)
    # clamped = ttnn.to_torch(clamped)
    # ttnn_res = ttnn_mul/clamped
    # print("shaeps",clamped.shape,ttnn_mul.shape)
    # # ttnn_res = ttnn.div(ttnn_mul,clamped)
    # assert_with_pcc(torch_res,ttnn_res,1.0)
    # torch_result = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # p(tt_attention_mask_ex,"expanded")

    #
    #
    # summ = ttnn.sum(summ,dim=1)
    # clamped = ttnn.clamp(summ1, min=1e-9)
    # ttnn_res = ttnn.div(summ,clamped)
    # print("torch out is",torch_result.shape)
    # ttnn_res = ttnn.to_torch(ttnn_res)
    # assert_with_pcc(ttnn_res,torch_result,1.0)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_torchdiv(device):
    input_tensor1 = torch.randn((8, 768), dtype=torch.float16)
    input_tensor2 = torch.randn((8, 768), dtype=torch.float16)
    # assert_with_pcc(input_tensor1, input_tensor2, 1.0)
    torch_out = input_tensor1 / input_tensor2

    tt_in1 = ttnn.from_torch(input_tensor1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_in2 = ttnn.from_torch(input_tensor2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    p(tt_in1, "1")
    p(tt_in2, "2")
    ttnn_out = ttnn.div(tt_in1, tt_in2)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(torch_out, ttnn_out, 1.0)
