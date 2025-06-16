# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=True):
    """
    Reference implementation of scaled dot-product attention.
    Args:
        query: (batch, seq_len_q, h_q, d_qk)
        key: (batch, seq_len_kv, h_kv, d_qk)
        value: (batch, seq_len_kv, h_kv, d_v)
        is_causal: whether to apply causal masking
    Returns:
        output: (batch, seq_len_q, h_q, d_v)
    """

    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)  # Repeat heads
    value = value.repeat_interleave(h_q // h_kv, dim=0)  # Repeat heads
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value


def flash_mla_decode_tt(
    query,
    key,
    value,
    h_q,
    h_kv,
    is_causal=True,
):
    pass


def run_flash_mla_decode_impl(
    device,
    batch,
    seq_len,
    h_q,
    h_kv,
    d_qk_nope,
    d_v,
    d_rope,
    q_dtype,
    dtype,
):
    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, h_q, 1, d_qk_nope + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, h_kv, seq_len, d_qk_nope + d_rope).float()  # (B, H, S, D)
    v = torch.randn(
        batch, h_kv, seq_len, d_v + d_rope
    ).float()  # (B, H, S, D) # TODO: REMOVE d_rope when validation is fixed!

    out_t = scaled_dot_product_attention(q, k, v, h_q, h_kv)

    ######################
    ### TT Setup
    #######################
    q_chunk_size = 32
    k_chunk_size = 32
    scale = (d_qk_nope + d_rope) ** -0.5

    max_start_idx = seq_len // 2
    start_indices = np.linspace(0, max_start_idx, batch, dtype=np.int32).tolist() if batch > 1 else [max_start_idx]

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(
        q.permute(2, 0, 1, 3),  # (B, H, S, D) -> (S, B, H, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        k,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v = ttnn.from_torch(
        v,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        cur_pos=start_indices,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    breakpoint()


@pytest.mark.parametrize(
    "batch, seq_len, h_q, h_kv, d_qk_nope, d_v, d_rope",
    # batch, seq_len, num heads q, num heads kv, dim q/k (nope), dim v, dim rope
    [
        (32, 1024, 16, 16, 64, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_decode(
    device,
    batch,
    seq_len,
    h_q,
    h_kv,
    d_qk_nope,
    d_v,
    d_rope,
    q_dtype,
    dtype,
    use_program_cache,
    function_level_defaults,
):
    run_flash_mla_decode_impl(
        device,
        batch,
        seq_len,
        h_q,
        h_kv,
        d_qk_nope,
        d_v,
        d_rope,
        q_dtype,
        dtype,
    )
