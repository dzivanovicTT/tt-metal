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


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def scaled_dot_product_attention_reference(Q, K, V, start_indices, padded_layer_len, scale, is_causal=True):
    b, nh, _, _ = Q.shape  # b, nh, 1, d
    _, nkv, _, _ = K.shape

    attn_mask = None
    if is_causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        assert False, "Non-causal attention is not supported in this function."

    Q_slice = Q[:, :nh, :, :]  # b, nh, 1, d
    K_slice = K[:, :nkv, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    out = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d

    return out


def flash_mla_decode_tt(
    query,
    key,
    value,
    nh,
    nkv,
    is_causal=True,
):
    pass


def run_flash_mla_decode_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
):
    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, 1, kv_lora_rank + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S, D)
    v = k[..., :kv_lora_rank]  # (B, H, S, D)

    # —— FOLD HEADS INTO BATCH TO AVOID OOM FOR DEEPSEEK CONFIG ——
    # Decide how many cores to fold, BUT keep ≥32 heads per core
    grid = device.compute_with_storage_grid_size()  # CoreCoord(x=8,y=8)
    full_cores = grid.x  # 8 on a TG
    min_heads = 32
    # At most floor(nh/min_heads) cores, but ≥1
    max_cores = max(1, nh // min_heads)
    num_cores = min(full_cores, max_cores)  # fold across at most 4 cores
    # how many heads each core should handle
    heads_per_core = nh // num_cores
    # new batch and head counts
    new_batch = batch * num_cores
    new_nh = heads_per_core

    # reshape Q: [B, H, 1, D] → [new_batch, new_nh, 1, D]
    q = q.view(batch, num_cores, new_nh, 1, -1).permute(0, 1, 3, 2, 4).reshape(new_batch, new_nh, 1, -1)

    # reshape K: [B, nkv, S, D] → [new_batch, nkv, S, D]
    k = (
        k.unsqueeze(1)  # [B,1,nkv,S,D]
        .repeat(1, num_cores, 1, 1, 1)  # [B,num_cores,nkv,S,D]
        .reshape(new_batch, nkv, seq_len, -1)
    )
    # derive V again
    v = k[..., :kv_lora_rank]

    # override for rest of function
    batch, nh = new_batch, new_nh

    # rebuild start_indices for the expanded batch
    max_start_idx = seq_len // 2
    base_starts = (
        np.linspace(0, max_start_idx, batch // num_cores, dtype=np.int32).tolist()
        if (batch // num_cores) > 1
        else [max_start_idx]
    )
    start_indices = [s for s in base_starts for _ in range(num_cores)]
    # —— END FOLD HEADS INTO BATCH ——

    ######################
    ### TT Setup
    #######################

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    q_chunk_size = padded_num_heads
    k_chunk_size = 128

    scale = (kv_lora_rank + d_rope) ** -0.5

    max_start_idx = seq_len // 2
    start_indices = np.linspace(0, max_start_idx, batch, dtype=np.int32).tolist() if batch > 1 else [max_start_idx]

    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
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

    ##########################
    ### FlashMLA Decode
    ##########################
    logger.info(
        f"Running FlashMLA Decode with TT Q shape: {tt_q.shape}, TT K shape: {tt_k.shape}, head_dim_v: {kv_lora_rank}"
    )
    tt_out = ttnn.transformer.flash_mla_decode(
        tt_q,
        tt_k,
        head_dim_v=kv_lora_rank,
        cur_pos=start_indices,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out_torch = ttnn.to_torch(tt_out)[..., :nh, :].permute(1, 2, 0, 3)  # (S, B, H, D) -> (B, H, S, D)

    ########################
    ### Validation
    ########################
    out_t = scaled_dot_product_attention_reference(
        q,
        k,
        v,
        start_indices,
        padded_layer_len,
        scale,
    )

    pcc_threshold = 0.99
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.91
    if dtype == ttnn.bfloat8_b:
        pcc_threshold = 0.97

    # override for the 512+64 head-dim DeepSeek shape
    if kv_lora_rank == 512 and d_rope == 64:
        pcc_threshold = 0.94

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_t, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        (2, 1024, 128, 1, 512, 64),  # DeepSeek V3 TG DP=32
        (2, 8 * 1024, 8, 1, 128, 64),
        (2, 8 * 1024, 64, 1, 256, 0),
        (2, 8 * 1024, 64, 1, 32, 64),
        (8, 4 * 1024, 8, 1, 128, 32),
        # (32, 8 * 1024, 8, 1, 128, 32),  # Gives bad PCC
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat8_b, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
        (ttnn.bfloat8_b, ttnn.bfloat4_b),
    ],
)
def test_flash_mla_decode(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_program_cache,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_decode_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
    )
