#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
from mla import MultiHeadLatentAttention

import ttnn


def main():
    # 1) Seeds & device
    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    # 2) Hyperparams
    d_model, num_head = 256, 8
    d_embed, d_c, d_c1 = 256, 16, 16
    d_rotate, dropout = 0, 0.0
    max_batch, max_len = 1, 10
    seq_len = 5

    # 3) MLA in BF16
    mla = (
        MultiHeadLatentAttention(
            d_model=d_model,
            num_head=num_head,
            d_embed=d_embed,
            d_c=d_c,
            d_c1=d_c1,
            d_rotate=d_rotate,
            dropout=dropout,
            bias=True,
            max_batch_size=max_batch,
            max_seq_len=max_len,
        )
        .to(torch.bfloat16)
        .eval()
    )

    # 4) Dummy prompt in BF16
    prompt = torch.randn(max_batch, seq_len, d_model).to(torch.bfloat16)

    # 5) CPU MLA forward → final output (not used below, just shape)
    out_cpu_bf = mla(sequence=prompt, key_value_states=None, att_mask=None, use_cache=False, start_pos=0)
    print("► CPU MLA final output (BF16) shape:", out_cpu_bf.shape)

    # 6) Re-extract raw Q/K/V & compute CPU raw attention in BF16
    with torch.no_grad():
        C_Q = mla.DQ_proj(prompt)
        C_KV = mla.DKV_proj(prompt)

        Qs = mla.UQ_proj(C_Q).view(max_batch, seq_len, num_head, d_model // num_head)
        Ks = mla.UK_proj(C_KV).view(max_batch, seq_len, num_head, d_model // num_head)
        Vs = mla.UV_proj(C_KV).view(max_batch, seq_len, num_head, d_model // num_head)

        Q = Qs.transpose(1, 2) * mla.scaler
        K = Ks.transpose(1, 2)
        V = Vs.transpose(1, 2)

        scores_bf = torch.matmul(Q, K.transpose(-1, -2))
        probs_bf = F.softmax(scores_bf, dim=-1)
        att_out_cpu = torch.matmul(probs_bf, V)  # [1,8,5,32]

    # 7) Hardware SDPA
    q_tt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_tt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_tt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=False, attn_mask=None)
    att_out_hw = ttnn.to_torch(out_tt)  # [1,8,5,32]

    # 8) Print both tensors and compare
    print("\n----- Raw CPU attention output (BF16) -----")
    print(att_out_cpu)
    print("dtype:", att_out_cpu.dtype, "shape:", att_out_cpu.shape)

    print("\n----- Raw HW attention output (BF16) -----")
    print(att_out_hw)
    print("dtype:", att_out_hw.dtype, "shape:", att_out_hw.shape)

    diff = (att_out_cpu - att_out_hw).abs().max()
    print(f"\n► Max |CPU–HW| difference on raw att_output: {diff.item()}")


if __name__ == "__main__":
    main()
