#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
from mla import MultiHeadLatentAttention

import ttnn


def main():
    # ── 1) Reproducibility & device setup ────────────────────────────────────────
    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    # ── 2) MLA hyperparams (head_dim=32, no RoPE, no dropout) ────────────────────
    d_model, num_head = 256, 8
    d_embed, d_c, d_c1 = 256, 16, 16  # ← d_embed must match d_model
    d_rotate, dropout = 0, 0.0
    max_batch, max_len = 1, 10
    seq_len = 5

    mla = MultiHeadLatentAttention(
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
    mla.eval()

    # ── 3) Dummy prompt → CPU forward ─────────────────────────────────────────────
    prompt = torch.randn(max_batch, seq_len, d_model)
    out_cpu = mla(sequence=prompt, key_value_states=None, att_mask=None, use_cache=False, start_pos=0)
    print("► CPU MLA final output shape:", out_cpu.shape)  # → (1, 5, 256)

    # ── 4) Recompute Q, K, V exactly as forward did ─────────────────────────────
    with torch.no_grad():
        C_Q = mla.DQ_proj(prompt)  # [1,5,16]
        C_KV = mla.DKV_proj(prompt)  # [1,5,16]

        Q_state = mla.UQ_proj(C_Q).view(max_batch, seq_len, num_head, d_model // num_head)
        K_state = mla.UK_proj(C_KV).view(max_batch, seq_len, num_head, d_model // num_head)
        V_state = mla.UV_proj(C_KV).view(max_batch, seq_len, num_head, d_model // num_head)

        Q = Q_state.transpose(1, 2) * mla.scaler
        K = K_state.transpose(1, 2)
        V = V_state.transpose(1, 2)

        scores_cpu = torch.matmul(Q, K.transpose(-1, -2))
        probs_cpu = F.softmax(scores_cpu, dim=-1)
        att_out_cpu = torch.matmul(probs_cpu, V)  # [1,8,5,32]

    # ── 5) Pack Q/K/V onto Tenstorrent & run fused SDPA ─────────────────────────
    Q_bf = Q.to(torch.bfloat16)
    K_bf = K.to(torch.bfloat16)
    V_bf = V.to(torch.bfloat16)

    q_tt = ttnn.from_torch(Q_bf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_tt = ttnn.from_torch(K_bf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_tt = ttnn.from_torch(V_bf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=False, attn_mask=None)
    att_out_hw = ttnn.to_torch(out_tt)  # [1,8,5,32]

    # ── 6) Print HW output shape & compare ───────────────────────────────────────
    print("► HW SDPA output shape:", att_out_hw.shape)
    diff = (att_out_cpu - att_out_hw.float()).abs().max()
    print("► Max |CPU–HW| difference on raw att_output:", diff.item())


if __name__ == "__main__":
    main()
