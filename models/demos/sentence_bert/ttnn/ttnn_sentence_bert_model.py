# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_embeddings import TtnnSentenceBertEmbeddings
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_encoder import TtnnSentenceBertEncoder
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_pooler import TtnnSentenceBertPooler


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


def ttnn_mean_pooling(ttnn_token_embeddings, ttnn_attention_mask, device=None):
    ttnn_token_embeddings = ttnn.sharded_to_interleaved(ttnn_token_embeddings, ttnn.L1_MEMORY_CONFIG)
    if ttnn_attention_mask.is_sharded():
        ttnn_attention_mask_interleaved = ttnn.sharded_to_interleaved(ttnn_attention_mask, ttnn.L1_MEMORY_CONFIG)
        ttnn_attention_mask_interleaved = ttnn.to_layout(ttnn_attention_mask_interleaved, ttnn.TILE_LAYOUT)
        ttnn.deallocate(ttnn_attention_mask)
    else:
        ttnn_attention_mask_interleaved = ttnn_attention_mask
    print("input to ttnn postporcess")
    p(ttnn_token_embeddings, "ttnn_token_embeddings")
    p(ttnn_attention_mask, "ttnn_attention_mask")
    ttnn_token_embeddings = ttnn.squeeze(ttnn_token_embeddings, dim=1)
    tt_input_mask_expanded = ttnn.unsqueeze(ttnn_attention_mask_interleaved, dim=-1)
    tt_input_mask_expanded = ttnn.repeat(tt_input_mask_expanded, [1, 1, ttnn_token_embeddings.shape[-1]])
    sum1 = ttnn.multiply(ttnn_token_embeddings, tt_input_mask_expanded, memory_config=ttnn.L1_MEMORY_CONFIG)
    sum1 = ttnn.from_torch(
        (torch.sum(ttnn.to_torch(sum1), 1)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    sum2 = ttnn.from_torch(
        (torch.sum(ttnn.to_torch(tt_input_mask_expanded), 1)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # sum1 = ttnn.sum(sum1, 1)
    # sum2 = ttnn.sum(tt_input_mask_expanded, 1)
    # sum1 = ttnn.experimental.cumsum(sum1, dim=1,dtype=ttnn.bfloat16)
    # sum1 = sum1[:, -1, :]
    # sum2 = ttnn.experimental.cumsum(tt_input_mask_expanded, dim=1,dtype=ttnn.bfloat16)
    # sum2 = sum2[:, -1, :]
    sum2 = ttnn.clamp(sum2, min=1e-9)
    result = ttnn.div(sum1, sum2)
    return result


class TtnnSentenceBertModel:
    def __init__(self, parameters, config):
        self.embeddings = TtnnSentenceBertEmbeddings(parameters.embeddings, config)
        self.encoder = TtnnSentenceBertEncoder(parameters.encoder, config)
        self.pooler = TtnnSentenceBertPooler(parameters.pooler)

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        attention_mask_2: ttnn.Tensor,
        token_type_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        apply_post_process=False,
        device=None,
    ):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, device=device)
        sequence_output = self.encoder(embedding_output, attention_mask, device=device)
        ttnn.deallocate(embedding_output)
        ttnn.DumpDeviceProfiler(device)
        if apply_post_process:
            sequence_output = ttnn_mean_pooling(sequence_output, attention_mask_2, device=device)
        ttnn.DumpDeviceProfiler(device)
        return (sequence_output,)
