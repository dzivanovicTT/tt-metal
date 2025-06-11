# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.ttnn_sentencebert_layer import TtnnSentenceBertLayer


class TtnnSentenceBertEncoder:
    def __init__(self, parameters, config):
        self.layers = {}
        for i in range(config.num_hidden_layers):
            self.layers[i] = TtnnSentenceBertLayer(parameters.layer[i], config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        if attention_mask.is_sharded():
            attention_mask_t = ttnn.sharded_to_interleaved(attention_mask, ttnn.L1_MEMORY_CONFIG)
            attention_mask_t = ttnn.to_layout(attention_mask_t, ttnn.TILE_LAYOUT)
        else:
            attention_mask_t = attention_mask
        for i in range(len(self.layers)):
            layer_outputs = self.layers[i](hidden_states, attention_mask_t, device=device)
            hidden_states = layer_outputs
        ttnn.deallocate(attention_mask)
        return hidden_states
