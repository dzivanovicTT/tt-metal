# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import ttnn
from models.demos.segformer.tt.ttnn_segformer_decode_head import TtSegformerDecodeHead
from models.demos.segformer.tt.ttnn_segformer_model import TtSegformerModel


@dataclass
class TtSemanticSegmenterOutput:
    loss: ttnn.bfloat16 = None
    logits: ttnn.bfloat16 = None
    hidden_states: ttnn.bfloat16 = None
    attentions: ttnn.bfloat16 = None


class TtSegformerForSemanticSegmentation:
    def __init__(self, config, parameters):
        super().__init__()
        self.segformer = TtSegformerModel(config, parameters=parameters.segformer)
        self.decode_head = TtSegformerDecodeHead(config, parameters=parameters.decode_head)
        self.config = config

    def __call__(
        self,
        device,
        pixel_values: ttnn.bfloat16,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: Optional[bool] = None,
        parameters=None,
    ) -> Union[Tuple, TtSemanticSegmenterOutput]:
        N, C, H, W = pixel_values.shape
        min_channels = 16  # Padding from image channels (3) to min channels (16)
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(pixel_values, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = pixel_values
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
        ttnn.deallocate(nchw)
        ttnn.deallocate(pixel_values)
        nhwc = ttnn.reallocate(nhwc)
        # pixel_values = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        outputs = self.segformer(
            device,
            nhwc,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
            parameters=parameters.segformer,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(device, encoder_hidden_states, parameters=parameters.decode_head)

        loss = None

        return TtSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
