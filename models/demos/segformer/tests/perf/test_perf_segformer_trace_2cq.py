# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn.torch_tracer
from loguru import logger
from transformers import SegformerForSemanticSegmentation

import ttnn
from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
    SegformerForSemanticSegmentationReference,
)
from models.demos.segformer.tests.perf.segformer_test_infra import SegformerTrace2CQ
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 1824800}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time",
    [
        [1, ttnn.bfloat16, ttnn.bfloat16, 65, 0.0125],
    ],
)
def test_perf_segformer_trace_2cq(
    device, batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time
):
    device.enable_program_cache()

    segformer_t2cq = SegformerTrace2CQ(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    )

    compile_finished = segformer_t2cq.compile()
    capture_finished = segformer_t2cq.trace_capture(compile_finished)
    segformer_t2cq.trace_execute(capture_finished)
    segformer_t2cq.validate_outputs()

    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    reference_model = SegformerForSemanticSegmentationReference(config=torch_model.config)
    reference_model.load_state_dict(torch_model.state_dict())
    reference_model.eval()

    input_shape = (1, 3, 512, 512)
    for iter in range(0, 10 - 1):
        input = torch.randn(input_shape, dtype=torch.float32)
        ref_logits = reference_model(input).logits
        input = torch.permute(input, (0, 2, 3, 1))
        ttnn_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn_output = segformer_t2cq.trace_execute_for_demo(capture_finished, ttnn_input)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

        h = w = int(math.sqrt(ttnn_output.shape[-1]))
        ttnn_final_output = torch.reshape(ttnn_output, (ttnn_output.shape[0], ttnn_output.shape[1], h, w))
        pcc_passed, pcc_message = assert_with_pcc(ref_logits, ttnn_final_output, 0.98)
        logger.info(pcc_message)

    prep_perf_report(
        model_name="segformer_e2e",
        batch_size=batch_size,
        inference_and_compile_time=segformer_t2cq.jit_time,
        inference_time=segformer_t2cq.inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="trace_2cq",
    )

    assert (
        segformer_t2cq.inference_time < expected_inference_time
    ), f"Segformer inference time {segformer_t2cq.inference_time} is too slow, expected {expected_inference_time}"
