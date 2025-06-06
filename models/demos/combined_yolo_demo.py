# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time

import cv2
import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn

# YOLOv4 imports
from models.demos.yolov4.common import image_to_tensor, load_image, load_torch_model
from models.demos.yolov4.post_processing import load_class_names, plot_boxes_cv2, post_processing
from models.demos.yolov4.runner.runner import YOLOv4Runner
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters

# YOLOv9c imports
from models.demos.yolov9c.demo.demo_utils import load_coco_class_names
from models.demos.yolov9c.reference import yolov9c
from models.demos.yolov9c.tt import ttnn_yolov9c
from models.demos.yolov9c.tt.model_preprocessing import create_yolov9c_input_tensors, create_yolov9c_model_parameters
from models.experimental.yolo_evaluation.yolo_common_evaluation import save_yolo_predictions_by_model
from models.experimental.yolo_evaluation.yolo_evaluation_utils import LoadImages, postprocess, preprocess
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0, skip_for_grayskull


@run_for_wormhole_b0()
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source",
    [
        "models/sample_data/huggingface_cat_image.jpg",
        # "models/demos/yolov4/resources/giraffe_320.jpg",  # Uncomment to use YOLOv4's default image
    ],
)
@pytest.mark.parametrize(
    "yolov4_resolution",
    [
        (320, 320),
        # (640, 640),  # Uncomment to test different resolution
    ],
)
def test_combined_yolo_demo(
    device, source, yolov4_resolution, model_location_generator, use_program_cache, reset_seeds
):
    """
    Combined demo that runs both YOLOv4 and YOLOv9c models on the same image.
    First runs YOLOv4 inference, then YOLOv9c inference on the same original image.
    """
    disable_persistent_kernel_cache()
    torch.manual_seed(0)

    logger.info(f"Running combined YOLO demo on image: {source}")
    logger.info(f"YOLOv4 resolution: {yolov4_resolution}")

    # Create output directory
    save_dir = "models/demos/combined_yolo_demo/runs"
    os.makedirs(save_dir, exist_ok=True)

    # =============================================================================
    # YOLOv4 Inference
    # =============================================================================
    logger.info("Starting YOLOv4 inference...")

    # Load image for YOLOv4
    yolov4_img = load_image(source, yolov4_resolution)
    yolov4_torch_input = image_to_tensor(yolov4_img)

    # Prepare TTNN input for YOLOv4
    yolov4_input_tensor = yolov4_torch_input.permute(0, 2, 3, 1)
    yolov4_ttnn_input = ttnn.from_torch(yolov4_input_tensor, ttnn.bfloat16)

    # Load YOLOv4 model and create parameters
    yolov4_torch_model = load_torch_model(model_location_generator)
    yolov4_parameters = create_yolov4_model_parameters(
        yolov4_torch_model, yolov4_torch_input, yolov4_resolution, device
    )

    # Create YOLOv4 runner and run inference
    yolov4_runner = YOLOv4Runner(device, yolov4_parameters, yolov4_resolution)

    # Start total timing
    total_start_time = time.time()

    # Start YOLOv4 timing
    yolov4_start_time = time.time()
    yolov4_output = yolov4_runner.run(yolov4_ttnn_input)
    yolov4_end_time = time.time()
    yolov4_execution_time = yolov4_end_time - yolov4_start_time

    # Post-process YOLOv4 results
    conf_thresh = 0.3
    nms_thresh = 0.4
    yolov4_boxes = post_processing(yolov4_img, conf_thresh, nms_thresh, yolov4_output)

    # Load class names and save YOLOv4 results
    yolov4_namesfile = "models/demos/yolov4/resources/coco.names"
    yolov4_class_names = load_class_names(yolov4_namesfile)
    yolov4_cv_img = cv2.imread(source)
    yolov4_output_path = os.path.join(save_dir, "yolov4_prediction.jpg")
    plot_boxes_cv2(yolov4_cv_img, yolov4_boxes[0], yolov4_output_path, yolov4_class_names)

    logger.info(f"YOLOv4 inference completed. Results saved to: {yolov4_output_path}")
    logger.info(f"YOLOv4 execution time: {yolov4_execution_time:.4f} seconds")

    # =============================================================================
    # YOLOv9c Inference
    # =============================================================================
    logger.info("Starting YOLOv9c inference...")

    # Prepare YOLOv9c model
    torch_input, ttnn_input = create_yolov9c_input_tensors(device)

    # Load YOLOv9c model with ultralytics weights
    ultralytics_model = YOLO("yolov9c.pt")
    state_dict = ultralytics_model.state_dict()

    yolov9c_torch_model = yolov9c.YoloV9()
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(
        yolov9c_torch_model.state_dict().items(), ds_state_dict.items()
    ):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2

    yolov9c_torch_model.load_state_dict(new_state_dict)
    yolov9c_torch_model.eval()

    # Create YOLOv9c parameters and model
    yolov9c_parameters = create_yolov9c_model_parameters(yolov9c_torch_model, torch_input, device=device)
    yolov9c_model = ttnn_yolov9c.YoloV9(device, yolov9c_parameters)

    # Prepare image for YOLOv9c
    dataset = LoadImages(path=source)
    yolov9c_names = load_coco_class_names()
    yolov9c_save_dir = os.path.join(save_dir, "yolov9c")
    os.makedirs(yolov9c_save_dir, exist_ok=True)

    # Process image with YOLOv9c
    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=(640, 640))
        img = torch.permute(im, (0, 2, 3, 1))
        img = img.reshape(
            1,
            1,
            img.shape[0] * img.shape[1] * img.shape[2],
            img.shape[3],
        )
        yolov9c_ttnn_im = ttnn.from_torch(img, dtype=ttnn.bfloat16)

        # Run YOLOv9c inference
        yolov9c_start_time = time.time()
        yolov9c_preds = yolov9c_model(yolov9c_ttnn_im)
        yolov9c_end_time = time.time()
        yolov9c_execution_time = yolov9c_end_time - yolov9c_start_time
        total_execution_time = time.time() - total_start_time
        yolov9c_preds = ttnn.to_torch(yolov9c_preds, dtype=torch.float32)

        # Post-process YOLOv9c results
        yolov9c_results = postprocess(yolov9c_preds, im, im0s, batch, yolov9c_names)[0]
        save_yolo_predictions_by_model(yolov9c_results, save_dir, source, "yolov9c")

        break  # Process only the first (and only) image

    logger.info(f"YOLOv9c inference completed. Results saved to: {yolov9c_save_dir}")
    logger.info(f"YOLOv9c execution time: {yolov9c_execution_time:.4f} seconds")

    # =============================================================================
    # Summary
    # =============================================================================

    logger.info("Combined YOLO demo completed successfully!")
    logger.info(f"YOLOv4 results: {yolov4_output_path}")
    logger.info(f"YOLOv9c results: {yolov9c_save_dir}")
    logger.info(f"Both models processed the same input image: {source}")
    logger.info("=" * 60)
    logger.info("TIMING SUMMARY:")
    logger.info(f"YOLOv4 execution time:  {yolov4_execution_time:.4f} seconds")
    logger.info(f"YOLOv9c execution time: {yolov9c_execution_time:.4f} seconds")
    logger.info(f"Total execution time:   {total_execution_time:.4f} seconds")
    logger.info(f"YOLOv4 percentage:      {(yolov4_execution_time/total_execution_time)*100:.2f}%")
    logger.info(f"YOLOv9c percentage:     {(yolov9c_execution_time/total_execution_time)*100:.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    # This allows running the demo directly for debugging
    import ttnn

    device = ttnn.open_device(device_id=0)

    # Mock the required parameters for direct execution
    class MockLocationGenerator:
        def __call__(self):
            return "/home/ttuser/tt-metal/models/demos/yolov4/tests/pcc/yolov4.pth"  # Update this path as needed

    test_combined_yolo_demo(
        device=device,
        source="models/sample_data/huggingface_cat_image.jpg",
        yolov4_resolution=(320, 320),
        model_location_generator=MockLocationGenerator(),
        use_program_cache=True,
        reset_seeds=lambda: None,
    )

    ttnn.close_device(device)
