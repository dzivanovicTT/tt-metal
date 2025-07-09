# SPDX-FileCopyrightText: © 2025 Tenstorrent AU ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest
from datetime import datetime


import torch
import ttnn

from models.experimental.yolov6l.reference.yolov6l_utils import fuse_model
from models.experimental.yolov6l.demo.demo_utils import *
from models.utility_functions import run_for_wormhole_b0
from models.experimental.yolov6l.demo.demo_utils import (
    LoadImages,
    load_coco_class_names,
    preprocess,
    rescale,
    non_max_suppression,
)

# from models.experimental.yolov6l.demo.demo_utils_1 import rescale,plot_box_and_label,non_max_suppression,generate_colors

sys.path.append("models/experimental/yolov6l/reference/")


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 255, 0)

    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    names = result["names"]

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {score.item():.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"prediction_{timestamp}.jpg"
    output_path = os.path.join(model_save_dir, output_name)

    cv2.imwrite(output_path, image)

    logger.info(f"Predictions saved to {output_path}")


from models.utility_functions import disable_persistent_kernel_cache


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "source",
    [
        "models/experimental/yolov6l/demo/images/bus.jpg",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        "torch_model",
    ],
)
def test_yolov6l_demo(device, source, model_type, reset_seeds):
    disable_persistent_kernel_cache()
    weights = "tests/ttnn/integration_tests/yolov6l/yolov6l.pt"
    ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = fuse_model(model).eval()
    stride = int(model.stride.max())

    save_dir = "models/experimental/yolov6l/demo/runs"

    dataset = LoadImages(path=source)

    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    names = load_coco_class_names()
    if model_type == "tt_model":
        from models.experimental.yolov6l.runner.performant_runner import YOLOv6lPerformantRunner

        performant_runner = YOLOv6lPerformantRunner(
            device,
            1,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            resolution=(640, 640),
            model_location_generator=None,
        )
        performant_runner._capture_yolov6l_trace_2cqs()

    for batch in dataset:
        paths, im0s, s = batch
        img_src = im0s[0]

        im = preprocess(im0s, res=(640, 640))

        if model_type == "torch_model":
            preds, _ = model(im)
            print("preds", preds.shape)
        else:
            import torch.nn.functional as F

            permute_im = im.permute(0, 2, 3, 1)
            permute_im = F.pad(permute_im, (0, 29), mode="constant", value=0)
            permute_im = ttnn.from_torch(permute_im, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            preds = performant_runner._execute_yolov6l_trace_2cqs_inference(permute_im)
            preds = ttnn.to_torch(preds, dtype=torch.float32)
            print("preds", preds.shape)

        conf_thres = 0.4
        max_det = 1000
        det = non_max_suppression(preds, conf_thres, max_det=max_det)[0]
        if model_type == "tt_model":
            performant_runner.release()

        results = []
        if len(det):
            det[:, :4] = rescale(im.shape[2:], det[:, :4], img_src.shape).round()
            results.append(Results(im0s, path=source, names=names, boxes=det))

        save_yolo_predictions_by_model(results[0], save_dir, source, model_type)
