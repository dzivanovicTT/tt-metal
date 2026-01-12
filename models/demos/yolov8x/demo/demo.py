# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os

import fiftyone
import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov8x.demo.demo_utils import (
    LoadImages,
    get_mesh_mappers,
    load_coco_class_names,
    postprocess,
    preprocess,
    save_yolo_predictions_by_model,
)
from models.demos.yolov8x.reference import yolov8x
from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner
from models.utility_functions import disable_persistent_kernel_cache


def run_yolov8x_demo(device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size_per_device = 1
    batch_size = batch_size_per_device * num_devices
    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8x.pt")
        torch_model = torch_model.model
        model = torch_model.eval()
    else:
        model = yolov8x.DetectionModel()

    if model_type == "tt_model":
        performant_runner = YOLOv8xPerformantRunner(
            device,
            batch_size,
            inputs_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            outputs_mesh_composer=outputs_mesh_composer,
        )

    save_dir = "models/demos/yolov8x/demo/runs"

    input_loc = os.path.abspath(input_loc)
    dataset = LoadImages(path=input_loc, batch=batch_size)

    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, _ = batch
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        preprocessed_im = []
        for i, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[i]}")
            processed = preprocess([img], res=res)
            preprocessed_im.append(processed)

        im = torch.cat(preprocessed_im, dim=0)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = performant_runner.run(im)
            preds = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=outputs_mesh_composer)
        results = postprocess(preds, im, im0s, paths, names)

        for result, image_path in zip(results, paths):
            save_yolo_predictions_by_model(result, save_dir, image_path, model_type)

    if model_type == "tt_model":
        performant_runner.release()

    logger.info("Inference done")


def run_yolov8x_demo_dataset(device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size_per_device = 1
    batch_size = batch_size_per_device * num_devices
    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8x.pt")
        torch_model = torch_model.model
        model = torch_model.eval()
    else:
        model = yolov8x.DetectionModel()

    if model_type == "tt_model":
        performant_runner = YOLOv8xPerformantRunner(
            device,
            batch_size,
            inputs_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            outputs_mesh_composer=outputs_mesh_composer,
        )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    data_set = LoadImages([sample["filepath"] for sample in dataset], batch=batch_size)
    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json", "r") as file:
        coco_info = json.load(file)
        class_names = coco_info["classes"]

    torch_images = []
    orig_images = []
    paths_images = []

    for batch in data_set:
        paths, im0s, _ = batch
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        paths_images.extend(paths)
        orig_images.extend(im0s)

        for i, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[i]}")
            tensor = preprocess([img], res)
            torch_images.append(tensor)

        if len(torch_images) >= batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    if model_type == "torch_model":
        preds = model(torch_input_tensor)
    else:
        preds = performant_runner.run(torch_input_tensor)
        preds = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=outputs_mesh_composer)

    conf_thresh = 0.3
    nms_thresh = 0.4

    names = load_coco_class_names()
    output_dir = f"models/demos/yolov8x/demo/runs/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    results = postprocess(preds, torch_input_tensor, orig_images, paths_images, names)
    save_dir = "models/demos/yolov8x/demo/runs"

    for result, image_path in zip(results, paths):
        save_yolo_predictions_by_model(result, save_dir, image_path, model_type)

    if model_type == "tt_model":
        performant_runner.release()

    logger.info("Inference done")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov8x/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo(device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res):
    run_yolov8x_demo(device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov8x/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_dp(mesh_device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res):
    run_yolov8x_demo(mesh_device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov8x/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_dataset(device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res):
    run_yolov8x_demo_dataset(device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov8x/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_dataset_dp(mesh_device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res):
    run_yolov8x_demo_dataset(
        mesh_device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res
    )
