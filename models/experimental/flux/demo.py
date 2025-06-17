# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import os

import ttnn
from models.experimental.flux import FluxPipeline

import pytest

"""
def run(
    *,
    mesh_width: int,
    mesh_height: int,
    num_images_per_prompt: int,
    use_torch_encoder: bool,
) -> None:
    assert num_images_per_prompt % mesh_height == 0

    # ruff: noqa: T201
    print(f"mesh_width = {mesh_width}")
    print(f"mesh_height = {mesh_height}")
    print(f"num_images_per_prompt = {num_images_per_prompt}")
    print(f"use_torch_encoder = {use_torch_encoder}")

    if ttnn.get_num_devices() > 1:
        is_blackhole = ttnn.get_arch_name() == "blackhole"
        dispatch_core_axis = ttnn.DispatchCoreAxis.COL if is_blackhole else ttnn.DispatchCoreAxis.ROW
        dispatch_core_config = ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.ETH, dispatch_core_axis)
    else:
        dispatch_core_config = None

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(mesh_height, mesh_width),
        l1_small_size=8192,
        trace_region_size=15210496,
        dispatch_core_config=dispatch_core_config,
    )
    #for device in mesh_device.get_devices():  # commented out
    #    ttnn.enable_program_cache(device)

    #device.enable_async(True)  # noqa: FBT003 # commented out

    pipeline = FluxPipeline(
        checkpoint="black-forest-labs/FLUX.1-schnell",
        device=mesh_device,
        use_torch_encoder=use_torch_encoder,
    )

    pipeline.prepare(
        width=1024,
        height=1024,
        prompt_count=1,
        num_images_per_prompt=num_images_per_prompt,
    )

    prompt = "A luxury sports car."

    for iteration in itertools.count(start=1):
        new_prompt = input("Enter the input prompt, or q to exit: ")
        if no_prompt == False:
            if new_prompt:
                prompt = new_prompt
            if prompt == "q":
                break


        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            num_inference_steps=4,
            seed=iteration,
        )

        for i, image in enumerate(images, start=1):
            image.save(f"flux_1024_{i}.png")

"""


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_inference_steps,image_w,image_h,batch_size,mesh_width,t5_on_device",
    [
        (4, 1024, 1024, 1, 8, True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024, "trace_region_size": 15210496}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_flux_1_schnell(
    *,
    mesh_device: ttnn.MeshDevice,
    num_inference_steps,
    image_w,
    image_h,
    batch_size,
    mesh_width,
    t5_on_device,
    no_prompt,
    model_location_generator,
) -> None:
    device_count = ttnn.get_num_devices()
    mesh_height = batch_size
    mesh_width = mesh_width if mesh_width is not None else device_count // mesh_height

    pipeline = FluxPipeline(
        checkpoint="black-forest-labs/FLUX.1-schnell",
        device=mesh_device,
        use_torch_encoder=t5_on_device,
    )

    pipeline.prepare(
        width=image_w,
        height=image_h,
        prompt_count=1,
        num_images_per_prompt=mesh_height,
    )

    prompt = "A luxury sports car."

    if no_prompt:
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
        )
        for i, image in enumerate(images, start=1):
            image.save(f"flux_{image_w}_{i}.png")
    else:
        for iteration in itertools.count(start=1):
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt == "q":
                break

            images = pipeline(
                prompt_1=[prompt],
                prompt_2=[prompt],
                num_inference_steps=num_inference_steps,
                seed=iteration,
            )
            for i, image in enumerate(images, start=1):
                image.save(f"flux_{image_w}_{i}.png")
