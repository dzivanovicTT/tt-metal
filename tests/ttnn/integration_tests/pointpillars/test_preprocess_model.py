import torch
from tests.ttnn.integration_tests.pointpillars.input_preprocess_utils import Compose
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
from mmengine.dataset.utils import pseudo_collate


def get_chunk_data(inputs: Iterable, chunk_size: int):
    """Get batch data from dataset.

    Args:
        inputs (Iterable): An iterable dataset.
        chunk_size (int): Equivalent to batch size.

    Yields:
        list: batch data.
    """
    inputs_iter = iter(inputs)
    # print("inputs",inputs)
    # print("inputs_iter",inputs_iter)
    # assert False
    while True:
        try:
            chunk_data = []
            for _ in range(chunk_size):
                processed_data = next(inputs_iter)
                chunk_data.append(processed_data)
            yield chunk_data
        except StopIteration:
            if chunk_data:
                yield chunk_data
            break


def preprocess(inputs: InputsType, batch_size: int = 1, pipeline=None, **kwargs):
    """Process the inputs into a model-feedable format.

    Customize your preprocess by overriding this method. Preprocess should
    return an iterable object, of which each item will be used as the
    input of ``model.test_step``.

    ``BaseInferencer.preprocess`` will return an iterable chunked data,
    which will be used in __call__ like this:

    .. code-block:: python

        def __call__(self, inputs, batch_size=1, **kwargs):
            chunked_data = self.preprocess(inputs, batch_size, **kwargs)
            for batch in chunked_data:
                preds = self.forward(batch, **kwargs)

    Args:
        inputs (InputsType): Inputs given by user.
        batch_size (int): batch size. Defaults to 1.

    Yields:
        Any: Data processed by the ``pipeline`` and ``collate_fn``.
    """
    # print("self.pipeline",self.pipeline)
    # print("inputs",inputs)
    # print("batch_size",batch_size)
    chunked_data = get_chunk_data(map(pipeline, inputs), batch_size)

    collate_fn = pseudo_collate

    # print("chunked_data",chunked_data) #added by me
    yield from map(collate_fn, chunked_data)


def test_preprocess_model():
    pipeline_cfg = [
        {
            "type": "LidarDet3DInferencerLoader",
            "coord_type": "LIDAR",
            "load_dim": 5,
            "use_dim": 5,
            "backend_args": None,
        },
        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 10, "test_mode": True, "backend_args": None},
        {
            "type": "MultiScaleFlipAug3D",
            "img_scale": (1333, 800),
            "pts_scale_ratio": 1,
            "flip": False,
            "transforms": [
                {
                    "type": "GlobalRotScaleTrans",
                    "rot_range": [0, 0],
                    "scale_ratio_range": [1.0, 1.0],
                    "translation_std": [0, 0, 0],
                },
                {"type": "RandomFlip3D"},
                {"type": "PointsRangeFilter", "point_cloud_range": [-50, -50, -5, 50, 50, 3]},
            ],
        },
        {"type": "Pack3DDetInputs", "keys": ["points"]},
    ]

    print("pipeline_cfg", pipeline_cfg)

    pipeline = Compose(pipeline_cfg)

    ori_inputs = [
        {
            "points": "/home/ubuntu/punith/tt-metal/models/experimental/functional_pointpillars/reference/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
        }
    ]
    batch_size = 1
    preprocess_kwargs = {}

    inputs = preprocess(ori_inputs, batch_size=batch_size, pipeline=pipeline, **preprocess_kwargs)

    print("inputs", inputs)

    for i in inputs:
        print("inputs", i)
