# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.scripts.generate_test_inputs_outputs import __file__ as REFERENCE_IO_SCRIPT_NAME
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_helpers import SEQ_LEN_CHUNK_SIZE
from models.utility_functions import comp_pcc


def load_state_dict(model_path: Path, module_path: str):
    if not not module_path:
        module_path += "."  # So that the later matches include the separating dot

    weight_paths = json.load(open(model_path / "model.safetensors.index.json", "r"))["weight_map"]
    per_safetensor_weights = {}

    for weight_name in weight_paths.keys():
        if not weight_name.startswith(module_path):
            continue
        per_safetensor_weights.setdefault(weight_paths[weight_name], []).append(weight_name)

    return {
        weight_name[len(module_path) :]: safetensor_state_dict[weight_name]
        for safetensor_file_path, weight_names in per_safetensor_weights.items()
        for safetensor_state_dict in [safetensors.torch.load_file(model_path / safetensor_file_path)]
        for weight_name in weight_names
    }


def load_reference_io(mode: str, module_range: str):
    path = (
        Path(os.getenv("DEEPSEEK_V3_CACHE", "/proj_sw/user_dev/deepseek-v3-cache"))
        / f"test_io_cache/{mode}.{module_range}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference IO cache file not found at {path}. Please run the {REFERENCE_IO_SCRIPT_NAME} script to create it. Did you set the 'HF_MODEL' environment variable coorectly?"
        )
    return torch.load(path)


def pad_tensor(tensor: torch.Tensor, mode: str, seq_len: int) -> torch.Tensor:
    assert mode in ["prefill", "decode"], f"Unsupported mode: {mode}"

    tensor_seq_len = tensor.shape[-2]
    seq_len = min(seq_len, tensor_seq_len)
    if mode == "decode" or seq_len < SEQ_LEN_CHUNK_SIZE:
        return tensor[..., :seq_len, :].clone()

    padded_seq_len = ttnn.core.roundup(seq_len, SEQ_LEN_CHUNK_SIZE)
    padded_tensor_shape = list(tensor.shape)
    padded_tensor_shape[-2] = padded_seq_len

    padded_tensor = torch.zeros(padded_tensor_shape, dtype=tensor.dtype, device=tensor.device)
    padded_tensor[..., : min(padded_seq_len, tensor_seq_len), :] = tensor

    return padded_tensor


def get_model_config(ModuleClass: type[AbstractModule], mode: str, *args, **kwargs) -> Any:
    """Get the module config for the given mode and kwargs."""
    if mode == "prefill":
        return ModuleClass.prefill_model_config(*args, **kwargs)
    elif mode == "decode":
        return ModuleClass.decode_model_config(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'prefill' and 'decode'.")


def run_module_forward(ModuleClass: type[AbstractModule], mode: str, *args, **kwargs) -> Any:
    """Run the module forward pass for the given mode and kwargs."""
    if mode == "prefill":
        return ModuleClass.forward_prefill(*args, **kwargs)
    elif mode == "decode":
        return ModuleClass.forward_decode(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'prefill' and 'decode'.")


def assert_hidden_dim_pcc(
    tt_output_torch: torch.Tensor, reference_output: torch.Tensor, pcc_required: float = 0.98
) -> float:
    assert (
        tt_output_torch.ndim == reference_output.ndim == 4
    ), f"Both model and reference outputs must be 4D tensors; got {tt_output_torch.ndim}D and {reference_output.ndim}D instead"
    assert (
        tt_output_torch.shape[0] == reference_output.shape[0]
    ), f"Model and reference output shape mismatch on dim 0 ({tt_output_torch.shape[0]} != {reference_output.shape[0]})"
    assert (
        tt_output_torch.shape[1] == reference_output.shape[1]
    ), f"Model and reference output shape mismatch on dim 1 ({tt_output_torch.shape[1]} != {reference_output.shape[1]})"
    assert (
        tt_output_torch.shape[3] == reference_output.shape[3]
    ), f"Model and reference output shape mismatch on dim 3 ({tt_output_torch.shape[3]} != {reference_output.shape[3]})"

    seq_len = min(tt_output_torch.shape[2], reference_output.shape[2])
    tt_output_torch = tt_output_torch[:, :, :seq_len, :]
    reference_output = reference_output[:, :, :seq_len, :]

    passing, pcc = comp_pcc(tt_output_torch, reference_output, pcc_required)
    logger.info(f"PCC: {pcc}")
    assert passing, f"Pearson Correlation Coefficient {pcc} is below required {pcc_required}."
    return pcc
