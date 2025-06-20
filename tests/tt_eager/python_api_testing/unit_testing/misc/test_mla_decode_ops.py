# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from dataclasses import dataclass


TP = 8
DP = 4


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    All Single-Device Shapes!
    """

    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


class ModelConfig:
    def __init__(self, model_args: ModelArgs):
        self.args = model_args
        self.args.qk_head_dim = self.args.qk_nope_head_dim + self.args.qk_rope_head_dim

        self.bsz = 64
        self.configs = {}

        #################
        ### MLA Configs
        #################

        # wq_a
        self.configs["WQA_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.dim // TP)
        self.configs["WQA_IN1_SHAPE"] = (1, 1, self.args.dim // TP, self.args.q_lora_rank)
        self.configs["WQA_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQA_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQA_PROGRAM_CFG"] = None
        self.configs["WQA_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wq_b
        self.configs["WQB_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.q_lora_rank)
        self.configs["WQB_IN1_SHAPE"] = (1, 1, self.args.q_lora_rank, (self.args.n_heads * self.args.qk_head_dim) // TP)
        self.configs["WQB_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQB_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQB_PROGRAM_CFG"] = None
        self.configs["WQB_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_a
        self.configs["WKV_A_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.dim // TP)
        self.configs["WKV_A_IN1_SHAPE"] = (
            1,
            1,
            self.args.dim // TP,
            self.args.kv_lora_rank + self.args.qk_rope_head_dim,
        )
        self.configs["WKV_A_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_A_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_A_PROGRAM_CFG"] = None
        self.configs["WKV_A_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b1
        self.configs["WKV_B1_IN0_SHAPE"] = (1, self.bsz // DP, self.args.n_heads // TP, self.args.qk_nope_head_dim)
        self.configs["WKV_B1_IN1_SHAPE"] = (
            1,
            self.args.n_heads // TP,
            self.args.qk_nope_head_dim,
            self.args.kv_lora_rank,
        )
        self.configs["WKV_B1_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B1_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B1_PROGRAM_CFG"] = None
        self.configs["WKV_B1_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b2
        self.configs["WKV_B2_IN0_SHAPE"] = (1, self.bsz // DP, self.args.n_heads // TP, self.args.kv_lora_rank)
        self.configs["WKV_B2_IN1_SHAPE"] = (1, self.args.n_heads // TP, self.args.kv_lora_rank, self.args.v_head_dim)
        self.configs["WKV_B2_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B2_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B2_PROGRAM_CFG"] = None
        self.configs["WKV_B2_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wo
        self.configs["WO_IN0_SHAPE"] = (1, self.bsz // DP, self.args.n_heads * self.args.v_head_dim)
        self.configs["WO_IN1_SHAPE"] = (1, 1, self.args.n_heads * self.args.v_head_dim, self.args.dim // TP)
        self.configs["WO_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WO_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WO_PROGRAM_CFG"] = None
        self.configs["WO_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG


cfg = ModelConfig(ModelArgs())


def run_matmul_impl(
    device,
    shapes,
    dtypes,
    program_config,
    memory_configs,
):
    layout = ttnn.TILE_LAYOUT

    in0_shape, in1_shape = shapes
    in0_dtype, in1_dtype = dtypes
    in0_mem_config, in1_mem_config, out_mem_config = memory_configs

    # Log configs
    logger.info("Running matmul with the following configurations:")
    logger.info(f"Input 0 Shape: {in0_shape}, Dtype: {in0_dtype}, Memory Config: {in0_mem_config}")
    logger.info(f"Input 1 Shape: {in1_shape}, Dtype: {in1_dtype}, Memory Config: {in1_mem_config}")
    logger.info(f"Output Memory Config: {out_mem_config}")
    logger.info(f"Program Config: {program_config}")

    #################
    ### Torch
    #################
    in0 = torch.randn(in0_shape).float()
    in1 = torch.randn(in1_shape).float()
    out_torch = in0 @ in1

    #################
    ### TT-NN
    #################
    tt_in0 = ttnn.from_torch(
        in0,
        device=device,
        dtype=in0_dtype,
        memory_config=in0_mem_config,
        layout=layout,
    )

    tt_in1 = ttnn.from_torch(
        in1,
        device=device,
        dtype=in1_dtype,
        memory_config=in1_mem_config,
        layout=layout,
    )

    tt_out = ttnn.matmul(
        tt_in0,
        tt_in1,
        memory_config=out_mem_config,
        program_config=program_config,
    )

    tt_out_torch = ttnn.to_torch(tt_out)

    #################
    ### Validation
    #################

    pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


@pytest.mark.parametrize(
    "shapes, dtypes, program_config, memory_configs",
    [
        (  # wq_a
            [cfg.configs["WQA_IN0_SHAPE"], cfg.configs["WQA_IN1_SHAPE"]],
            [cfg.configs["WQA_IN0_DTYPE"], cfg.configs["WQA_IN1_DTYPE"]],
            cfg.configs["WQA_PROGRAM_CFG"],
            [cfg.configs["WQA_IN0_MEM_CFG"], cfg.configs["WQA_IN1_MEM_CFG"], cfg.configs["WQA_OUT_MEM_CFG"]],
        ),
        (  # wq_b
            [cfg.configs["WQB_IN0_SHAPE"], cfg.configs["WQB_IN1_SHAPE"]],
            [cfg.configs["WQB_IN0_DTYPE"], cfg.configs["WQB_IN1_DTYPE"]],
            cfg.configs["WQB_PROGRAM_CFG"],
            [cfg.configs["WQB_IN0_MEM_CFG"], cfg.configs["WQB_IN1_MEM_CFG"], cfg.configs["WQB_OUT_MEM_CFG"]],
        ),
        (  # wkv_a
            [cfg.configs["WKV_A_IN0_SHAPE"], cfg.configs["WKV_A_IN1_SHAPE"]],
            [cfg.configs["WKV_A_IN0_DTYPE"], cfg.configs["WKV_A_IN1_DTYPE"]],
            cfg.configs["WKV_A_PROGRAM_CFG"],
            [cfg.configs["WKV_A_IN0_MEM_CFG"], cfg.configs["WKV_A_IN1_MEM_CFG"], cfg.configs["WKV_A_OUT_MEM_CFG"]],
        ),
        (  # wkv_b1
            [cfg.configs["WKV_B1_IN0_SHAPE"], cfg.configs["WKV_B1_IN1_SHAPE"]],
            [cfg.configs["WKV_B1_IN0_DTYPE"], cfg.configs["WKV_B1_IN1_DTYPE"]],
            cfg.configs["WKV_B1_PROGRAM_CFG"],
            [cfg.configs["WKV_B1_IN0_MEM_CFG"], cfg.configs["WKV_B1_IN1_MEM_CFG"], cfg.configs["WKV_B1_OUT_MEM_CFG"]],
        ),
        (  # wkv_b2
            [cfg.configs["WKV_B2_IN0_SHAPE"], cfg.configs["WKV_B2_IN1_SHAPE"]],
            [cfg.configs["WKV_B2_IN0_DTYPE"], cfg.configs["WKV_B2_IN1_DTYPE"]],
            cfg.configs["WKV_B2_PROGRAM_CFG"],
            [cfg.configs["WKV_B2_IN0_MEM_CFG"], cfg.configs["WKV_B2_IN1_MEM_CFG"], cfg.configs["WKV_B2_OUT_MEM_CFG"]],
        ),
        (  # wo
            [cfg.configs["WO_IN0_SHAPE"], cfg.configs["WO_IN1_SHAPE"]],
            [cfg.configs["WO_IN0_DTYPE"], cfg.configs["WO_IN1_DTYPE"]],
            cfg.configs["WO_PROGRAM_CFG"],
            [cfg.configs["WO_IN0_MEM_CFG"], cfg.configs["WO_IN1_MEM_CFG"], cfg.configs["WO_OUT_MEM_CFG"]],
        ),
    ],
    ids=[
        "wq_a",
        "wq_b",
        "wkv_a",
        "wkv_b1",
        "wkv_b2",
        "wo",
    ],
)
def test_matmuls(
    device,
    shapes,
    dtypes,
    program_config,
    memory_configs,
    use_program_cache,
    function_level_defaults,
    reset_seeds,
):
    run_matmul_impl(
        device,
        shapes=shapes,
        dtypes=dtypes,
        program_config=program_config,
        memory_configs=memory_configs,
    )
