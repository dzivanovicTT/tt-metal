# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

import ttnn
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, SamplingParams, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision

# Global variables to store models and configurations
app = FastAPI(title="TT-Metal Combined Model Demo API", version="1.0.0")
models_store = {}


# Request/Response models
class SetupRequest(BaseModel):
    model1_name: str
    model2_name: str
    batch_size: int = 1
    max_seq_len: int = 2048
    data_parallel: int = 1
    instruct: bool = True


class SetupResponse(BaseModel):
    status: str
    message: str
    model1_setup_time: float
    model2_setup_time: float


class RunRequest(BaseModel):
    prompt1: str
    prompt2: Optional[str] = None
    max_generated_tokens: int = 100


class RunResponse(BaseModel):
    status: str
    model1_output: str
    model2_output: str
    metrics: Dict[str, Any]


def setup_logging():
    """Setup file logging for the demo"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"combined_llama_demo_{timestamp}.log"

    # Add file handler to logger
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info(f"Logging to file: {log_file}")
    return log_file


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    page_table = None
    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )
    return page_table


def setup_model(mesh_device, model_name, instruct=True, batch_size=1, max_seq_len=2048, data_parallel=1):
    """Setup a single TT model with given parameters"""
    logger.info(f"Setting up model: {model_name}")

    # Set the HF_MODEL environment variable for this model
    os.environ["HF_MODEL"] = model_name

    # Configuration
    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    paged_attention = True

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    global_batch_size = batch_size * data_parallel

    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    # Hybrid requires a model per submesh
    model_args = []
    model = []
    tt_kv_cache = []

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )

    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[0].tokenizer
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    return generator, model_args, page_table, tt_kv_cache, tokenizer


def run_inference(generator, model_args, page_table, tt_kv_cache, tokenizer, prompt, max_generated_tokens=100):
    """Run inference on a single model with given prompt"""
    instruct = True
    batch_size = 1
    max_seq_len = 2048

    # Preprocess inputs
    input_prompts = [prompt]
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        input_prompts, tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,  # Prefill warmup for all users, in case some users have different seqlens than others
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
    )

    logger.info("Starting prefill...")
    # Prefill forward
    prefill_start_time = time.time()
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
    )
    prefilled_token = torch.argmax(logits, dim=-1)
    prefill_time = time.time() - prefill_start_time
    logger.info(f"Prefill completed in {prefill_time:.2f} seconds")

    # Keep track of generated outputs
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        user_tok = int(prefilled_token[user].item())
        all_outputs[user].append(user_tok)

    user_done = [False] * batch_size

    # Sampling parameters (argmax/greedy decode)
    sampling_params = {"temperature": 0, "top_p": 0.08}
    device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)  # Use host sampling for simplicity

    # Initial positions
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

    # Start decoding
    iteration = 0
    users_decoding = True
    out_tok = prefilled_token
    enable_trace = True  # Enable tracing for better performance

    logger.info("Starting decode loop...")
    start_time = time.time()
    while users_decoding:
        # Run decode forward
        logits = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=device_sampling_params,
        )

        # Get the next token
        if device_sampling_params is not None:
            out_tok = logits.unsqueeze(1)
        else:
            _, out_tok = sample_host(
                logits,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                on_host=True,
            )

        current_pos += 1

        # Save output token
        for user in range(batch_size):
            user_tok = out_tok[user].item()
            if user_tok not in tokenizer.stop_tokens and user_done[user] == False:
                all_outputs[user].append(user_tok)
            else:
                user_done[user] = True
                logger.info(f"[User {user}] Finished decoding at iteration {iteration}")
                if all(user_done):
                    users_decoding = False

        iteration += 1

        # Upper limit of generated tokens
        if iteration >= max_generated_tokens:
            users_decoding = False

    decoding_time = time.time() - start_time
    tokens_per_second = iteration / decoding_time if decoding_time > 0 else 0

    logger.info(f"Decoding loop took {decoding_time:.2f} seconds")
    logger.info(f"Generated {iteration} tokens in {decoding_time:.2f} seconds")
    logger.info(f"Tokens per second: {tokens_per_second:.2f}")

    # Decode the final output
    output_text = tokenizer.decode(all_outputs[0])

    # Remove the original prompt from the output
    prompt_including_assistant_tags = tokenizer.decode(model_args[0].encode_prompt(prompt, instruct=instruct))
    output_text = output_text.replace(prompt_including_assistant_tags, "", 1)

    # Return both the output text and performance metrics
    return output_text.strip(), {
        "tokens_generated": iteration,
        "decoding_time": decoding_time,
        "tokens_per_second": tokens_per_second,
        "prefill_time": prefill_time,
    }


# Test function for pytest
@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, 2)  # Default to single device
        )
    ],
    indirect=True,
)
def test_combined_llama_demo(mesh_device, use_program_cache, reset_seeds):
    """Test function to run the combined Llama demo and calculate metrics"""
    # Setup first model
    model1_setup_start = time.time()
    generator1, model_args1, page_table1, tt_kv_cache1, tokenizer1 = setup_model(
        mesh_device, "meta-llama/Llama-3.2-1B-Instruct", instruct=True, batch_size=1, max_seq_len=2048, data_parallel=1
    )
    model1_setup_time = time.time() - model1_setup_start

    # Setup second model
    switching_start = time.time()
    generator2, model_args2, page_table2, tt_kv_cache2, tokenizer2 = setup_model(
        mesh_device, "meta-llama/Llama-3.2-3B-Instruct", instruct=True, batch_size=1, max_seq_len=2048, data_parallel=1
    )
    switching_time = time.time() - switching_start

    logger.info("\n=== Starting benchmark ===")

    # Run inference on first model
    model1_inference_start = time.time()
    model1_output, model1_metrics = run_inference(
        generator1,
        model_args1,
        page_table1,
        tt_kv_cache1,
        tokenizer1,
        "What is the capital of France? Give a brief explanation.",
        max_generated_tokens=100,
    )
    model1_inference_time = time.time() - model1_inference_start

    # Run inference on second model
    second_prompt = f"Based on this information: '{model1_output}', can you expand on the topic and provide more historical context?"
    model2_inference_start = time.time()
    model2_output, model2_metrics = run_inference(
        generator2, model_args2, page_table2, tt_kv_cache2, tokenizer2, second_prompt, max_generated_tokens=100
    )
    model2_inference_time = time.time() - model2_inference_start

    # Calculate and log statistics
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)

    logger.info("\nTimes:")
    logger.info(f"Model 1 Setup: {model1_setup_time:.2f}s")
    logger.info(f"Model 1 Inference: {model1_inference_time:.2f}s")
    logger.info(f"Model 1 Prefill: {model1_metrics['prefill_time']:.2f}s")
    logger.info(f"Model 1 Decode: {model1_metrics['decoding_time']:.2f}s")
    logger.info(f"Model Switching: {switching_time:.2f}s")
    logger.info(f"Model 2 Inference: {model2_inference_time:.2f}s")
    logger.info(f"Model 2 Prefill: {model2_metrics['prefill_time']:.2f}s")
    logger.info(f"Model 2 Decode: {model2_metrics['decoding_time']:.2f}s")
    logger.info(f"Total Time: {model1_inference_time + model2_inference_time:.2f}s")

    # Tokens per second metrics
    logger.info("\nTokens per second metrics:")
    logger.info(f"Model 1 Tokens/s: {model1_metrics['tokens_per_second']:.2f}")
    logger.info(f"Model 2 Tokens/s: {model2_metrics['tokens_per_second']:.2f}")
    logger.info(f"Model 1 Tokens Generated: {model1_metrics['tokens_generated']:.1f}")
    logger.info(f"Model 2 Tokens Generated: {model2_metrics['tokens_generated']:.1f}")

    # Calculate combined metrics
    total_tokens_generated = model1_metrics["tokens_generated"] + model2_metrics["tokens_generated"]
    total_decoding_time = model1_metrics["decoding_time"] + model2_metrics["decoding_time"]
    combined_tokens_per_second = total_tokens_generated / total_decoding_time if total_decoding_time > 0 else 0

    # Calculate combined prefill and decode metrics
    total_prefill_time = model1_metrics["prefill_time"] + model2_metrics["prefill_time"]
    total_decode_time = model1_metrics["decoding_time"] + model2_metrics["decoding_time"]

    # Verify timing relationship: Total Inference = Prefill + Decode
    calculated_total_inference = (
        model1_metrics["prefill_time"]
        + model1_metrics["decoding_time"]
        + model2_metrics["prefill_time"]
        + model2_metrics["decoding_time"]
    )
    measured_total_inference = model1_inference_time + model2_inference_time

    logger.info(f"\nCombined metrics:")
    logger.info(f"Total Tokens Generated: {total_tokens_generated}")
    logger.info(f"Combined Tokens/s: {combined_tokens_per_second:.2f}")
    logger.info(f"Total Prefill Time: {total_prefill_time:.2f}s")
    logger.info(f"Total Decode Time: {total_decode_time:.2f}s")

    logger.info(f"\nTiming Verification:")
    logger.info(f"Calculated Total Inference (Prefill + Decode): {calculated_total_inference:.2f}s")
    logger.info(f"Measured Total Inference: {measured_total_inference:.2f}s")
    logger.info(f"Difference: {abs(calculated_total_inference - measured_total_inference):.3f}s")
    logger.info("=" * 80)


# FastAPI endpoints
@app.post("/setup", response_model=SetupResponse)
async def setup_models(request: SetupRequest):
    """Setup two models for inference"""
    try:
        logger.info(f"Setting up models: {request.model1_name} and {request.model2_name}")

        # Initialize mesh device for N300 (1x2 mesh) with proper device parameter processing
        os.environ["MESH_DEVICE"] = "N300"

        # Helper functions for proper device configuration (copied from tests/scripts/common.py)
        def get_dispatch_core_type():
            dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
            if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
                dispatch_core_type = ttnn.device.DispatchCoreType.ETH
            return dispatch_core_type

        def get_updated_device_params(device_params):
            dispatch_core_type = get_dispatch_core_type()
            new_device_params = device_params.copy()

            is_blackhole = ttnn.get_arch_name() == "blackhole"
            dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)

            # Set default if not specified
            if dispatch_core_axis is None:
                dispatch_core_axis = ttnn.DispatchCoreAxis.COL if is_blackhole else ttnn.DispatchCoreAxis.ROW

            # Force COL for blackhole regardless of user setting
            if is_blackhole and dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                logger.warning(
                    "blackhole arch does not support DispatchCoreAxis.Row, using DispatchCoreAxis.COL instead."
                )
                dispatch_core_axis = ttnn.DispatchCoreAxis.COL

            dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
            new_device_params["dispatch_core_config"] = dispatch_core_config
            return new_device_params

        # Use the same device parameter processing as pytest fixtures
        device_params = {"trace_region_size": 23887872, "num_command_queues": 1}
        updated_device_params = get_updated_device_params(device_params)

        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 2), **updated_device_params  # N300 uses 1x2 mesh
        )

        # Enable program cache - REQUIRED for tracing to work properly
        # This is what pytest fixtures do automatically but FastAPI needs to do manually
        mesh_device.enable_program_cache()
        logger.info("Program cache enabled for tracing support")

        # Setup first model
        model1_setup_start = time.time()
        generator1, model_args1, page_table1, tt_kv_cache1, tokenizer1 = setup_model(
            mesh_device,
            request.model1_name,
            instruct=request.instruct,
            batch_size=request.batch_size,
            max_seq_len=request.max_seq_len,
            data_parallel=request.data_parallel,
        )
        model1_setup_time = time.time() - model1_setup_start

        # Setup second model
        model2_setup_start = time.time()
        generator2, model_args2, page_table2, tt_kv_cache2, tokenizer2 = setup_model(
            mesh_device,
            request.model2_name,
            instruct=request.instruct,
            batch_size=request.batch_size,
            max_seq_len=request.max_seq_len,
            data_parallel=request.data_parallel,
        )
        model2_setup_time = time.time() - model2_setup_start

        # Store models globally
        models_store.clear()  # Clear any existing models
        models_store.update(
            {
                "model1": {
                    "generator": generator1,
                    "model_args": model_args1,
                    "page_table": page_table1,
                    "tt_kv_cache": tt_kv_cache1,
                    "tokenizer": tokenizer1,
                    "name": request.model1_name,
                },
                "model2": {
                    "generator": generator2,
                    "model_args": model_args2,
                    "page_table": page_table2,
                    "tt_kv_cache": tt_kv_cache2,
                    "tokenizer": tokenizer2,
                    "name": request.model2_name,
                },
                "mesh_device": mesh_device,
                "config": {
                    "batch_size": request.batch_size,
                    "max_seq_len": request.max_seq_len,
                    "data_parallel": request.data_parallel,
                    "instruct": request.instruct,
                },
            }
        )

        logger.info(f"Models setup completed. Model 1: {model1_setup_time:.2f}s, Model 2: {model2_setup_time:.2f}s")

        return SetupResponse(
            status="success",
            message=f"Successfully setup models {request.model1_name} and {request.model2_name}",
            model1_setup_time=model1_setup_time,
            model2_setup_time=model2_setup_time,
        )

    except Exception as e:
        logger.error(f"Error setting up models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to setup models: {str(e)}")


@app.post("/run", response_model=RunResponse)
async def run_inference_endpoint(request: RunRequest):
    """Run inference on both models"""
    try:
        if not models_store:
            raise HTTPException(status_code=400, detail="Models not setup. Call /setup endpoint first.")

        logger.info("Starting inference run")

        # Get models from store
        model1 = models_store["model1"]
        model2 = models_store["model2"]

        # Run inference on first model
        model1_inference_start = time.time()
        model1_output, model1_metrics = run_inference(
            model1["generator"],
            model1["model_args"],
            model1["page_table"],
            model1["tt_kv_cache"],
            model1["tokenizer"],
            request.prompt1,
            max_generated_tokens=request.max_generated_tokens,
        )
        model1_inference_time = time.time() - model1_inference_start

        # Determine second prompt
        if request.prompt2:
            second_prompt = request.prompt2
        else:
            second_prompt = f"Based on this information: '{model1_output}', can you expand on the topic and provide more historical context?"

        # Run inference on second model
        model2_inference_start = time.time()
        model2_output, model2_metrics = run_inference(
            model2["generator"],
            model2["model_args"],
            model2["page_table"],
            model2["tt_kv_cache"],
            model2["tokenizer"],
            second_prompt,
            max_generated_tokens=request.max_generated_tokens,
        )
        model2_inference_time = time.time() - model2_inference_start

        # Calculate combined metrics
        total_tokens_generated = model1_metrics["tokens_generated"] + model2_metrics["tokens_generated"]
        total_decoding_time = model1_metrics["decoding_time"] + model2_metrics["decoding_time"]
        combined_tokens_per_second = total_tokens_generated / total_decoding_time if total_decoding_time > 0 else 0

        # Calculate combined prefill and decode metrics
        total_prefill_time = model1_metrics["prefill_time"] + model2_metrics["prefill_time"]
        total_decode_time = model1_metrics["decoding_time"] + model2_metrics["decoding_time"]

        # Verify timing relationship
        calculated_total_inference = (
            model1_metrics["prefill_time"]
            + model1_metrics["decoding_time"]
            + model2_metrics["prefill_time"]
            + model2_metrics["decoding_time"]
        )
        measured_total_inference = model1_inference_time + model2_inference_time

        metrics = {
            "model1": {
                "inference_time": model1_inference_time,
                "prefill_time": model1_metrics["prefill_time"],
                "decoding_time": model1_metrics["decoding_time"],
                "tokens_per_second": model1_metrics["tokens_per_second"],
                "tokens_generated": model1_metrics["tokens_generated"],
            },
            "model2": {
                "inference_time": model2_inference_time,
                "prefill_time": model2_metrics["prefill_time"],
                "decoding_time": model2_metrics["decoding_time"],
                "tokens_per_second": model2_metrics["tokens_per_second"],
                "tokens_generated": model2_metrics["tokens_generated"],
            },
            "combined": {
                "total_tokens_generated": total_tokens_generated,
                "combined_tokens_per_second": combined_tokens_per_second,
                "total_prefill_time": total_prefill_time,
                "total_decode_time": total_decode_time,
                "calculated_total_inference": calculated_total_inference,
                "measured_total_inference": measured_total_inference,
                "timing_difference": abs(calculated_total_inference - measured_total_inference),
            },
        }

        logger.info(f"Inference completed. Total time: {measured_total_inference:.2f}s")

        return RunResponse(status="success", model1_output=model1_output, model2_output=model2_output, metrics=metrics)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {str(e)}")


@app.get("/status")
async def get_status():
    """Get the current status of the API"""
    if models_store:
        return {
            "status": "ready",
            "models_loaded": True,
            "model1_name": models_store["model1"]["name"],
            "model2_name": models_store["model2"]["name"],
            "config": models_store["config"],
        }
    else:
        return {
            "status": "not_ready",
            "models_loaded": False,
            "message": "No models loaded. Call /setup endpoint first.",
        }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TT-Metal Combined Model Demo API",
        "version": "1.0.0",
        "endpoints": {
            "/setup": "POST - Setup two models for inference",
            "/run": "POST - Run inference on loaded models",
            "/status": "GET - Check API status",
            "/cleanup": "POST - Clean up models and close mesh device",
            "/docs": "GET - API documentation",
        },
    }


@app.post("/cleanup")
async def cleanup_models():
    """Clean up models and close mesh device"""
    try:
        if models_store and "mesh_device" in models_store:
            mesh_device = models_store["mesh_device"]
            if hasattr(mesh_device, "close") or mesh_device is not None:
                try:
                    ttnn.close_mesh_device(mesh_device)
                    logger.info("Mesh device closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing mesh device: {e}")

        models_store.clear()
        logger.info("Models cleaned up successfully")

        return {"status": "success", "message": "Models and mesh device cleaned up successfully"}

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return {"status": "error", "message": f"Error during cleanup: {str(e)}"}


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
