# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, SamplingParams, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision


def setup_logging():
    """Setup file logging for the demo"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"single_llama_demo_{timestamp}.log"

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
    enable_trace = True  # Disable tracing for simplicity

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


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, 4)  # Default to single device
        )
    ],
    indirect=True,
)
def test_single_llama_demo(mesh_device, use_program_cache, reset_seeds):
    """Test function to run the single Llama demo and calculate average metrics"""
    num_runs = 10
    total_times = []
    model_setup_times = []
    model_inference_times = []

    # Add tokens per second tracking
    model_tokens_per_second = []
    model_tokens_generated = []

    # Add prefill time tracking
    model_prefill_times = []

    # Add decode time tracking
    model_decode_times = []

    # Setup model
    model_setup_start = time.time()
    generator, model_args, page_table, tt_kv_cache, tokenizer = setup_model(
        mesh_device, "/home/ttuser/atupe/QwQ-32B", instruct=True, batch_size=1, max_seq_len=2048, data_parallel=1
    )
    model_setup_time = time.time() - model_setup_start
    model_setup_times.append(model_setup_time)

    logger.info(f"\n=== Starting benchmark with {num_runs} runs ===")

    for i in range(num_runs):
        logger.info(f"\nRun {i+1}/{num_runs}")
        run_start = time.time()

        # Run inference on model
        model_inference_start = time.time()
        model_output, model_metrics = run_inference(
            generator,
            model_args,
            page_table,
            tt_kv_cache,
            tokenizer,
            "What is the capital of France? Give a brief explanation.",
            max_generated_tokens=100,
        )
        model_inference_time = time.time() - model_inference_start
        if i > 0:
            model_inference_times.append(model_inference_time)
            model_tokens_per_second.append(model_metrics["tokens_per_second"])
            model_tokens_generated.append(model_metrics["tokens_generated"])
            model_prefill_times.append(model_metrics["prefill_time"])
            model_decode_times.append(model_metrics["decoding_time"])

        total_time = time.time() - run_start
        if i > 0:
            total_times.append(total_time)

    # Calculate and log statistics
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)
    num_runs = num_runs - 1
    logger.info(f"Number of runs: {num_runs}")
    logger.info("\nAverage times:")
    logger.info(f"Model Setup: {sum(model_setup_times):.2f}s")
    logger.info(f"Model Inference: {sum(model_inference_times)/num_runs:.2f}s")
    logger.info(f"Model Average Prefill: {sum(model_prefill_times)/num_runs:.2f}s")
    logger.info(f"Model Average Decode: {sum(model_decode_times)/num_runs:.2f}s")
    logger.info(f"Total Time per Run: {sum(total_times)/num_runs:.2f}s")

    # Add tokens per second metrics
    logger.info("\nTokens per second metrics:")
    logger.info(f"Model Average Tokens/s: {sum(model_tokens_per_second)/num_runs:.2f}")
    logger.info(f"Model Average Tokens Generated: {sum(model_tokens_generated)/num_runs:.1f}")

    # Calculate combined metrics
    total_tokens_generated = sum(model_tokens_generated)
    total_decoding_time = sum(model_decode_times)
    tokens_per_second = total_tokens_generated / total_decoding_time if total_decoding_time > 0 else 0

    # Calculate prefill metrics
    total_prefill_time = sum(model_prefill_times)
    average_prefill_time = total_prefill_time / num_runs if num_runs > 0 else 0

    # Calculate decode metrics
    total_decode_time = sum(model_decode_times)
    average_decode_time = total_decode_time / num_runs if num_runs > 0 else 0

    # Verify timing relationship: Total Inference = Prefill + Decode
    calculated_total_inference = (sum(model_prefill_times) + sum(model_decode_times)) / num_runs
    measured_total_inference = sum(model_inference_times) / num_runs

    logger.info(f"\nCombined metrics:")
    logger.info(f"Total Tokens Generated: {total_tokens_generated}")
    logger.info(f"Average Tokens/s: {tokens_per_second:.2f}")
    logger.info(f"Average Prefill Time: {average_prefill_time:.2f}s")
    logger.info(f"Average Decode Time: {average_decode_time:.2f}s")
    logger.info(f"Total Prefill Time Across All Runs: {total_prefill_time:.2f}s")
    logger.info(f"Total Decode Time Across All Runs: {total_decode_time:.2f}s")

    logger.info(f"\nTiming Verification:")
    logger.info(f"Calculated Total Inference (Prefill + Decode): {calculated_total_inference:.2f}s")
    logger.info(f"Measured Total Inference: {measured_total_inference:.2f}s")
    logger.info(f"Difference: {abs(calculated_total_inference - measured_total_inference):.3f}s")
    logger.info("=" * 80)
