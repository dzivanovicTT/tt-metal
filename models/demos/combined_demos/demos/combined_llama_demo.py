# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time

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
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision


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

    logger.info("Starting prefill...")
    # Prefill forward
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
    )
    prefilled_token = torch.argmax(logits, dim=-1)
    logger.info("Prefill finished")

    # Keep track of generated outputs
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        user_tok = int(prefilled_token[user].item())
        all_outputs[user].append(user_tok)

    user_done = [False] * batch_size

    # Sampling parameters (argmax/greedy decode)
    sampling_params = {"temperature": 0, "top_p": 0.08}
    device_sampling_params = None  # Use host sampling for simplicity

    # Initial positions
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

    # Start decoding
    iteration = 0
    users_decoding = True
    out_tok = prefilled_token
    enable_trace = False  # Disable tracing for simplicity

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
    logger.info(f"Decoding loop took {decoding_time:.2f} seconds")
    logger.info(f"Generated {iteration} tokens in {decoding_time:.2f} seconds")
    logger.info(f"Tokens per second: {iteration / decoding_time:.2f}")

    # Decode the final output
    output_text = tokenizer.decode(all_outputs[0])

    # Remove the original prompt from the output
    prompt_including_assistant_tags = tokenizer.decode(model_args[0].encode_prompt(prompt, instruct=instruct))
    output_text = output_text.replace(prompt_including_assistant_tags, "", 1)

    return output_text.strip()


def run_combined_demo(mesh_device):
    """Main function to run the combined demo with two models"""

    # Start overall timing
    overall_start_time = time.time()

    # First model: Llama-3.2-1B-Instruct
    model1_name = "meta-llama/Llama-3.2-1B-Instruct"
    initial_prompt = "What is the capital of France? Give a brief explanation."

    logger.info(f"=== Running Model 1: {model1_name} ===")
    logger.info(f"Initial prompt: {initial_prompt}")

    # Setup first model
    model1_setup_start = time.time()
    generator1, model_args1, page_table1, tt_kv_cache1, tokenizer1 = setup_model(
        mesh_device, model1_name, instruct=True, batch_size=1, max_seq_len=2048, data_parallel=1
    )
    model1_setup_time = time.time() - model1_setup_start
    logger.info(f"Model 1 setup took {model1_setup_time:.2f} seconds")

    # Run inference on first model
    model1_inference_start = time.time()
    model1_output = run_inference(
        generator1, model_args1, page_table1, tt_kv_cache1, tokenizer1, initial_prompt, max_generated_tokens=100
    )
    model1_inference_time = time.time() - model1_inference_start

    logger.info(f"Model 1 output: {model1_output}")
    logger.info(f"Model 1 total inference time: {model1_inference_time:.2f} seconds")

    # Start tracking switching time
    switching_start_time = time.time()

    # Second model: Llama-3.2-3B-Instruct
    model2_name = "meta-llama/Llama-3.2-3B-Instruct"
    # Use the output from model 1 as input to model 2
    second_prompt = f"Based on this information: '{model1_output}', can you expand on the topic and provide more historical context?"

    logger.info(f"\n=== Running Model 2: {model2_name} ===")
    logger.info(f"Second prompt: {second_prompt}")

    # Setup second model
    generator2, model_args2, page_table2, tt_kv_cache2, tokenizer2 = setup_model(
        mesh_device, model2_name, instruct=True, batch_size=1, max_seq_len=2048, data_parallel=1
    )

    switching_time = time.time() - switching_start_time
    logger.info(f"Model switching time (including Model 2 setup): {switching_time:.2f} seconds")

    # Run inference on second model
    model2_inference_start = time.time()
    model2_output = run_inference(
        generator2, model_args2, page_table2, tt_kv_cache2, tokenizer2, second_prompt, max_generated_tokens=150
    )
    model2_inference_time = time.time() - model2_inference_start

    logger.info(f"Model 2 output: {model2_output}")
    logger.info(f"Model 2 total inference time: {model2_inference_time:.2f} seconds")

    # Calculate total times
    total_inference_time = model1_inference_time + model2_inference_time
    overall_total_time = time.time() - overall_start_time

    # Print final results with timing summary
    logger.info("\n" + "=" * 80)
    logger.info("COMBINED DEMO RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model 1 ({model1_name}):")
    logger.info(f"Input: {initial_prompt}")
    logger.info(f"Output: {model1_output}")
    logger.info(f"Inference Time: {model1_inference_time:.2f} seconds")
    logger.info("\n" + "-" * 80)
    logger.info(f"Model 2 ({model2_name}):")
    logger.info(f"Input: {second_prompt}")
    logger.info(f"Output: {model2_output}")
    logger.info(f"Inference Time: {model2_inference_time:.2f} seconds")
    logger.info("\n" + "-" * 80)
    logger.info("TIMING SUMMARY:")
    logger.info(f"Model 1 Setup Time: {model1_setup_time:.2f} seconds")
    logger.info(f"Model 1 Inference Time: {model1_inference_time:.2f} seconds")
    logger.info(f"Model Switching Time: {switching_time:.2f} seconds")
    logger.info(f"Model 2 Inference Time: {model2_inference_time:.2f} seconds")
    logger.info(f"Total Inference Time (Model 1 + Model 2): {total_inference_time:.2f} seconds")
    logger.info(f"Overall Total Time: {overall_total_time:.2f} seconds")
    logger.info("=" * 80)


# Test function for pytest
@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, 1)  # Default to single device
        )
    ],
    indirect=True,
)
def test_combined_llama_demo(mesh_device, use_program_cache, reset_seeds):
    """Test function to run the combined Llama demo"""
    run_combined_demo(mesh_device)


if __name__ == "__main__":
    # Setup mesh device (this will be handled by the test framework in actual usage)
    try:
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), device_ids=[0])
        run_combined_demo(mesh_device)
    finally:
        ttnn.close_mesh_device(mesh_device)
