#!/usr/bin/env python3
"""
PQCache Mathematics CoT (Chain-of-Thought) Reasoning Evaluation Script
支持 HuggingFaceH4/aime_2024 和 HuggingFaceH4/MATH-500 数据集
参考 R-KV 项目设计，适配 PQCache 方法
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import torch
from loguru import logger
from datasets import load_dataset
import traceback

# Add PQCache imports
sys.path.append('/home/pai/data/PQCache')
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.retrieval_based.pq_search import initialize_objects
from transformers import AutoTokenizer, AutoConfig

# R-KV style prompt template (adapted from R-KV/HuggingFace/run_math.py)
R_KV_PROMPT_TEMPLATE = "You are given a math problem.\n\nProblem: {problem}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"

# Dataset key mappings (based on actual local file format)
DATASET_KEYS = {
    "aime_2024": ["problem", "answer"],  # AIME uses "problem" field
    "math_500": ["problem", "answer"],   # MATH-500 uses "problem" field
}

def parse_args():
    parser = argparse.ArgumentParser(description="PQCache Mathematics CoT Reasoning Evaluation")
    
    # Model and task arguments
    parser.add_argument('--model', type=str, default="llama-3.1", 
                        choices=["llama-3.1"], help="Model name")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["aime_2024", "math_500"], 
                        help="Mathematics dataset to evaluate")
    parser.add_argument('--dataset_path', type=str, default=None,
                        help="Path to custom dataset file (optional)")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Path to save results (optional, overrides output_dir)")
    parser.add_argument('--num_eval_examples', type=int, default=-1,
                        help="Number of examples to evaluate (-1 for all)")
    
    # Generation parameters
    parser.add_argument('--max_seq_length', type=int, default=32768,
                        help="Maximum sequence length for input")
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help="Maximum new tokens to generate (for CoT reasoning)")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="Temperature for generation")
    parser.add_argument('--top_p', type=float, default=1.0,
                        help="Top-p for nucleus sampling")
    
    # Method configuration
    parser.add_argument("--method", type=str, default="pqcache",
                        choices=["pqcache", "fullkv"], 
                        help="Compression method")
    parser.add_argument('--enable_vq_cache', action='store_true', default=True,
                        help="Enable VQ cache compression (default: True)")
    parser.add_argument('--enable_h2o_cache', action='store_true',
                        help="Enable H2O cache compression")
    
    # PQCache specific configuration
    parser.add_argument("--compress_ratio", type=float, default=0.1,
                        help="KV cache compression ratio")
    parser.add_argument("--important_ratio", type=float, default=0.5,
                        help="Ratio of important tokens to keep")
    parser.add_argument("--recent_ratio", type=float, default=0.5,
                        help="Ratio of recent tokens to keep")
    parser.add_argument("--n_subvec_per_head", type=int, default=2,
                        help="Number of sub-vectors per attention head")
    parser.add_argument("--n_subbits", type=int, default=6,
                        help="Number of sub-bits for quantization")
    parser.add_argument("--compressor", type=str, default="pq_search",
                        choices=["pq_search"], help="PQCache compressor type")
    
    # Hardware and output
    parser.add_argument('--device_id', type=int, default=0,
                        help="CUDA device ID")
    parser.add_argument('--output_dir', type=str, default='math_cot_results',
                        help="Output directory for results")
    parser.add_argument('--verbose', action='store_true',
                        help="Verbose output")
    
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load PQCache model and tokenizer"""
    
    # Model paths mapping
    model2path = {
        "llama-3.1": "meta-llama/Llama-3.1-8B-Instruct"
    }
    
    model_name_or_path = model2path[args.model]
    device = torch.device(f"cuda:{args.device_id}")
    
    logger.info(f"Loading model from {model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config and set PQCache parameters
    config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Fix RoPE scaling issues
    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
        if 'type' not in config.rope_scaling and 'rope_type' not in config.rope_scaling:
            config.rope_scaling['type'] = 'linear'
        elif 'type' not in config.rope_scaling and 'rope_type' in config.rope_scaling:
            config.rope_scaling['type'] = config.rope_scaling['rope_type']
    
    # Set PQCache configuration
    config.compress_ratio = args.compress_ratio
    config.important_ratio = args.important_ratio
    config.recent_ratio = args.recent_ratio
    config.pp_size = 1
    config.sink_size = 16
    config.keyformer_mode = False
    config.drop_ratio = 0.0
    config.preserve_layer = 0
    config.score_func = "sum"
    config.compressor = args.compressor
    config.threshold = 1.0
    config.n_subvec_per_head = args.n_subvec_per_head
    config.n_subbits = args.n_subbits
    config.topr = 32
    config.gqa = True
    config.max_iter = 0
    config.device = device
    
    # Initialize PQCache objects
    if config.compressor == "pq_search":
        config.max_seq_len = args.max_seq_length
        config.cache_block_size = 128
        config.global_cache_size = 4096
        config.cache_topk = 32
        initialize_objects(config, model=args.model)
    
    # Load model based on cache method
    if args.enable_vq_cache:
        model = VQLlama31ForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    elif args.enable_h2o_cache:
        from h2o_method.h2o_attention import H2OLlamaForCausalLM
        model = H2OLlamaForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    else:
        # Default to VQ cache for backwards compatibility
        model = VQLlama31ForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    
    # Device mapping for CUDA_VISIBLE_DEVICES
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        model = model.half().eval().to("cuda:0")
    else:
        model = model.half().eval().to(device)
    
    logger.info("Model loaded successfully")
    return model, tokenizer

def load_math_dataset(dataset_name: str, num_examples: int = -1):
    """Load mathematics dataset from local JSONL files"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Local file mapping
    dataset_files = {
        "aime_2024": "/home/pai/data/PQCache/data/cot_math/aime_2024_train.jsonl",
        "math_500": "/home/pai/data/PQCache/data/cot_math/math_500_test.jsonl"
    }
    
    if dataset_name not in dataset_files:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataset_path = dataset_files[dataset_name]
    logger.info(f"Loading from: {dataset_path}")
    
    # Load JSONL file
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    logger.info(f"Loaded {len(examples)} samples")
    
    if num_examples > 0:
        examples = examples[:num_examples]
        logger.info(f"Selected {len(examples)} samples for evaluation")
    
    return examples

def create_cot_prompt(sample: Dict[str, Any], dataset_name: str) -> str:
    """Create Chain-of-Thought prompt following R-KV style"""
    
    # Get problem using dataset key mapping
    problem_key = DATASET_KEYS[dataset_name][0]
    problem = sample.get(problem_key, sample.get("problem", sample.get("question", "")))
    
    # Use R-KV style prompt template
    prompt = R_KV_PROMPT_TEMPLATE.format(problem=problem)
    
    return prompt

def generate_cot_response(model, tokenizer, prompt: str, args) -> tuple:
    """Generate Chain-of-Thought response using PQCache"""
    try:
        # Tokenize input directly (R-KV style - no chat template needed)
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=args.max_seq_length
        )
        
        # Device mapping
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            input_ids = inputs["input_ids"].to("cuda:0")
            attention_mask = inputs["attention_mask"].to("cuda:0")
        else:
            input_ids = inputs["input_ids"].to(f"cuda:{args.device_id}")
            attention_mask = inputs["attention_mask"].to(f"cuda:{args.device_id}")
        
        input_length = input_ids.shape[1]
        start_time = time.time()
        
        # Generate with CoT-specific parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1  # Prevent repetition in long reasoning
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_length = len(generated_tokens)
        
        return response, generation_time, input_length, output_length
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "", 0.0, 0, 0

def extract_final_answer(response: str, dataset_name: str) -> str:
    """Extract final numerical answer from CoT response"""
    
    # Prioritize patterns - boxed format first (most reliable)
    patterns = [
        r"\\boxed\{([^}]+)\}",  # LaTeX boxed format (highest priority)
        r"(?:final answer|answer|solution).*?(?:is|:|=)\s*([+-]?\d+(?:\.\d+)?)",
        r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*([+-]?\d+(?:\.\d+)?)",
        r"\$([+-]?\d+(?:\.\d+)?)\$",  # LaTeX math format
        r"([+-]?\d+(?:\.\d+)?)(?:\s*$|\s*\.?\s*$)"  # Number at end
    ]
    
    for pattern in patterns:
        # Don't convert to lowercase for boxed pattern to preserve content
        if "boxed" in pattern:
            matches = re.findall(pattern, response)
        else:
            matches = re.findall(pattern, response.lower(), re.IGNORECASE)
        
        if matches:
            answer = matches[-1].strip()  # Return last match, remove whitespace
            # For boxed answers, try to extract just the number if it's simple
            if "boxed" in pattern and answer.isdigit():
                return answer
            elif "boxed" in pattern:
                # Try to extract number from more complex boxed content
                number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
                if number_match:
                    return number_match.group(1)
                else:
                    return answer  # Return as-is if can't extract number
            else:
                return answer
    
    return ""

def evaluate_math_cot(args):
    """Main evaluation function for mathematics CoT reasoning"""
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    logger.info("Starting PQCache Mathematics CoT Evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load model and dataset
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_math_dataset(args.dataset, args.num_eval_examples)
    
    # Prepare output directory
    output_dir = Path(args.output_dir) / args.model / args.dataset / "pqcache_cot"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"cot_results_compress_{args.compress_ratio}_ratio_{args.important_ratio}_{args.recent_ratio}.jsonl"
    incremental_file = output_dir / f"incremental_cot_results_compress_{args.compress_ratio}_ratio_{args.important_ratio}_{args.recent_ratio}.jsonl"
    
    # Clear incremental file if it exists
    if incremental_file.exists():
        incremental_file.unlink()
    
    logger.info(f"增量保存文件: {incremental_file}")
    logger.info(f"最终保存文件: {output_file}")
    
    # Evaluation loop
    results = []
    correct_count = 0
    total_time = 0
    total_input_length = 0
    total_output_length = 0
    
    logger.info(f"Starting evaluation on {len(dataset)} examples")
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {args.dataset}")):
        try:
            # Create CoT prompt (R-KV style)
            prompt = create_cot_prompt(sample, args.dataset)
            
            # Generate response
            response, gen_time, input_len, output_len = generate_cot_response(
                model, tokenizer, prompt, args
            )
            
            # Extract predicted answer
            predicted_answer = extract_final_answer(response, args.dataset)
            
            # Get ground truth answer
            ground_truth = sample.get("answer", sample.get("solution", ""))
            if isinstance(ground_truth, str):
                # Try to extract numerical answer from ground truth
                gt_answer = extract_final_answer(ground_truth, args.dataset)
                if not gt_answer:
                    gt_answer = ground_truth.strip()
            else:
                gt_answer = str(ground_truth)
            
            # Simple accuracy check (can be improved with more sophisticated matching)
            is_correct = predicted_answer.strip() == gt_answer.strip() if predicted_answer else False
            if is_correct:
                correct_count += 1
            
            # Prepare result record
            result = {
                "id": i,
                "problem": sample.get("problem", sample.get("question", "")),
                "ground_truth": gt_answer,
                "predicted_answer": predicted_answer,
                "full_response": response,
                "is_correct": is_correct,
                "generation_time": gen_time,
                "input_length": input_len,
                "output_length": output_len
            }
            
            # Add dataset-specific fields
            if args.dataset == "math_500":
                result["level"] = sample.get("level", "")
                result["type"] = sample.get("type", "")
            
            results.append(result)
            
            # 增量保存：每完成一个样本就立即保存
            with open(incremental_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # Update statistics
            total_time += gen_time
            total_input_length += input_len
            total_output_length += output_len
            
            # Progress logging
            if (i + 1) % 10 == 0 or args.verbose:
                current_accuracy = correct_count / (i + 1)
                logger.info(f"✅ 已完成 {i + 1}/{len(dataset)} 样本 | 当前准确率: {current_accuracy:.3f}")
            
            if args.verbose and i < 5:  # Log first 5 examples
                logger.info(f"Example {i+1}:")
                logger.info(f"  Problem: {sample.get('problem', '')[:100]}...")
                logger.info(f"  Predicted: {predicted_answer}")
                logger.info(f"  Ground Truth: {gt_answer}")
                logger.info(f"  Correct: {is_correct}")
                logger.info(f"  Generation time: {gen_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 即使出错也要记录到增量文件
            error_result = {
                "id": i,
                "problem": sample.get("problem", sample.get("question", "")) if 'sample' in locals() else "",
                "ground_truth": "",
                "predicted_answer": "",
                "full_response": "",
                "is_correct": False,
                "generation_time": 0,
                "input_length": 0,
                "output_length": 0,
                "error": str(e)
            }
            
            if args.dataset == "math_500":
                error_result["level"] = sample.get("level", "") if 'sample' in locals() else ""
                error_result["type"] = sample.get("type", "") if 'sample' in locals() else ""
            
            results.append(error_result)
            
            with open(incremental_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
            
            continue
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Calculate and log final statistics
    accuracy = correct_count / len(results) if results else 0
    avg_time = total_time / len(results) if results else 0
    avg_input_length = total_input_length / len(results) if results else 0
    avg_output_length = total_output_length / len(results) if results else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PQCache Mathematics CoT Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Total examples: {len(results)}")
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Average generation time: {avg_time:.2f}s")
    logger.info(f"Average input length: {avg_input_length:.0f} tokens")
    logger.info(f"Average output length: {avg_output_length:.0f} tokens")
    logger.info(f"PQCache compression ratio: {args.compress_ratio}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Incremental results saved to: {incremental_file}")
    logger.info(f"{'='*60}")
    
    return results

def main():
    args = parse_args()
    results = evaluate_math_cot(args)
    return results

if __name__ == "__main__":
    main()
