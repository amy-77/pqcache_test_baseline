#!/usr/bin/env python3
"""
Full-Attention AIME 2024数学推理评估脚本
基于原始的run_math_cot.py，但修改为使用full-attention（不压缩KV cache）
用于对比PQCache和Full-Attention的性能差异
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
import traceback

# Add PQCache imports
sys.path.append('/home/pai/data/PQCache')
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.retrieval_based.pq_search import initialize_objects
from transformers import AutoTokenizer, AutoConfig

# R-KV style prompt template
R_KV_PROMPT_TEMPLATE = "You are given a math problem.\n\nProblem: {problem}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"

# Dataset key mappings
DATASET_KEYS = {
    "aime_2024": ["problem", "answer"],
    "math_500": ["problem", "answer"],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Full-Attention Mathematics CoT Reasoning Evaluation")
    
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
    
    # Method configuration - 重点修改这里
    parser.add_argument("--method", type=str, default="fullattention",
                        choices=["fullattention", "pqcache", "h2o"], 
                        help="Attention method")
    parser.add_argument("--compressor", type=str, default="original",
                        choices=["original", "pq_search", "h2o"], 
                        help="Compressor type (original=full-attention)")
    
    # Hardware and output
    parser.add_argument('--device_id', type=int, default=0,
                        help="CUDA device ID")
    parser.add_argument('--output_dir', type=str, default='math_cot_results',
                        help="Output directory for results")
    parser.add_argument('--verbose', action='store_true',
                        help="Verbose output")
    
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load model and tokenizer with full-attention configuration"""
    
    model2path = {
        "llama-3.1": "meta-llama/Llama-3.1-8B-Instruct"
    }
    
    model_name_or_path = model2path[args.model]
    device = torch.device(f"cuda:{args.device_id}")
    
    logger.info(f"Loading model from {model_name_or_path}")
    logger.info(f"使用方法: {args.method} (compressor: {args.compressor})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Fix RoPE scaling issues
    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
        if 'type' not in config.rope_scaling and 'rope_type' not in config.rope_scaling:
            config.rope_scaling['type'] = 'linear'
        elif 'type' not in config.rope_scaling and 'rope_type' in config.rope_scaling:
            config.rope_scaling['type'] = config.rope_scaling['rope_type']
    
    # 根据method配置模型
    if args.method == "fullattention":
        # Full-Attention配置：禁用压缩
        config.compress_ratio = 1.0  # 不压缩
        config.compressor = "original"  # 使用原始attention
        config.enable_vq_cache = False  # 禁用VQ cache
        
        # 设置必要的配置属性（即使不使用）
        config.important_ratio = 0.5
        config.recent_ratio = 0.5
        config.pp_size = 1
        config.sink_size = 16
        config.keyformer_mode = False
        config.drop_ratio = 0.0
        config.preserve_layer = 0
        config.score_func = "sum"
        config.threshold = 1.0
        config.n_subvec_per_head = 2
        config.n_subbits = 6
        config.topr = 32
        config.gqa = True
        config.max_iter = 0
        
        logger.info("🔄 配置为Full-Attention模式（无压缩）")
        
        # 加载标准的VQLlama31ForCausalLM，但配置为不压缩
        model = VQLlama31ForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True
        )
        
    elif args.method == "pqcache":
        # PQCache配置
        config.compress_ratio = 0.1
        config.important_ratio = 0.5
        config.recent_ratio = 0.5
        config.compressor = "pq_search"
        config.enable_vq_cache = True
        
        # 设置必要的配置属性
        config.pp_size = 1
        config.sink_size = 16
        config.keyformer_mode = False
        config.drop_ratio = 0.0
        config.preserve_layer = 0
        config.score_func = "sum"
        config.threshold = 1.0
        config.n_subvec_per_head = 2
        config.n_subbits = 6
        config.topr = 32
        config.gqa = True
        config.max_iter = 0
        
        logger.info("🗜️ 配置为PQCache模式")
        
        # Initialize PQCache objects
        config.max_seq_len = args.max_seq_length
        config.cache_block_size = 128
        config.global_cache_size = 4096
        config.cache_topk = 32
        config.device = device
        initialize_objects(config, model=args.model)
        
        model = VQLlama31ForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True
        )
        
    elif args.method == "h2o":
        # H2O配置
        config.compress_ratio = 0.1
        config.compressor = "h2o"
        config.enable_vq_cache = False
        
        # 设置必要的配置属性
        config.important_ratio = 0.5
        config.recent_ratio = 0.5
        config.pp_size = 1
        config.sink_size = 16
        config.keyformer_mode = False
        config.drop_ratio = 0.0
        config.preserve_layer = 0
        config.score_func = "sum"
        config.threshold = 1.0
        config.n_subvec_per_head = 2
        config.n_subbits = 6
        config.topr = 32
        config.gqa = True
        config.max_iter = 0
        
        logger.info("💧 配置为H2O模式")
        
        from h2o_method.h2o_attention import H2OLlamaForCausalLM
        model = H2OLlamaForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    
    # Device mapping
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        model = model.half().eval().to("cuda:0")
    else:
        model = model.half().eval().to(device)
    
    logger.info(f"模型加载成功，使用方法: {args.method}")
    return model, tokenizer

def load_math_dataset(dataset_name: str, num_examples: int = -1):
    """Load mathematics dataset from local JSONL files"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    dataset_files = {
        "aime_2024": "/home/pai/data/PQCache/data/cot_math/aime_2024_train.jsonl",
        "math_500": "/home/pai/data/PQCache/data/cot_math/math_500_test.jsonl"
    }
    
    if dataset_name not in dataset_files:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataset_path = dataset_files[dataset_name]
    logger.info(f"Loading from: {dataset_path}")
    
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
    problem_key = DATASET_KEYS[dataset_name][0]
    problem = sample.get(problem_key, sample.get("problem", sample.get("question", "")))
    prompt = R_KV_PROMPT_TEMPLATE.format(problem=problem)
    return prompt

def generate_cot_response(model, tokenizer, prompt: str, args) -> tuple:
    """Generate Chain-of-Thought response"""
    try:
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
                repetition_penalty=1.1
            )
        
        generation_time = time.time() - start_time
        
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
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"(?:final answer|answer|solution).*?(?:is|:|=)\s*([+-]?\d+(?:\.\d+)?)",
        r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*([+-]?\d+(?:\.\d+)?)",
        r"\$([+-]?\d+(?:\.\d+)?)\$",
        r"([+-]?\d+(?:\.\d+)?)(?:\s*$|\s*\.?\s*$)"
    ]
    
    for pattern in patterns:
        if "boxed" in pattern:
            matches = re.findall(pattern, response)
        else:
            matches = re.findall(pattern, response.lower(), re.IGNORECASE)
        
        if matches:
            answer = matches[-1].strip()
            if "boxed" in pattern and answer.isdigit():
                return answer
            elif "boxed" in pattern:
                number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
                if number_match:
                    return number_match.group(1)
                else:
                    return answer
            else:
                return answer
    
    return ""

def evaluate_math_cot(args):
    """Main evaluation function for mathematics CoT reasoning"""
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    logger.info(f"Starting {args.method.upper()} Mathematics CoT Evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load model and dataset
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_math_dataset(args.dataset, args.num_eval_examples)
    
    # Prepare output directory
    method_name = args.method
    output_dir = Path(args.output_dir) / args.model / args.dataset / f"{method_name}_cot"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"cot_results_{method_name}.jsonl"
    incremental_file = output_dir / f"incremental_cot_results_{method_name}.jsonl"
    
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
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {args.dataset} ({method_name})")):
        try:
            prompt = create_cot_prompt(sample, args.dataset)
            
            response, gen_time, input_len, output_len = generate_cot_response(
                model, tokenizer, prompt, args
            )
            
            predicted_answer = extract_final_answer(response, args.dataset)
            
            ground_truth = sample.get("answer", sample.get("solution", ""))
            if isinstance(ground_truth, str):
                gt_answer = extract_final_answer(ground_truth, args.dataset)
                if not gt_answer:
                    gt_answer = ground_truth.strip()
            else:
                gt_answer = str(ground_truth)
            
            is_correct = predicted_answer.strip() == gt_answer.strip() if predicted_answer else False
            if is_correct:
                correct_count += 1
            
            result = {
                "id": i,
                "problem": sample.get("problem", sample.get("question", "")),
                "ground_truth": gt_answer,
                "predicted_answer": predicted_answer,
                "full_response": response,
                "is_correct": is_correct,
                "generation_time": gen_time,
                "input_length": input_len,
                "output_length": output_len,
                "method": method_name
            }
            
            if args.dataset == "math_500":
                result["level"] = sample.get("level", "")
                result["type"] = sample.get("type", "")
            
            results.append(result)
            
            # 增量保存
            with open(incremental_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            total_time += gen_time
            total_input_length += input_len
            total_output_length += output_len
            
            if (i + 1) % 10 == 0 or args.verbose:
                current_accuracy = correct_count / (i + 1)
                logger.info(f"✅ 已完成 {i + 1}/{len(dataset)} 样本 | 当前准确率: {current_accuracy:.3f}")
            
            if args.verbose and i < 3:
                logger.info(f"Example {i+1}:")
                logger.info(f"  Problem: {sample.get('problem', '')[:100]}...")
                logger.info(f"  Predicted: {predicted_answer}")
                logger.info(f"  Ground Truth: {gt_answer}")
                logger.info(f"  Correct: {is_correct}")
                logger.info(f"  Generation time: {gen_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
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
                "method": method_name,
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
    logger.info(f"{method_name.upper()} Mathematics CoT Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Method: {method_name}")
    logger.info(f"Total examples: {len(results)}")
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Average generation time: {avg_time:.2f}s")
    logger.info(f"Average input length: {avg_input_length:.0f} tokens")
    logger.info(f"Average output length: {avg_output_length:.0f} tokens")
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
