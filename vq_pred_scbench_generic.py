#!/usr/bin/env python3
"""
Generic SCBench evaluation script for PQCache
Supports all SCBench datasets with proper task categorization
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import torch
from loguru import logger
from datasets import load_dataset

# Add PQCache imports
sys.path.append('/home/pai/data/PQCache')
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.retrieval_based.pq_search import initialize_objects
from transformers import AutoTokenizer, AutoConfig

# Task categories based on SCBench paper analysis
TASK_CATEGORIES = {
    # 长生成导致解码内注意力迁移 (Long generation causes intra-decode attention shift)
    "long_generation_decode_shift": [
        "scbench_summary",           # En.Sum - English summarization
        "scbench_summary_with_needles"  # Mix.Sum+NIAH - Mixed summarization with needle-in-haystack
    ],
    
    # 强漂移 (Strong drift)
    "strong_drift": [
        "scbench_qa_eng",           # En.QA - English QA
        "scbench_qa_chn",           # Zh.QA - Chinese QA  
        "scbench_choice_eng",       # En.MultiChoice - English multiple choice
        "scbench_mf"                # Math.Find - Mathematical finding tasks
    ],
    
    # 多轮/多请求下"关注KV"漂移最明显 (Multi-turn/multi-request KV attention drift)
    "multiturn_kv_drift": [
        "scbench_kv",               # Retr.KV - Key-value retrieval
        "scbench_prefix_suffix",    # Retr.Prefix-Suffix - Prefix-suffix retrieval
        "scbench_vt"                # Retr.MultiHop - Multi-hop variable tracing
    ],
    
    # 其他重要任务 (Other important tasks)
    "other_tasks": [
        # "scbench_repoqa",           # Code retrieval
        # "scbench_repoqa_and_kv",    # Mixed code+KV retrieval  
        "scbench_many_shot"         # In-context learning
    ]
}

# Flatten all tasks for easy access
ALL_TASKS = []
for category_tasks in TASK_CATEGORIES.values():
    ALL_TASKS.extend(category_tasks)

# Max new tokens per task (from SCBench official eval_utils.py)
# These represent the maximum number of tokens to generate during decode phase
# Based on: https://huggingface.co/datasets/microsoft/SCBench

DATA_NAME_TO_MAX_NEW_TOKENS = {
    # String Retrieval Tasks
    "scbench_kv": 150,              # Retr.KV - Key-value lookup (100 samples)
    "scbench_prefix_suffix": 150,   # Retr.Prefix-Suffix - String pattern matching (100 samples)
    "scbench_vt": 30,               # Retr.MultiHop - Variable tracing (90 samples)
    
    # Semantic Retrieval Tasks  
    "scbench_qa_eng": 40,           # English QA on long texts (69 samples)
    "scbench_qa_chn": 40,           # Chinese QA on long texts (35 samples)
    "scbench_choice_eng": 40,       # English multiple choice (58 samples)
    
    # Global Information Processing Tasks
    "scbench_many_shot": 10,        # Many-shot ICL (54 samples)
    "scbench_mf": 5,                # Math.Find - Statistical tasks (100 samples)
    "scbench_summary": 200,         # En.Sum - Document summarization (70 samples)
    
    # Multi-Tasking
    "scbench_summary_with_needles": {"scbench_summary": 800, "scbench_passkey": 15},  # Mix.Sum+NIAH (70 samples)
    
    # Code tasks (temporarily disabled for initial testing)
    # "scbench_repoqa": 1024,         # Code.RepoQA - Function retrieval (88 samples)
    # "scbench_repoqa_and_kv": {"scbench_repoqa": 1024, "scbench_kv": 80},  # Mix.RepoQA+KV (88 samples)
}

# Note: Dataset-specific information will be automatically detected from the actual data
# SCBench datasets have variable numbers of turns per sample (typically 2-5 turns)
# Sample counts and turn numbers are determined dynamically when loading the dataset

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    compressor_choices = ["h2o", "original", "no_drop_lb", "pq_search","sparq_f"]
    parser.add_argument('--model', type=str, default="llama-3.1", choices=[
        "llama-7b", "llama2-7b-chat-4k", "llama2-7b-32K", "mistral-7b-Instruct-32k", "llama-3.1","longchat-v1.5-7b-32k",
        "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    
    # SCBench specific arguments
    parser.add_argument('--task', type=str, default='scbench_prefix_suffix', 
                        choices=ALL_TASKS + ["all"] + list(TASK_CATEGORIES.keys()))
    parser.add_argument('--task_category', type=str, default=None, 
                        choices=list(TASK_CATEGORIES.keys()),
                        help='Task category to evaluate (overrides --task)')
    parser.add_argument("--compress_ratio", type=float, default=0.1)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    parser.add_argument("--recent_ratio", type=float, default=0.5)
    parser.add_argument('--enable_vq_cache', action='store_true')
    parser.add_argument('--enable_h2o_cache', action='store_true')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sink-size", type=int, default=16)
    parser.add_argument("--keyformer_mode",type=int, default=0)
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument("--exp_name", type=str, default="scbench_generic")
    parser.add_argument("--preserve_layer", type=int, default=0)
    parser.add_argument("--score_func", type=str, default="sum")
    parser.add_argument("--compressor", type=str, default="pq_search", choices=compressor_choices)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--n_subvec_per_head", type=int, default=2)
    parser.add_argument("--n_subbits", type=int, default=6)
    parser.add_argument("--topr", type=int, default=32)
    parser.add_argument("--recent_size", type=int, default=32)
    parser.add_argument("--gqa", type=str, default="True")
    parser.add_argument("--sparq_mean_v_trick", type=str, default="False")
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision")
    parser.add_argument('--dp', action='store_true', help='whether data parallel')
    parser.add_argument('--pp-size', type=int, default=1, choices=[1,2,4,8])
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--enable_neff_analysis', action='store_true')
    parser.add_argument('--neff_samples', type=int, default=None)
    
    # SCBench specific arguments
    parser.add_argument('--max_seq_length', type=int, default=131072)
    parser.add_argument('--max_new_tokens', type=int, default=150)  # prefix_suffix默认值
    parser.add_argument('--num_eval_examples', type=int, default=-1)
    parser.add_argument('--max_turns', type=int, default=-1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='pred_scbench_prefix_suffix')
    parser.add_argument('--disable_golden_context', action='store_true', 
                        help='Use generated answers instead of ground truth answers for multi-turn')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args(args)



def load_model_and_tokenizer(args, model2path, model_name, device, pp_size):
    """Load VQLlama31ForCausalLM model and tokenizer (adapted from vq_pred_scbench_official.py)"""
    
    # Get model path from model2path mapping
    model_name_or_path = model2path[model_name]
    logger.info(f"Loading model from {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_name_or_path)
    
    # 修复RoPE配置问题
    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
        if 'type' not in config.rope_scaling and 'rope_type' not in config.rope_scaling:
            config.rope_scaling['type'] = 'linear'  # 添加默认type
        elif 'type' not in config.rope_scaling and 'rope_type' in config.rope_scaling:
            config.rope_scaling['type'] = config.rope_scaling['rope_type']
    
    # 设置PQCache相关配置参数到config对象
    config.compress_ratio = args.compress_ratio
    config.important_ratio = args.important_ratio
    config.recent_ratio = args.recent_ratio
    config.pp_size = getattr(args, 'pp_size', pp_size)
    config.sink_size = getattr(args, 'sink_size', 16)
    config.keyformer_mode = bool(args.keyformer_mode)
    config.drop_ratio = args.drop_ratio
    config.preserve_layer = args.preserve_layer
    config.score_func = args.score_func
    config.compressor = args.compressor
    config.threshold = args.threshold
    config.n_subvec_per_head = args.n_subvec_per_head
    config.n_subbits = args.n_subbits
    config.topr = args.topr
    config.gqa = args.gqa.lower() == 'true'
    config.max_iter = args.max_iter
    config.device = device
    
    # 设置缓存配置
    if args.enable_vq_cache:
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
    elif args.enable_h2o_cache:
        config.hh_ratio = args.important_ratio
    
    # Initialize PQCache objects if using pq_search compressor
    if config.compressor == "pq_search":
        config.max_seq_len = args.max_seq_length
        config.cache_block_size = 128
        config.global_cache_size = 4096
        config.cache_topk = 32
        initialize_objects(config, model=args.model)
    
    # 根据启用的缓存类型加载相应的模型
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
        # 如果都没启用，默认使用VQLlama31ForCausalLM
        model = VQLlama31ForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True
        )
    
    # 当使用CUDA_VISIBLE_DEVICES时，设备映射为cuda:0
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        model = model.half().eval().to("cuda:0")
    else:
        model = model.half().eval().to(device)
    logger.info("Model loaded successfully")
    return model, tokenizer




# Official SCBench prompt templates (from eval_utils.py)
MULTITURN_TEMPLATES = {
    "scbench_passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}",
    "scbench_kv": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",
    "scbench_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe the correct answer is",
    "scbench_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    "scbench_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",
    "scbench_mf": "{prefix}\n\n{context}\n\n{input}",
    "scbench_summary": "{context}\n\n{input}",
    "scbench_vt": "{context}\n\n{input}",
    "scbench_many_shot": "{context}\n\n{input}",
    "scbench_summary_with_needles": "{context}\n\n{input}",
    "scbench_prefix_suffix": "{context}\n\n{input}",
}

MULTITURN_FOLLOW_UP_TEMPLATES = {
    "scbench_passkey": "{pre_ans}.\n\n{input}",
    "scbench_kv": "{pre_ans}\n\n{input}",
    "scbench_choice_eng": "{pre_ans}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
    "scbench_qa_eng": "{pre_ans}\n\nQuestion: {question}\nAnswer:",
    "scbench_qa_chn": "{pre_ans}\n\n问题：{question}\n答案：",
    "scbench_mf": "{pre_ans}\n\n{prefix}\n\n{input}",
    "scbench_summary": "{pre_ans}\n\n{input}",
    "scbench_vt": "{pre_ans}\n\n{input}",
    "scbench_many_shot": "{pre_ans}\n\n{input}",
    "scbench_summary_with_needles": "{pre_ans}\n\n{input}",
    "scbench_prefix_suffix": "{pre_ans}\n\n{input}",
}

def create_multiturn_prompt(eg: Dict[str, Any], data_name: str, disable_golden_context: bool = False) -> Dict[str, Any]:
    """
    Create multi-turn prompt for SCBench evaluation (adapted from official eval_utils.py)
    """
    template = MULTITURN_TEMPLATES.get(data_name)
    follow_up_template = MULTITURN_FOLLOW_UP_TEMPLATES.get(data_name)
    
    if not template or not follow_up_template:
        raise ValueError(f"Unsupported task: {data_name}")
    
    context = eg.get("context", "")
    multi_turns = eg.get("multi_turns", [])
    
    if not multi_turns:
        raise ValueError(f"No multi_turns found in sample {eg.get('id', 'unknown')}")
    
    if data_name == "scbench_choice_eng":
        first_turn = multi_turns[0]
        options = first_turn.get("options", [])
        
        first_turn_prompt = template.format(
            context=context,
            question=first_turn["input"],
            OPTION_A=options[0] if len(options) > 0 else "A",
            OPTION_B=options[1] if len(options) > 1 else "B", 
            OPTION_C=options[2] if len(options) > 2 else "C",
            OPTION_D=options[3] if len(options) > 3 else "D",
        )
        
        follow_up_prompts = []
        for i in range(1, len(multi_turns)):
            if disable_golden_context:
                pre_ans = None
            else:
                pre_ans = multi_turns[i - 1]["answer"]
            
            follow_up_prompt = follow_up_template.format(
                pre_ans=pre_ans if pre_ans is not None else "",
                question=multi_turns[i]["input"],
                OPTION_A=multi_turns[i]["options"][0],
                OPTION_B=multi_turns[i]["options"][1],
                OPTION_C=multi_turns[i]["options"][2],
                OPTION_D=multi_turns[i]["options"][3],
            )
            follow_up_prompts.append(follow_up_prompt)
        
        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [turn["answer"] for turn in multi_turns],
            "options": options,
        }
    
    elif data_name in ["scbench_qa_eng", "scbench_qa_chn"]:
        first_turn = multi_turns[0]
        
        first_turn_prompt = template.format(
            context=context,
            question=first_turn["input"],
        )
        
        follow_up_prompts = []
        for i in range(1, len(multi_turns)):
            if disable_golden_context:
                pre_ans = None
            else:
                pre_ans = multi_turns[i - 1]["answer"]
            
            follow_up_prompt = follow_up_template.format(
                pre_ans=pre_ans if pre_ans is not None else "",
                question=multi_turns[i]["input"],
            )
            follow_up_prompts.append(follow_up_prompt)
        
        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [turn["answer"] for turn in multi_turns],
        }
    
    elif data_name == "scbench_mf":
        first_turn = multi_turns[0]
        prefix = first_turn.get("prefix", "")
        
        first_turn_prompt = template.format(
            prefix=prefix,
            context=context,
            input=first_turn["input"],
        )
        
        follow_up_prompts = []
        for i in range(1, len(multi_turns)):
            if disable_golden_context:
                pre_ans = None
            else:
                pre_ans = multi_turns[i - 1]["answer"]
            
            follow_up_prompt = follow_up_template.format(
                pre_ans=pre_ans if pre_ans is not None else "",
                prefix=multi_turns[i].get("prefix", ""),
                input=multi_turns[i]["input"],
            )
            follow_up_prompts.append(follow_up_prompt)
        
        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [turn["answer"] for turn in multi_turns],
        }
    
    else:
        # Generic template for other tasks
        first_turn = multi_turns[0]
        
        first_turn_prompt = template.format(
            context=context,
            input=first_turn["input"],
        )
        
        follow_up_prompts = []
        for i in range(1, len(multi_turns)):
            if disable_golden_context:
                pre_ans = None
            else:
                pre_ans = multi_turns[i - 1]["answer"]
            
            follow_up_prompt = follow_up_template.format(
                pre_ans=pre_ans if pre_ans is not None else "",
                input=multi_turns[i]["input"],
            )
            follow_up_prompts.append(follow_up_prompt)
        
        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [turn["answer"] for turn in multi_turns],
        }



def generate_response(model, tokenizer, prompt: str, max_new_tokens: int, device_id: int) -> tuple:
    """Generate response using PQCache model"""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100000)
        # 当使用CUDA_VISIBLE_DEVICES时，使用cuda:0
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            input_ids = inputs["input_ids"].to("cuda:0")
            attention_mask = inputs["attention_mask"].to("cuda:0")
        else:
            input_ids = inputs["input_ids"].to(f"cuda:{device_id}")
            attention_mask = inputs["attention_mask"].to(f"cuda:{device_id}")
        
        input_length = input_ids.shape[1]
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        generation_time = time.time() - start_time
        
        # Decode output
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_length = len(generated_tokens)
        
        return response, generation_time, input_length, output_length
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "", 0.0, 0, 0




def evaluate_single_task(args, task_name: str):
    """Evaluate a single SCBench task with incremental saving"""
    logger.info(f"Evaluating task: {task_name}")
    
    # Dataset info will be detected from actual data
    logger.info(f"Task: {task_name} - info will be detected from loaded dataset")
    
    # Load model using same method as vq_pred_scbench_official.py
    model2path = json.load(open("config/model2path.json", "r"))
    device = torch.device(f"cuda:{args.device_id}")
    model, tokenizer = load_model_and_tokenizer(args, model2path, args.model, device, getattr(args, 'pp_size', 1))
    
    # Load dataset from local JSONL files
    try:
        # Map task names to local file paths
        local_dataset_files = {
            "scbench_summary": "/home/pai/data/PQCache/data/scbench/scbench_summary.jsonl",
            "scbench_summary_with_needles": "/home/pai/data/PQCache/data/scbench/scbench_summary_with_needles.jsonl",
            "scbench_qa_eng": "/home/pai/data/PQCache/data/scbench/scbench_qa_eng.jsonl",
            "scbench_qa_chn": "/home/pai/data/PQCache/data/scbench/scbench_qa_chn.jsonl",
            "scbench_choice_eng": "/home/pai/data/PQCache/data/scbench/scbench_choice_eng.jsonl",
            "scbench_mf": "/home/pai/data/PQCache/data/scbench/scbench_mf.jsonl",
            "scbench_many_shot": "/home/pai/data/PQCache/data/scbench/scbench_many_shot.jsonl",
            "scbench_kv": "/home/pai/data/PQCache/data/scbench/scbench_kv.jsonl",
            "scbench_prefix_suffix": "/home/pai/data/PQCache/data/scbench/scbench_prefix_suffix.jsonl",
            "scbench_vt": "/home/pai/data/PQCache/data/scbench/scbench_vt.jsonl",
        }
        
        # Special handling for JSON files
        json_files = {
            "scbench_repoqa": "/home/pai/data/PQCache/data/scbench/scbench_repoqa.json",
            "scbench_repoqa_and_kv": "/home/pai/data/PQCache/data/scbench/scbench_repoqa_and_kv.json",
        }
        
        dataset_path = None
        if task_name in local_dataset_files:
            dataset_path = local_dataset_files[task_name]
        elif task_name in json_files:
            dataset_path = json_files[task_name]
        else:
            logger.error(f"Unknown task: {task_name}")
            return
            
        logger.info(f"Loading dataset from local file: {dataset_path}")
        
        # Load data based on file extension
        dataset = []
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dataset.append(json.loads(line))
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle different JSON structures
                if isinstance(data, list):
                    dataset = data
                elif isinstance(data, dict) and 'data' in data:
                    dataset = data['data']
                else:
                    dataset = [data]
        
        logger.info(f"Loaded {len(dataset)} samples for {task_name}")
        
        # Log actual dataset characteristics
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            if 'multi_turns' in sample:
                logger.info(f"Actual turns per sample: {len(sample.get('multi_turns', []))}")
            if 'context' in sample:
                context_len = len(sample.get('context', ''))
                logger.info(f"Context length (chars): {context_len:,}")
            
    except Exception as e:
        logger.error(f"Failed to load dataset {task_name}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return
    
    # Limit number of examples based on dataset size and user request
    total_samples = len(dataset)
    if args.num_eval_examples > 0:
        num_to_eval = min(args.num_eval_examples, total_samples)
    else:
        num_to_eval = total_samples
    
    # Use list slicing instead of dataset.select (which doesn't exist for regular lists)
    dataset = dataset[:num_to_eval]
    logger.info(f"Will evaluate {num_to_eval} out of {total_samples} samples")
    
    # Get max_new_tokens for this task
    max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS.get(task_name, 100)
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    if isinstance(max_new_tokens, dict):
        # For mixed tasks like summary_with_needles, use first task's setting for simplicity
        max_new_tokens = list(max_new_tokens.values())[0]
    
    logger.info(f"Using max_new_tokens: {max_new_tokens} (decode phase generation limit)")
    
    # Prepare output directory
    output_dir = Path(args.output_dir) / "llama-3.1" / task_name / "pqcache_official"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"compress_{args.compress_ratio}_important_{args.important_ratio}_recent_{args.recent_ratio}_subvec_{args.n_subvec_per_head}_subbits_{args.n_subbits}.jsonl"
    
    # Create incremental save file path
    incremental_file = output_dir / f"incremental_compress_{args.compress_ratio}_important_{args.important_ratio}_recent_{args.recent_ratio}_subvec_{args.n_subvec_per_head}_subbits_{args.n_subbits}.jsonl"
    
    # Clear incremental file if it exists
    if incremental_file.exists():
        incremental_file.unlink()
    
    logger.info(f"增量保存文件: {incremental_file}")
    logger.info(f"最终保存文件: {output_file}")
    
    # Evaluate samples
    results = []
    total_time = 0
    total_input_length = 0
    total_output_length = 0
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {task_name}")):
        try:
            # Create multi-turn prompts using official SCBench method
            prompt_data = create_multiturn_prompt(sample, task_name, args.disable_golden_context)
            prompts = prompt_data["prompts"]
            ground_truths = prompt_data["ground_truth"]
            
            # Limit number of turns if specified
            if args.max_turns > 0:
                num_turns = min(args.max_turns, len(prompts))
                prompts = prompts[:num_turns]
                ground_truths = ground_truths[:num_turns]
            else:
                num_turns = len(prompts)
            
            # Process each turn for this sample
            sample_results = []
            for turn_idx, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
                # Generate response
                response, gen_time, input_len, output_len = generate_response(
                    model, tokenizer, prompt, max_new_tokens, args.device_id
                )
                
                # Prepare result record
                result = {
                    "id": i,
                    "turn_idx": turn_idx,
                    "prediction": response,
                    "ground_truth": ground_truth,
                    "generation_time": gen_time,
                    "input_length": input_len,
                    "output_length": output_len
                }
                
                # Add task-specific fields
                if task_name == "scbench_repoqa":
                    result["lang"] = sample.get("lang", "")
                    result["repo"] = sample.get("repo", "")
                    if turn_idx < len(sample["multi_turns"]):
                        result["func_name"] = sample["multi_turns"][turn_idx].get("name", "")
                elif task_name == "scbench_repoqa_and_kv":
                    result["lang"] = sample.get("lang", "")
                    result["repo"] = sample.get("repo", "")
                    if turn_idx < len(sample["multi_turns"]) and sample["multi_turns"][turn_idx].get("task") == "scbench_repoqa":
                        result["func_name"] = sample["multi_turns"][turn_idx].get("name", "")
                
                results.append(result)
                sample_results.append(result)
                
                # Update statistics
                total_time += gen_time
                total_input_length += input_len
                total_output_length += output_len
                
                if args.verbose and turn_idx < 2:  # Log first 2 turns
                    logger.info(f"Sample {i}, Turn {turn_idx + 1}: Generated {len(response)} chars in {gen_time:.2f}s")
                    logger.info(f"Preview: {response[:200]}...")
            
            # 增量保存：每完成一个样本就立即保存该样本的所有轮次
            with open(incremental_file, 'a', encoding='utf-8') as f:
                for result in sample_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logger.info(f"✅ 已保存样本 {i + 1}/{len(dataset)} 的结果到增量文件")
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            # 即使出错也要记录到增量文件
            error_result = {
                "id": i,
                "turn_idx": 0,
                "prediction": "",
                "ground_truth": "",
                "generation_time": 0,
                "input_length": 0,
                "output_length": 0,
                "error": str(e)
            }
            
            # Add task-specific fields for error cases
            if task_name in ["scbench_repoqa", "scbench_repoqa_and_kv"]:
                error_result["lang"] = sample.get("lang", "") if 'sample' in locals() else ""
                error_result["repo"] = sample.get("repo", "") if 'sample' in locals() else ""
                if task_name == "scbench_repoqa":
                    error_result["func_name"] = ""
            
            results.append(error_result)
            
            with open(incremental_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
            
            continue
    
    # Save final results (保持与原始格式兼容)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Log statistics
    num_turns = len(results)
    num_unique_samples = len(set([r["id"] for r in results]))
    if num_turns > 0:
        avg_time = total_time / num_turns
        avg_input_length = total_input_length / num_turns
        avg_output_length = total_output_length / num_turns
        avg_turns_per_sample = num_turns / num_unique_samples if num_unique_samples > 0 else 0
        
        logger.info(f"Task {task_name} completed:")
        logger.info(f"  Processed samples: {num_unique_samples}")
        logger.info(f"  Total turns: {num_turns}")
        logger.info(f"  Average turns per sample: {avg_turns_per_sample:.1f}")
        logger.info(f"  Average generation time per turn: {avg_time:.2f}s")
        logger.info(f"  Average input length per turn: {avg_input_length:.0f} tokens")
        logger.info(f"  Average output length per turn: {avg_output_length:.0f} tokens")
        logger.info(f"  Results saved to: {output_file}")
        logger.info(f"  Incremental results saved to: {incremental_file}")
    
    return results





def main():
    
    args = parse_args()
    
    # 确保用户选择了一种缓存方法
    if not (args.enable_vq_cache or args.enable_h2o_cache):
        logger.error("Must enable either --enable_vq_cache or --enable_h2o_cache")
        logger.info("推荐使用 --enable_vq_cache 启用PQCache压缩功能")
        return
    
    if args.enable_vq_cache and args.enable_h2o_cache:
        logger.error("Cannot enable both VQ cache and H2O cache simultaneously")
        return
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    logger.info("Starting SCBench Generic evaluation for PQCache")
    logger.info(f"Arguments: {vars(args)}")
    
    # Determine tasks to run
    tasks_to_run = []
    
    if args.task_category:
        tasks_to_run = TASK_CATEGORIES[args.task_category]
        logger.info(f"Running task category '{args.task_category}': {tasks_to_run}")
    elif args.task == "all":
        tasks_to_run = ALL_TASKS
        logger.info(f"Running all tasks: {tasks_to_run}")
    elif args.task in TASK_CATEGORIES:
        tasks_to_run = TASK_CATEGORIES[args.task]
        logger.info(f"Running task category '{args.task}': {tasks_to_run}")
    else:
        tasks_to_run = [args.task]
        logger.info(f"Running single task: {args.task}")
    
    # Run evaluation
    all_results = {}
    for task_name in tasks_to_run:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting evaluation for task: {task_name}")
            logger.info(f"{'='*50}")
            
            results = evaluate_single_task(args, task_name)
            all_results[task_name] = results
            
            # Clear GPU memory between tasks
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to evaluate task {task_name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            continue
    
    logger.info(f"\n{'='*50}")
    logger.info("SCBench Generic evaluation completed!")
    logger.info(f"Evaluated {len(all_results)} tasks")
    for task_name, results in all_results.items():
        logger.info(f"  {task_name}: {len(results) if results else 0} samples")
    logger.info(f"{'='*50}")



if __name__ == "__main__":
    main()
