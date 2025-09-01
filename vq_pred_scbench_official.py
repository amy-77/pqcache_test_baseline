import os
from datasets import load_dataset
import torch
import json
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse

# 导入PQCache相关模块
from vq_method.llama_patch import VQLlamaForCausalLM
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.mistral_patch import VQMistralForCausalLM
from h2o_method.h2o_attention import H2OLlamaForCausalLM, H2OLlamaAttention
from vq_method.retrieval_based.pq_search import initialize_objects

import torch.distributed as dist
import torch.multiprocessing as mp
import time
from loguru import logger


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    compressor_choices = ["h2o", "original", "no_drop_lb", "pq_search","sparq_f"]
    parser.add_argument('--model', type=str, default="llama-3.1", choices=[
        "llama-7b", "llama2-7b-chat-4k", "llama2-7b-32K", "mistral-7b-Instruct-32k", "llama-3.1","longchat-v1.5-7b-32k",
        "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    
    # SCBench specific arguments
    parser.add_argument('--task', type=str, default='scbench_repoqa', 
                        choices=['scbench_repoqa', 'scbench_repoqa_and_kv'])
    parser.add_argument("--compress_ratio", type=float, default=0.05)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    parser.add_argument("--recent_ratio", type=float, default=0.5)
    parser.add_argument('--enable_vq_cache', action='store_true')
    parser.add_argument('--enable_h2o_cache', action='store_true')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sink-size", type=int, default=16)
    parser.add_argument("--keyformer_mode",type=int, default=0)
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument("--exp_name", type=str, default="scbench_official")
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
    parser.add_argument('--max_seq_length', type=int, default=50000)
    parser.add_argument('--max_new_tokens', type=int, default=1024)  # 官方scbench_repoqa默认值
    parser.add_argument('--num_eval_examples', type=int, default=2)
    parser.add_argument('--max_turns', type=int, default=3)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results_scbench')
    parser.add_argument('--disable_golden_context', action='store_true', 
                        help='Use generated answers instead of ground truth answers for multi-turn')
    return parser.parse_args(args)



def create_multiturn_prompt(sample, max_turns=3, disable_golden_context=False):
    """
    基于官方create_multiturn_prompt的实现
    创建多轮对话的prompt列表
    """
    # 官方模板 - for scbench_repoqa
    template = "Based on the function description and code context, please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:\n\n{context}\n\n{input}"
    follow_up_template = "{pre_ans}\n\n{input}"
    
    context = sample["context"]
    multi_turns = sample["multi_turns"][:max_turns]
    # print(f"multi_turns: {multi_turns}")
    # 第一轮prompt
    first_turn = multi_turns[0]
    first_turn_prompt = template.format(
        context=context,
        input=first_turn["input"]
    )
    # print(f"first_turn_prompt: {first_turn_prompt}")
    # 后续轮次prompt
    follow_up_prompts = []
    for i in range(1, len(multi_turns)):
        if disable_golden_context:
            # 如果禁用金标准答案，使用None（在实际推理时会用生成的答案）
            pre_ans = None
        else:
            # 使用数据集中的标准答案
            pre_ans = multi_turns[i-1]["answer"]
        
        follow_up_prompt = follow_up_template.format(
            pre_ans=pre_ans if pre_ans is not None else "",  # 如果是None就用空字符串
            input=multi_turns[i]["input"]
        )
        follow_up_prompts.append(follow_up_prompt)
        
        # print(f"第{i}轮")
        # print(f"follow_up_prompts: {follow_up_prompts}")
    
    prompts = [first_turn_prompt] + follow_up_prompts
    
    ground_truth = []
    for i, turn in enumerate(multi_turns):
        # print(f"第{i}轮 answer: {turn['answer']}")
        ground_truth.append(turn["answer"])
    
    
    return {
        "prompts": prompts,
        "ground_truth": ground_truth,
        "sample_info": {
            "id": sample.get("id", -1),
            "repo": sample.get("repo", ""),
            "lang": sample.get("lang", ""),
        }
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 加载模型和分词器
def load_model_and_tokenizer(args, model2path, model_name, device, pp_size):
    """加载模型和tokenizer - 基于原始vq_pred.py"""
    path = model2path[model_name]
    
    if "llama" in model_name and "3." in model_name:
        config = AutoConfig.from_pretrained(path)
        
        # 修复RoPE配置问题
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if 'type' not in config.rope_scaling and 'rope_type' not in config.rope_scaling:
                config.rope_scaling['type'] = 'linear'
            elif 'type' not in config.rope_scaling and 'rope_type' in config.rope_scaling:
                config.rope_scaling['type'] = config.rope_scaling['rope_type']
        
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        sink_size = getattr(args, 'sink_size', None) or getattr(args, 'sink-size', 16)
        config.sink_size = sink_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device(f"cuda:{args.device_id}")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        config.enable_neff_analysis = args.enable_neff_analysis
        
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio
        
        if config.compressor == "pq_search":
            config.max_seq_len = args.max_seq_length
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model=model_name)
            
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if args.enable_vq_cache:
            #加载经过PQCache改造的Llama-3.1模型，（（压缩KV缓存）
            model = VQLlama31ForCausalLM.from_pretrained(path, config=config)
            
        elif args.enable_h2o_cache:
            model = H2OLlamaForCausalLM.from_pretrained(path, config=config)
        
        if args.fp16:
            model = model.half()
        model = model.eval()
        model = model.to(device)
        
    return model, tokenizer


def get_config_str_list(args):
    """生成配置字符串列表"""
    config_str_list = []
    if args.enable_vq_cache:
        config_str_list.append(f"compress_{args.compress_ratio}")
        config_str_list.append(f"important_{args.important_ratio}")
        config_str_list.append(f"recent_{args.recent_ratio}")
        if args.compressor == "pq_search":
            config_str_list.append(f"subvec_{args.n_subvec_per_head}")
            config_str_list.append(f"subbits_{args.n_subbits}")
    return config_str_list



def multiturn_test(args, model, tokenizer, encoded_example, max_new_tokens):
    """
    基于官方test函数的多轮推理实现
    """
    device = model.device
    results = []
    
    # 累积的input_ids，用于维护对话历史
    input_ids = None
    
    for turn_idx, prompt in enumerate(encoded_example["prompts"]):
        try:
            if turn_idx == 0:
                # 第一轮：直接使用第一个prompt
                input_text = prompt
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                
                # 检查并截断输入长度
                if input_ids.shape[1] > args.max_seq_length:
                    # 截断策略：保留前一半和后一半
                    half_len = args.max_seq_length // 2
                    input_ids = torch.cat([
                        input_ids[:, :half_len],
                        input_ids[:, -half_len:]
                    ], dim=1)
                    # print(f"当前input_ids长度: {input_ids.shape[1]}")
                    logger.warning(f"Input truncated from {input_ids.shape[1]} to {args.max_seq_length} tokens")
            else:
                # 后续轮次：添加到累积的input_ids中
                if args.disable_golden_context and turn_idx > 0:
                    # 如果禁用金标准上下文，使用前一轮的生成结果
                    prev_generated = results[turn_idx - 1]["prediction"]
                    current_prompt = f"{prev_generated}\n\n{encoded_example['prompts'][turn_idx].split(chr(10)+chr(10))[-1]}"
                else:
                    # 使用标准的prompt（已包含标准答案+下一轮的问题）
                    current_prompt = prompt
                
                current_ids = tokenizer.encode(current_prompt, add_special_tokens=False, return_tensors="pt").to(device)
                # print(f"当前current_ids长度: {current_ids.shape[1]}")
                input_ids = torch.cat([input_ids, current_ids], dim=1)
                # print(f"当前input_ids长度: {input_ids.shape[1]}")
                
            # 生成
            begin_gen = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            end_gen = time.perf_counter()
            
            # 计算生成的token数量（在更新input_ids之前）
            original_length = input_ids.shape[1]
            # print(f"original_length: {original_length}")
            generated_length = output.shape[1] - original_length
            # print(f"generated_length: {generated_length}")
            
            # 解码新生成的部分
            generated_text = tokenizer.decode(
                output[0][original_length:], 
                skip_special_tokens=True
            )
            
            # 更新input_ids包含生成的内容
            input_ids = output
            
            # 清理H2O缓存
            if args.enable_h2o_cache:
                for name, m in model.named_modules():
                    if isinstance(m, H2OLlamaAttention):
                        m._clean_cache()
            
            # 保存结果
            result = {
                "turn_idx": turn_idx,
                "prediction": generated_text.strip(),
                "ground_truth": encoded_example["ground_truth"][turn_idx],
                "generation_time": end_gen - begin_gen,
                "input_length": original_length if turn_idx == 0 else current_ids.shape[1],
                "output_length": generated_length
            }
            results.append(result)
            
            # 打印进度
            if turn_idx < 2:  # 只打印前2轮
                logger.info(f"Turn {turn_idx + 1}: Generated {len(generated_text)} chars in {end_gen - begin_gen:.2f}s")
                logger.info(f"Prediction preview: {generated_text[:200]}...")
                
        except Exception as e:
            logger.error(f"Error in turn {turn_idx}: {e}")
            results.append({
                "turn_idx": turn_idx,
                "prediction": "",
                "ground_truth": encoded_example["ground_truth"][turn_idx],
                "generation_time": 0,
                "input_length": 0,
                "output_length": 0,
                "error": str(e)
            })
    
    return results





def evaluate_scbench_official(args, model, tokenizer, data, max_length, max_gen, model_name, incremental_save_path=None):
    """官方风格的SCBench评估，支持增量保存"""
    device = model.device
    all_results = []
    
    # 限制评估样本数量
    eval_data = data[:args.num_eval_examples]
    
    # 创建增量保存目录
    if incremental_save_path:
        os.makedirs(os.path.dirname(incremental_save_path), exist_ok=True)
        logger.info(f"增量保存文件: {incremental_save_path}")
    
    for sample_idx, sample in enumerate(tqdm(eval_data, desc="Evaluating SCBench (Official Style)")):
        try:
            # 创建多轮prompt
            encoded_example = create_multiturn_prompt(
                sample, 
                max_turns=args.max_turns,
                disable_golden_context=args.disable_golden_context
            )
            
            # 多轮推理
            turn_results = multiturn_test(args, model, tokenizer, encoded_example, max_gen)
            
            # 添加样本信息并转换为输出格式
            sample_results = []
            for result in turn_results:
                result.update({
                    "sample_id": sample_idx,
                    "id": encoded_example["sample_info"]["id"],
                    "repo": encoded_example["sample_info"]["repo"], 
                    "lang": encoded_example["sample_info"]["lang"],
                })
                all_results.append(result)
                
                # 转换为输出格式（与main函数中的格式保持一致）
                output_item = {
                    "id": result["sample_id"],
                    "turn_idx": result["turn_idx"],
                    "prediction": result["prediction"],
                    "ground_truth": result["ground_truth"],
                    "lang": result["lang"],
                    "repo": result["repo"],
                    "generation_time": result["generation_time"],
                    "input_length": result["input_length"],
                    "output_length": result["output_length"]
                }
                if "error" in result:
                    output_item["error"] = result["error"]
                sample_results.append(output_item)
            
            # 增量保存：每完成一个样本就立即保存
            if incremental_save_path:
                with open(incremental_save_path, 'a', encoding='utf-8') as f:
                    for output_item in sample_results:
                        json.dump(output_item, f, ensure_ascii=False)
                        f.write('\n')
                logger.info(f"✅ 已保存样本 {sample_idx + 1}/{len(eval_data)} 的结果到增量文件")
                
            logger.info(f"Completed sample {sample_idx + 1}/{len(eval_data)}")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            # 即使出错也要记录到增量文件
            if incremental_save_path:
                error_item = {
                    "id": sample_idx,
                    "turn_idx": 0,
                    "prediction": "",
                    "ground_truth": "",
                    "lang": "",
                    "repo": "",
                    "generation_time": 0,
                    "input_length": 0,
                    "output_length": 0,
                    "error": str(e)
                }
                with open(incremental_save_path, 'a', encoding='utf-8') as f:
                    json.dump(error_item, f, ensure_ascii=False)
                    f.write('\n')
            continue
    
    return all_results


def main():
    seed_everything(42)
    args = parse_args()
    assert args.enable_vq_cache + args.enable_h2o_cache == 1
    
    # 加载配置文件
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    model_name = args.model
    max_length = model2maxlen[model_name]
    max_gen = args.max_new_tokens
    
    # 创建输出目录
    if not os.path.exists("pred"):
        os.makedirs("pred")
    
    device = torch.device(f"cuda:0")  # 使用CUDA_VISIBLE_DEVICES映射
    model, tokenizer = load_model_and_tokenizer(args, model2path, model_name, device, args.pp_size)
    
    # 加载SCBench数据集
    logger.info(f"Loading SCBench dataset: {args.task}")
    data_file = f'data/scbench/{args.task}.json'
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    logger.info(f"Loaded {len(data)} samples from {args.task}")
    
    # 设置输出路径
    exp_name = args.exp_name
    disable_golden_suffix = "_disable_golden" if args.disable_golden_context else ""
    
    # 创建输出目录结构
    output_dir = f"pred/{model_name}/{args.task}/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    config_str_list = get_config_str_list(args)
    if args.enable_h2o_cache:
        out_path = f"{output_dir}/h2o_hh_{args.important_ratio}_recent_{args.recent_ratio}{disable_golden_suffix}.jsonl"
        incremental_path = f"{output_dir}/incremental_h2o_hh_{args.important_ratio}_recent_{args.recent_ratio}{disable_golden_suffix}.jsonl"
    
    elif args.enable_vq_cache:
        out_path = f"{output_dir}/{'_'.join(config_str_list)}{disable_golden_suffix}.jsonl"
        incremental_path = f"{output_dir}/incremental_{'_'.join(config_str_list)}{disable_golden_suffix}.jsonl"

    # 评估模型
    logger.info("Starting official SCBench evaluation...")
    logger.info(f"Max turns: {args.max_turns}, Disable golden context: {args.disable_golden_context}")
    logger.info(f"增量保存路径: {incremental_path}")
    logger.info(f"最终保存路径: {out_path}")
    
    results = evaluate_scbench_official(args, model, tokenizer, data, max_length, max_gen, model_name, incremental_save_path=incremental_path)
    
    # 保存结果（保持与原始格式兼容）
    with open(out_path, 'w', encoding='utf-8') as f:
        for result in results:
            # 转换为原始期望的格式
            output_item = {
                "id": result["sample_id"],
                "turn_idx": result["turn_idx"],
                "prediction": result["prediction"],
                "ground_truth": result["ground_truth"],
                "lang": result["lang"],
                "repo": result["repo"],
                "generation_time": result["generation_time"],
                "input_length": result["input_length"],
                "output_length": result["output_length"]
            }
            if "error" in result:
                output_item["error"] = result["error"]
                
            json.dump(output_item, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Evaluation completed. Results saved to {out_path}")
    logger.info(f"Processed {len(set(r['sample_id'] for r in results))} samples with {len(results)} total turns")
    
    # 计算统计信息
    if results:
        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            avg_gen_time = sum(r["generation_time"] for r in successful_results) / len(successful_results)
            avg_input_len = sum(r["input_length"] for r in successful_results) / len(successful_results)
            avg_output_len = sum(r["output_length"] for r in successful_results) / len(successful_results)
            
            logger.info(f"Average generation time: {avg_gen_time:.2f}s")
            logger.info(f"Average input length: {avg_input_len:.1f} tokens")
            logger.info(f"Average output length: {avg_output_len:.1f} tokens")

if __name__ == '__main__':
    main()
