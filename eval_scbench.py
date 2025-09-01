#!/usr/bin/env python3
"""
SCBench评估指标计算脚本
计算PQCache在SCBench数据集上的性能指标
"""
import os
import json
import argparse
import numpy as np
from collections import defaultdict
import re
from typing import Dict, List, Any

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_scbench', help='结果目录')
    parser.add_argument('--dataset', type=str, default=None, help='指定数据集，None表示全部')
    parser.add_argument('--output_file', type=str, default='scbench_evaluation.json', help='输出文件')
    return parser.parse_args()

def extract_answer(text: str) -> str:
    """从生成文本中提取答案"""
    # 移除多余的空白字符
    text = text.strip()
    
    # 如果文本很短，直接返回
    if len(text) < 10:
        return text
    
    # 尝试提取关键信息（针对不同任务类型）
    # 1. 代码相关任务 - 提取函数名或代码片段
    code_patterns = [
        r'def\s+(\w+)',  # Python函数
        r'function\s+(\w+)',  # JavaScript函数
        r'(\w+)\s*\(',  # 函数调用
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    
    # 2. 键值查找任务 - 提取值
    kv_patterns = [
        r'"([^"]+)"',  # 引号内的值
        r':\s*([^\s,\]]+)',  # 冒号后的值
    ]
    
    for pattern in kv_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    
    # 3. 通用答案提取 - 取第一句话
    sentences = text.split('.')
    if sentences:
        return sentences[0].strip()
    
    return text[:100]  # 截断到100字符

def calculate_exact_match(pred: str, ref: str) -> float:
    """计算精确匹配分数"""
    pred_clean = pred.strip().lower()
    ref_clean = ref.strip().lower()
    return 1.0 if pred_clean == ref_clean else 0.0

def calculate_contains_match(pred: str, ref: str) -> float:
    """计算包含匹配分数"""
    pred_clean = pred.strip().lower()
    ref_clean = ref.strip().lower()
    
    if ref_clean in pred_clean or pred_clean in ref_clean:
        return 1.0
    return 0.0

def calculate_token_overlap(pred: str, ref: str) -> float:
    """计算token重叠分数"""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    
    if not ref_tokens:
        return 0.0
    
    overlap = len(pred_tokens & ref_tokens)
    return overlap / len(ref_tokens)

def evaluate_dataset_results(results_file: str, dataset_name: str) -> Dict[str, Any]:
    """评估单个数据集的结果"""
    if not os.path.exists(results_file):
        return {'error': f'Results file not found: {results_file}'}
    
    # 加载结果
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    
    if not results:
        return {'error': 'No results found'}
    
    # 计算各种指标
    metrics = {
        'exact_match': [],
        'contains_match': [],
        'token_overlap': [],
        'response_length': [],
        'input_length': [],
        'error_rate': 0,
        'total_samples': len(results)
    }
    
    error_count = 0
    
    for result in results:
        pred = result.get('response', '')
        ref = result.get('reference', '')
        
        # 检查是否有错误
        if pred.startswith('ERROR'):
            error_count += 1
            continue
        
        # 提取答案
        pred_answer = extract_answer(pred)
        ref_answer = extract_answer(ref)
        
        # 计算指标
        metrics['exact_match'].append(calculate_exact_match(pred_answer, ref_answer))
        metrics['contains_match'].append(calculate_contains_match(pred_answer, ref_answer))
        metrics['token_overlap'].append(calculate_token_overlap(pred_answer, ref_answer))
        
        # 长度统计
        metrics['response_length'].append(len(pred))
        metrics['input_length'].append(result.get('input_length', 0))
    
    # 计算平均值
    metrics['error_rate'] = error_count / len(results)
    
    for key in ['exact_match', 'contains_match', 'token_overlap', 'response_length', 'input_length']:
        if metrics[key]:
            metrics[f'{key}_mean'] = np.mean(metrics[key])
            metrics[f'{key}_std'] = np.std(metrics[key])
        else:
            metrics[f'{key}_mean'] = 0.0
            metrics[f'{key}_std'] = 0.0
    
    # 按轮次分析（如果是多轮对话）
    turn_metrics = defaultdict(list)
    for result in results:
        turn = result.get('turn', 0)
        pred = result.get('response', '')
        ref = result.get('reference', '')
        
        if not pred.startswith('ERROR'):
            pred_answer = extract_answer(pred)
            ref_answer = extract_answer(ref)
            turn_metrics[turn].append({
                'exact_match': calculate_exact_match(pred_answer, ref_answer),
                'contains_match': calculate_contains_match(pred_answer, ref_answer),
                'token_overlap': calculate_token_overlap(pred_answer, ref_answer)
            })
    
    # 计算每轮次的平均指标
    turn_analysis = {}
    for turn, turn_results in turn_metrics.items():
        turn_analysis[f'turn_{turn}'] = {
            'exact_match': np.mean([r['exact_match'] for r in turn_results]),
            'contains_match': np.mean([r['contains_match'] for r in turn_results]),
            'token_overlap': np.mean([r['token_overlap'] for r in turn_results]),
            'sample_count': len(turn_results)
        }
    
    return {
        'dataset': dataset_name,
        'overall_metrics': metrics,
        'turn_analysis': turn_analysis
    }

def main():
    args = parse_args()
    
    print(f"🔍 开始评估SCBench结果")
    print(f"📂 结果目录: {args.results_dir}")
    
    if not os.path.exists(args.results_dir):
        print(f"❌ 结果目录不存在: {args.results_dir}")
        return
    
    all_evaluations = {}
    
    # 遍历结果目录
    for item in os.listdir(args.results_dir):
        item_path = os.path.join(args.results_dir, item)
        
        if os.path.isdir(item_path):
            # 提取数据集名称
            dataset_name = item.split('_')[0] + '_' + item.split('_')[1] + '_' + item.split('_')[2]
            
            # 如果指定了特定数据集，跳过其他数据集
            if args.dataset and args.dataset not in dataset_name:
                continue
            
            # 查找结果文件
            results_file = os.path.join(item_path, 'results.jsonl')
            
            if os.path.exists(results_file):
                print(f"📊 评估数据集: {dataset_name}")
                evaluation = evaluate_dataset_results(results_file, dataset_name)
                all_evaluations[dataset_name] = evaluation
                
                # 打印关键指标
                if 'overall_metrics' in evaluation:
                    metrics = evaluation['overall_metrics']
                    print(f"  ✅ 精确匹配: {metrics.get('exact_match_mean', 0):.3f}")
                    print(f"  ✅ 包含匹配: {metrics.get('contains_match_mean', 0):.3f}")
                    print(f"  ✅ Token重叠: {metrics.get('token_overlap_mean', 0):.3f}")
                    print(f"  ❌ 错误率: {metrics.get('error_rate', 0):.3f}")
                    print(f"  📏 平均输入长度: {metrics.get('input_length_mean', 0):.0f}")
                    print(f"  📏 平均输出长度: {metrics.get('response_length_mean', 0):.0f}")
                else:
                    print(f"  ❌ 评估失败: {evaluation.get('error', 'Unknown error')}")
    
    # 计算总体统计
    if all_evaluations:
        overall_stats = {
            'total_datasets': len(all_evaluations),
            'avg_exact_match': np.mean([
                eval_data['overall_metrics'].get('exact_match_mean', 0) 
                for eval_data in all_evaluations.values() 
                if 'overall_metrics' in eval_data
            ]),
            'avg_contains_match': np.mean([
                eval_data['overall_metrics'].get('contains_match_mean', 0)
                for eval_data in all_evaluations.values()
                if 'overall_metrics' in eval_data
            ]),
            'avg_token_overlap': np.mean([
                eval_data['overall_metrics'].get('token_overlap_mean', 0)
                for eval_data in all_evaluations.values()
                if 'overall_metrics' in eval_data
            ])
        }
        
        all_evaluations['overall_summary'] = overall_stats
    
    # 保存评估结果
    output_path = os.path.join(args.results_dir, args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_evaluations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 评估完成!")
    print(f"📊 详细结果保存到: {output_path}")
    
    if 'overall_summary' in all_evaluations:
        summary = all_evaluations['overall_summary']
        print(f"\n📈 总体统计:")
        print(f"  🎯 平均精确匹配: {summary['avg_exact_match']:.3f}")
        print(f"  🎯 平均包含匹配: {summary['avg_contains_match']:.3f}")  
        print(f"  🎯 平均Token重叠: {summary['avg_token_overlap']:.3f}")
        print(f"  📊 评估数据集数量: {summary['total_datasets']}")

if __name__ == "__main__":
    main()
