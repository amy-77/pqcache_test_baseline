#!/usr/bin/env python3
"""
AIME 2024数学推理结果评估脚本
参考R-KV算法的评估方法，对PQCache生成的AIME 2024推理结果进行准确性评估
"""

import json
import re
from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple

def extract_answer_from_response(response: str, dataset_name: str = "aime_2024") -> str:
    """
    从模型响应中提取最终答案
    参考R-KV的答案提取逻辑，优化数学答案提取
    """
    if not response:
        return ""
    
    # 优先级模式 - boxed格式最可靠（AIME标准格式）
    patterns = [
        r"\\boxed\{([^}]+)\}",  # LaTeX boxed格式（最高优先级）
        r"Final answer:\s*([+-]?\d+(?:\.\d+)?)",  # Final answer格式
        r"(?:final answer|answer|solution).*?(?:is|:|=)\s*([+-]?\d+(?:\.\d+)?)",
        r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*([+-]?\d+(?:\.\d+)?)",
        r"\$([+-]?\d+(?:\.\d+)?)\$",  # LaTeX数学格式
        r"([+-]?\d+(?:\.\d+)?)(?:\s*$|\s*\.?\s*$)"  # 末尾数字
    ]
    
    for pattern in patterns:
        if "boxed" in pattern:
            # 对boxed格式不转换为小写，保持原始内容
            matches = re.findall(pattern, response)
        else:
            matches = re.findall(pattern, response.lower(), re.IGNORECASE)
        
        if matches:
            answer = matches[-1].strip()  # 取最后一个匹配
            
            # 对boxed答案进行特殊处理
            if "boxed" in pattern:
                # 尝试从复杂的boxed内容中提取数字
                if answer.isdigit():
                    return answer
                else:
                    # 尝试提取数字
                    number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
                    if number_match:
                        return number_match.group(1)
                    else:
                        return answer  # 如果无法提取数字，返回原始内容
            else:
                return answer
    
    return ""

def normalize_answer(answer: str) -> str:
    """标准化答案格式，用于比较"""
    if not answer:
        return ""
    
    # 移除多余的空格和标点
    answer = answer.strip().rstrip('.')
    
    # 对于AIME，答案通常是整数
    try:
        # 尝试转换为整数（AIME答案范围0-999）
        num = float(answer)
        if num == int(num) and 0 <= int(num) <= 999:
            return str(int(num))
    except ValueError:
        pass
    
    return answer

def evaluate_accuracy(predicted: str, ground_truth: str) -> bool:
    """
    评估预测答案的准确性
    使用更严格的匹配标准，参考R-KV的评估方法
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not gt_norm:
        return False
    
    # 精确匹配
    return pred_norm == gt_norm

def calculate_detailed_metrics(results: List[Dict]) -> Dict[str, Any]:
    """计算详细的评估指标"""
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}
    
    correct = sum(1 for r in results if r.get('is_correct', False))
    
    # 按问题ID分析（如果有的话）
    by_problem_id = {}
    generation_times = []
    input_lengths = []
    output_lengths = []
    
    for result in results:
        problem_id = result.get('id', 'unknown')
        by_problem_id[problem_id] = result.get('is_correct', False)
        
        if 'generation_time' in result:
            generation_times.append(result['generation_time'])
        if 'input_length' in result:
            input_lengths.append(result['input_length'])
        if 'output_length' in result:
            output_lengths.append(result['output_length'])
    
    metrics = {
        "total_samples": total,
        "correct_samples": correct,
        "accuracy": correct / total,
        "error_rate": 1 - (correct / total),
        "by_problem_accuracy": by_problem_id
    }
    
    # 性能指标
    if generation_times:
        metrics["avg_generation_time"] = sum(generation_times) / len(generation_times)
        metrics["total_generation_time"] = sum(generation_times)
    
    if input_lengths:
        metrics["avg_input_length"] = sum(input_lengths) / len(input_lengths)
    
    if output_lengths:
        metrics["avg_output_length"] = sum(output_lengths) / len(output_lengths)
    
    return metrics

def re_evaluate_results(results_file: str, output_file: str = None) -> Tuple[Dict, List[Dict]]:
    """
    重新评估已有的推理结果，使用改进的答案提取和匹配逻辑
    """
    print(f"📊 重新评估结果文件: {results_file}")
    
    # 读取结果文件
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    
    print(f"📈 加载了 {len(results)} 个样本")
    
    # 重新评估每个样本
    re_evaluated_results = []
    correct_count = 0
    
    for i, result in enumerate(results):
        # 获取原始数据
        problem = result.get('problem', '')
        ground_truth = result.get('ground_truth', '')
        full_response = result.get('full_response', '')
        
        # 重新提取预测答案
        predicted_answer = extract_answer_from_response(full_response, "aime_2024")
        
        # 重新评估准确性
        is_correct = evaluate_accuracy(predicted_answer, ground_truth)
        
        if is_correct:
            correct_count += 1
        
        # 更新结果
        updated_result = result.copy()
        updated_result['predicted_answer_re_extracted'] = predicted_answer
        updated_result['is_correct_re_evaluated'] = is_correct
        updated_result['original_predicted'] = result.get('predicted_answer', '')
        updated_result['original_is_correct'] = result.get('is_correct', False)
        
        re_evaluated_results.append(updated_result)
        
        # 显示前几个样本的详细信息
        if i < 5:
            print(f"\n样本 {i+1}:")
            print(f"  问题: {problem[:100]}...")
            print(f"  标准答案: {ground_truth}")
            print(f"  原预测答案: {result.get('predicted_answer', '')}")
            print(f"  重新提取答案: {predicted_answer}")
            print(f"  原正确性: {result.get('is_correct', False)}")
            print(f"  重新评估正确性: {is_correct}")
    
    # 计算详细指标
    metrics = calculate_detailed_metrics(re_evaluated_results)
    
    # 比较原始评估和重新评估的结果
    original_correct = sum(1 for r in results if r.get('is_correct', False))
    
    print(f"\n{'='*60}")
    print(f"🔍 AIME 2024 重新评估结果对比")
    print(f"{'='*60}")
    print(f"总样本数: {len(results)}")
    print(f"原始评估 - 正确: {original_correct}, 准确率: {original_correct/len(results):.4f}")
    print(f"重新评估 - 正确: {correct_count}, 准确率: {correct_count/len(results):.4f}")
    print(f"准确率变化: {(correct_count - original_correct)/len(results):+.4f}")
    
    if metrics.get('avg_generation_time'):
        print(f"平均生成时间: {metrics['avg_generation_time']:.2f}秒")
    if metrics.get('avg_input_length'):
        print(f"平均输入长度: {metrics['avg_input_length']:.0f} tokens")
    if metrics.get('avg_output_length'):
        print(f"平均输出长度: {metrics['avg_output_length']:.0f} tokens")
    
    # 保存重新评估的结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in re_evaluated_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"✅ 重新评估结果已保存到: {output_file}")
    
    return metrics, re_evaluated_results

def analyze_error_patterns(results: List[Dict]) -> Dict[str, Any]:
    """分析错误模式，帮助改进模型"""
    
    errors = [r for r in results if not r.get('is_correct_re_evaluated', False)]
    
    if not errors:
        return {"message": "没有错误样本进行分析"}
    
    error_analysis = {
        "total_errors": len(errors),
        "error_types": {},
        "common_patterns": [],
        "answer_extraction_failures": 0,
        "wrong_calculations": 0
    }
    
    for error in errors:
        predicted = error.get('predicted_answer_re_extracted', '')
        ground_truth = error.get('ground_truth', '')
        response = error.get('full_response', '')
        
        # 分析错误类型
        if not predicted:
            error_analysis["answer_extraction_failures"] += 1
        elif predicted != ground_truth:
            error_analysis["wrong_calculations"] += 1
        
        # 分析响应长度（可能指示推理深度）
        response_length = len(response)
        if response_length < 100:
            error_analysis["error_types"]["too_short_response"] = error_analysis["error_types"].get("too_short_response", 0) + 1
        elif response_length > 5000:
            error_analysis["error_types"]["too_long_response"] = error_analysis["error_types"].get("too_long_response", 0) + 1
    
    return error_analysis

def main():
    parser = argparse.ArgumentParser(description="AIME 2024结果评估脚本")
    parser.add_argument('--results_file', type=str, required=True,
                        help="推理结果文件路径")
    parser.add_argument('--output_file', type=str, default=None,
                        help="重新评估结果保存路径")
    parser.add_argument('--detailed_analysis', action='store_true',
                        help="进行详细的错误分析")
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not Path(args.results_file).exists():
        print(f"❌ 结果文件不存在: {args.results_file}")
        return
    
    # 如果没有指定输出文件，生成默认名称
    if not args.output_file:
        results_path = Path(args.results_file)
        args.output_file = str(results_path.parent / f"re_evaluated_{results_path.name}")
    
    # 执行重新评估
    metrics, re_evaluated_results = re_evaluate_results(args.results_file, args.output_file)
    
    # 详细分析（如果要求）
    if args.detailed_analysis:
        print(f"\n🔍 进行详细错误分析...")
        error_analysis = analyze_error_patterns(re_evaluated_results)
        
        print(f"\n错误分析结果:")
        print(f"  总错误数: {error_analysis['total_errors']}")
        print(f"  答案提取失败: {error_analysis['answer_extraction_failures']}")
        print(f"  计算错误: {error_analysis['wrong_calculations']}")
        
        if error_analysis.get('error_types'):
            print(f"  错误类型分布: {error_analysis['error_types']}")
    
    print(f"\n✅ 评估完成！")

if __name__ == "__main__":
    main()
