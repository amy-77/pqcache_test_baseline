#!/usr/bin/env python3
"""
从日志中提取当前已完成的SCBench结果，模拟增量保存的效果
"""
import re
import json
from datetime import datetime

def extract_progress_from_log(log_file="/home/pai/data/PQCache/scbench_run.log"):
    """从日志文件中提取已完成的样本信息"""
    
    print("📊 从日志中提取SCBench RepoQA进展...")
    
    # 用于存储结果的变量
    completed_samples = []
    current_sample = None
    current_turns = []
    
    # 正则表达式模式
    turn_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| INFO.*Turn (\d+): Generated (\d+) chars in ([\d.]+)s"
    sample_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| INFO.*Completed sample (\d+)/(\d+)"
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # 检查Turn完成
                turn_match = re.search(turn_pattern, line)
                if turn_match:
                    timestamp, turn_idx, chars, time_taken = turn_match.groups()
                    turn_info = {
                        "turn_idx": int(turn_idx) - 1,  # 转换为0-based
                        "timestamp": timestamp,
                        "generated_chars": int(chars),
                        "generation_time": float(time_taken),
                        "prediction_preview": f"Generated {chars} characters"  # 实际内容在日志中难以提取
                    }
                    current_turns.append(turn_info)
                
                # 检查Sample完成
                sample_match = re.search(sample_pattern, line)
                if sample_match:
                    timestamp, sample_idx, total_samples = sample_match.groups()
                    sample_idx = int(sample_idx) - 1  # 转换为0-based
                    
                    # 保存前一个样本的信息
                    if current_turns:
                        sample_info = {
                            "sample_id": sample_idx,
                            "completion_timestamp": timestamp,
                            "turns": current_turns.copy(),
                            "total_turns": len(current_turns)
                        }
                        completed_samples.append(sample_info)
                        current_turns = []  # 重置
                
                # 显示进度（每100万行）
                if line_num % 1000000 == 0:
                    print(f"   处理了 {line_num:,} 行日志...")
    
    except FileNotFoundError:
        print(f"❌ 日志文件不存在: {log_file}")
        return []
    except Exception as e:
        print(f"❌ 读取日志文件出错: {e}")
        return []
    
    return completed_samples

def save_extracted_results(completed_samples, output_file="extracted_scbench_progress.jsonl"):
    """将提取的结果保存为JSONL格式"""
    
    if not completed_samples:
        print("❌ 没有找到已完成的样本")
        return
    
    print(f"💾 保存 {len(completed_samples)} 个已完成样本到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in completed_samples:
            for turn in sample["turns"]:
                # 模拟SCBench输出格式
                result_item = {
                    "id": sample["sample_id"],
                    "turn_idx": turn["turn_idx"],
                    "prediction": turn["prediction_preview"],
                    "ground_truth": "N/A (not available in log)",
                    "lang": "unknown",
                    "repo": "unknown", 
                    "generation_time": turn["generation_time"],
                    "input_length": "N/A",
                    "output_length": turn["generated_chars"],
                    "completion_timestamp": turn["timestamp"],
                    "extracted_from_log": True
                }
                json.dump(result_item, f, ensure_ascii=False)
                f.write('\n')
    
    print("✅ 结果已保存")

def analyze_progress(completed_samples):
    """分析进展情况"""
    
    if not completed_samples:
        print("❌ 没有可分析的数据")
        return
    
    print(f"\n📈 进展分析:")
    print(f"已完成样本数: {len(completed_samples)}/88 ({len(completed_samples)/88*100:.1f}%)")
    
    total_turns = sum(len(sample["turns"]) for sample in completed_samples)
    print(f"已完成轮次: {total_turns}")
    
    # 计算平均生成时间
    all_times = []
    all_chars = []
    for sample in completed_samples:
        for turn in sample["turns"]:
            all_times.append(turn["generation_time"])
            all_chars.append(turn["generated_chars"])
    
    if all_times:
        avg_time = sum(all_times) / len(all_times)
        avg_chars = sum(all_chars) / len(all_chars)
        print(f"平均生成时间: {avg_time:.1f} 秒")
        print(f"平均生成字符数: {avg_chars:.0f} 字符")
        
        # 时间分析
        min_time = min(all_times)
        max_time = max(all_times)
        print(f"生成时间范围: {min_time:.1f}s - {max_time:.1f}s")
    
    # 最新完成的样本
    if completed_samples:
        latest = completed_samples[-1]
        print(f"最新完成样本: #{latest['sample_id']} (时间: {latest['completion_timestamp']})")

def main():
    print("🔍 开始提取SCBench RepoQA当前进展...")
    
    # 提取进展
    completed_samples = extract_progress_from_log()
    
    if completed_samples:
        # 分析进展
        analyze_progress(completed_samples)
        
        # 保存结果
        save_extracted_results(completed_samples)
        
        print(f"\n🎉 成功提取了 {len(completed_samples)} 个已完成样本的信息！")
        print("📁 结果已保存到: extracted_scbench_progress.jsonl")
        
    else:
        print("❌ 未能从日志中提取到有效的进展信息")

if __name__ == "__main__":
    main()
