#!/usr/bin/env python3
"""
下载SCBench数据集，特别是scbench_repoqa_and_kv和scbench_repoqa数据集
"""
import os
import json
from datasets import load_dataset

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_scbench():
    """下载SCBench数据集"""
    # 目标数据集列表，重点关注decode-heavy任务
    target_datasets = [
        "scbench_repoqa_and_kv",  # 代码功能检索+键值查找的多任务场景  
        "scbench_repoqa",         # 基于自然语言描述的代码功能检索
        "scbench_kv",             # 键值查找测试
        "scbench_qa_eng",         # 英文问答
        "scbench_many_shot",      # 多样本上下文学习
    ]
    
    # 确保数据目录存在
    os.makedirs("data/scbench", exist_ok=True)
    
    for dataset_name in target_datasets:
        print(f"正在下载 {dataset_name}...")
        try:
            # 从HuggingFace加载数据集
            data = load_dataset("microsoft/SCBench", dataset_name, split="test")
            
            # 转换为JSONL格式保存
            output_file = f"data/scbench/{dataset_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✅ {dataset_name} 下载完成，保存到 {output_file}")
            print(f"   数据量: {len(data)} 条")
            
        except Exception as e:
            print(f"❌ 下载 {dataset_name} 失败: {e}")
    
    print("\n📊 SCBench数据集下载完成！")

if __name__ == "__main__":
    download_scbench()
