#!/usr/bin/env python3
"""
下载SCBench数据集：scbench_kv 和 scbench_vt
"""

import os
import json
from datasets import load_dataset

def download_and_save_dataset(dataset_name, output_dir):
    """下载数据集并保存为JSONL格式"""
    print(f"正在下载 {dataset_name}...")
    
    try:
        # 从Hugging Face下载数据集
        dataset = load_dataset("microsoft/SCBench", dataset_name, split="test")
        print(f"  - 数据集大小: {len(dataset)} 个样本")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSONL格式
        output_file = os.path.join(output_dir, f"{dataset_name}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"  - 已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"  - 下载失败: {e}")
        return False

def main():
    # 输出目录
    output_dir = "./data/scbench"
    
    # 要下载的数据集
    datasets_to_download = ["scbench_kv", "scbench_vt"]
    
    print("开始下载SCBench数据集...")
    print(f"输出目录: {output_dir}")
    print("=" * 50)
    
    success_count = 0
    for dataset_name in datasets_to_download:
        if download_and_save_dataset(dataset_name, output_dir):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"下载完成！成功下载 {success_count}/{len(datasets_to_download)} 个数据集")
    
    if success_count == len(datasets_to_download):
        print("所有数据集都已成功下载并保存！")
    else:
        print("部分数据集下载失败，请检查错误信息")

if __name__ == "__main__":
    main()
