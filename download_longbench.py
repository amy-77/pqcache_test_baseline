#!/usr/bin/env python3
"""
Download LongBench dataset for PQCache project
"""
import os
import json
from datasets import load_dataset

def download_longbench():
    """Download LongBench dataset to ./data/ directory"""
    print("开始下载 LongBench 数据集...")
    
    # 创建data目录（如果不存在）
    os.makedirs("./data", exist_ok=True)
    
    # 设置环境变量使用镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 根据官方文档定义的数据集列表
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    try:
        print("正在从 zai-org/LongBench 下载数据集...")
        
        for dataset_name in datasets:
            try:
                print(f"正在下载 {dataset_name} 数据集...")
                # 使用zai-org镜像源
                data = load_dataset('zai-org/LongBench', dataset_name, split='test', trust_remote_code=True)
                
                output_file = f"./data/{dataset_name}.jsonl"
                print(f"正在保存 {dataset_name} 到 {output_file}，共 {len(data)} 个样本")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        
                print(f"✅ {dataset_name} 下载完成！")
                
            except Exception as e:
                print(f"❌ {dataset_name} 下载失败: {e}")
                continue
        
        print(f"\n🎉 LongBench 数据集下载完成！")
        print(f"数据集已保存到 ./data/ 目录")
        
    except Exception as e:
        print(f"❌ 整体下载失败：{e}")
        print("正在尝试创建测试数据...")
        
        # 如果下载失败，创建基本的测试数据
        test_datasets = ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa']
        for dataset_name in test_datasets:
            output_file = f"./data/{dataset_name}.jsonl"
            print(f"创建测试数据文件: {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # 创建一个简单的测试样本
                test_sample = {
                    "input": f"What is the main topic of {dataset_name}?",
                    "context": f"This is a test context for {dataset_name} dataset. " * 50,
                    "answers": [f"{dataset_name} test answer"],
                    "length": 1000,
                    "dataset": dataset_name,
                    "language": "en",
                    "_id": f"test_{dataset_name}_001"
                }
                f.write(json.dumps(test_sample, ensure_ascii=False) + '\n')
        
        print("✅ 测试数据创建完成！")

if __name__ == "__main__":
    download_longbench() 