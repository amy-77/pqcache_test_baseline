#!/usr/bin/env python3
"""
Neff计算简单例子
===============

用简单的例子解释Neff计算过程
"""

import torch
import numpy as np

def simple_neff_example():
    """简单的Neff计算例子"""
    print("=== Neff计算简单例子 ===\n")
    
    # 假设我们有一个简单的注意力权重矩阵
    # 形状: [seq_len=3, kv_seq_len=5]
    # 每行是一个query position对所有key position的注意力分布
    
    print("1. 原始注意力权重矩阵 (head_weights):")
    print("   每行是一个query position对所有key position的注意力分布")
    print("   每行的和应该等于1.0")
    
    # 创建一个简单的注意力权重矩阵
    head_weights = torch.tensor([
        [0.5, 0.3, 0.1, 0.05, 0.05],  # 第1个query position的注意力分布
        [0.2, 0.4, 0.2, 0.15, 0.05],  # 第2个query position的注意力分布  
        [0.1, 0.2, 0.4, 0.2, 0.1]     # 第3个query position的注意力分布
    ])
    
    print(f"head_weights形状: {head_weights.shape}")
    print("head_weights内容:")
    print(head_weights)
    print(f"每行权重和: {head_weights.sum(dim=1)}")  # 应该都是1.0
    print()
    
    print("2. 聚合权重 (aggregated_weights):")
    print("   对所有query position取平均，得到该head对各token的整体关注度")
    
    # 对所有query position取平均
    aggregated_weights = torch.mean(head_weights, dim=0)  # [kv_seq_len]
    print(f"aggregated_weights形状: {aggregated_weights.shape}")
    print(f"aggregated_weights内容: {aggregated_weights}")
    print(f"aggregated_weights和: {aggregated_weights.sum():.6f}")
    print()
    
    print("3. 归一化:")
    print("   确保聚合后的权重是概率分布（和为1.0）")
    
    # 重新归一化
    aggregated_weights = aggregated_weights / aggregated_weights.sum()
    print(f"归一化后aggregated_weights: {aggregated_weights}")
    print(f"归一化后和: {aggregated_weights.sum():.6f}")
    print()
    
    print("4. 计算Neff:")
    print("   Neff = 1 / sum(w_i^2)")
    
    # 计算权重平方和
    weights_squared_sum = torch.sum(aggregated_weights ** 2)
    print(f"权重平方和: {weights_squared_sum:.6f}")
    
    # 计算Neff
    neff = 1.0 / weights_squared_sum
    print(f"Neff值: {neff:.2f}")
    print()
    
    print("5. Neff的含义:")
    print(f"   - 如果注意力很集中（稀疏），Neff值较小: {neff:.2f}")
    print(f"   - 如果注意力很分散（密集），Neff值较大")
    print(f"   - 理论上，Neff的最小值是1.0（完全集中在一个token）")
    print(f"   - 理论上，Neff的最大值是token数量（完全均匀分布）")
    print()
    
    # 对比不同分布的例子
    print("6. 不同分布的Neff对比:")
    
    # 稀疏分布（注意力集中）
    sparse_weights = torch.tensor([0.8, 0.1, 0.05, 0.03, 0.02])
    sparse_neff = 1.0 / torch.sum(sparse_weights ** 2)
    print(f"稀疏分布 [0.8, 0.1, 0.05, 0.03, 0.02]: Neff = {sparse_neff:.2f}")
    
    # 均匀分布（注意力分散）
    uniform_weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    uniform_neff = 1.0 / torch.sum(uniform_weights ** 2)
    print(f"均匀分布 [0.2, 0.2, 0.2, 0.2, 0.2]: Neff = {uniform_neff:.2f}")
    
    print()
    print("结论: Neff值越小，注意力越稀疏（集中）；Neff值越大，注意力越密集（分散）")

def why_take_mean():
    """解释为什么要取平均"""
    print("\n=== 为什么要取平均？ ===\n")
    
    print("问题：我们有一个注意力权重矩阵 [seq_len, kv_seq_len]")
    print("      每行是一个query position对所有key position的注意力分布")
    print("      我们想要得到这个head对所有token的整体关注度")
    print()
    
    print("方法1：取平均")
    print("  - 含义：这个head在所有query position上对各token的平均关注度")
    print("  - 优点：简单直观，考虑了所有query position的贡献")
    print("  - 缺点：可能掩盖了不同position的差异")
    print()
    
    print("方法2：取最大值")
    print("  - 含义：这个head对每个token的最大关注度")
    print("  - 优点：突出最强的注意力连接")
    print("  - 缺点：可能过于激进")
    print()
    
    print("方法3：加权平均")
    print("  - 含义：根据query position的重要性加权平均")
    print("  - 优点：更精细的控制")
    print("  - 缺点：需要额外的权重定义")
    print()
    
    print("当前实现选择方法1（取平均），因为：")
    print("1. 简单且直观")
    print("2. 能够反映head的整体行为模式")
    print("3. 适合用于压缩策略的决策")

if __name__ == "__main__":
    simple_neff_example()
    why_take_mean() 