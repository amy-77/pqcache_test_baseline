#!/usr/bin/env python3
"""
PQCache调试示例 - 追踪函数调用流程
使用这个脚本来理解代码执行路径
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.append('/home/pai/data/PQCache')

def debug_attention_flow():
    """调试attention计算流程"""
    print("🎯 开始调试Attention流程...")
    
    # 模拟创建一些简单的tensor用于调试
    batch_size = 1
    seq_len = 10
    num_heads = 8
    head_dim = 64
    
    # 创建模拟的query, key, value
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"📊 Query shape: {query.shape}")
    print(f"📊 Key shape: {key.shape}")
    print(f"📊 Value shape: {value.shape}")
    
    # 计算attention weights
    attention_weights = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(head_dim)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    print(f"📊 Attention weights shape: {attention_weights.shape}")
    
    # 计算Neff (有效注意力数)
    neff_values = []
    for head_idx in range(num_heads):
        head_weights = attention_weights[0, head_idx, :, :]  # [seq_len, seq_len]
        # 对每个query位置计算Neff
        pos_neff = 1.0 / torch.clamp((head_weights ** 2).sum(dim=-1), min=1e-12)
        head_neff = pos_neff.mean().item()  # 简单平均
        neff_values.append(head_neff)
        print(f"🔍 Head {head_idx}: Neff = {head_neff:.3f}")
    
    return neff_values

def debug_import_pqcache():
    """调试PQCache模块导入"""
    print("🎯 开始调试PQCache导入...")
    
    try:
        # 尝试导入PQCache相关模块
        from vq_method.llama31_patch import VQLlama31ForCausalLM
        print("✅ 成功导入 VQLlama31ForCausalLM")
        
        from vq_method.retrieval_based.pq_search import PqBasedSearchCompressor
        print("✅ 成功导入 PqBasedSearchCompressor")
        
        # 查看类的方法
        methods = [method for method in dir(PqBasedSearchCompressor) if not method.startswith('_')]
        print(f"📋 PqBasedSearchCompressor 方法: {methods}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def debug_config_loading():
    """调试配置加载过程"""
    print("🎯 开始调试配置加载...")
    
    # 模拟配置对象
    class DebugConfig:
        def __init__(self):
            self.compressor = "pq_search"  # 可以改成 "original" 或 "sparq_f"
            self.compress_ratio = 0.5
            self.recent_ratio = 0.1
            self.n_subvec_per_head = 4
            self.n_subbits = 8
            self.gqa = True
            self.sink_size = 32
            self.num_hidden_layers = 32
            self.num_key_value_heads = 8
            self.hidden_size = 4096
            self.num_attention_heads = 32
            self.max_iter = 100
    
    config = DebugConfig()
    print(f"📋 使用压缩器: {config.compressor}")
    print(f"📋 压缩比率: {config.compress_ratio}")
    
    return config

def main():
    """主调试函数"""
    print("=" * 60)
    print("🚀 PQCache 调试会话开始")
    print("=" * 60)
    
    # 步骤1: 调试基础attention计算
    print("\n🔍 步骤1: 调试基础Attention计算")
    neff_values = debug_attention_flow()
    
    # 步骤2: 调试模块导入
    print("\n🔍 步骤2: 调试模块导入")
    import_success = debug_import_pqcache()
    
    # 步骤3: 调试配置
    print("\n🔍 步骤3: 调试配置加载")
    config = debug_config_loading()
    
    print("\n" + "=" * 60)
    print("✅ 调试会话完成")
    print("=" * 60)
    
    # 在这里设置断点可以检查所有变量
    breakpoint_here = True  # 在此行设置断点
    
if __name__ == "__main__":
    main() 