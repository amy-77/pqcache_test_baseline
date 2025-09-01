#!/usr/bin/env python3
"""
KV缓存分配策略示例
演示COMPRESS, TOPK, RECENT_RATIO, SINK_SIZE, RECENT_SIZE的作用
"""

def demonstrate_cache_allocation():
    # 原始参数
    original_length = 10000  # 原始序列长度
    compress_ratio = 0.1     # COMPRESS=0.1
    topk_ratio = 0.5        # TOPK=0.5  
    recent_ratio = 0.5      # RECENT_RATIO=0.5
    sink_size = 32          # SINK_SIZE=32
    recent_size = 32        # RECENT_SIZE=32
    
    print("🎯 KV缓存分配策略演示")
    print("=" * 50)
    
    # 步骤1: 总体压缩
    compressed_length = int(original_length * compress_ratio)
    print(f"📊 原始长度: {original_length} tokens")
    print(f"📊 压缩后长度: {compressed_length} tokens")
    print(f"📊 节省显存: {(1-compress_ratio)*100:.1f}%")
    print()
    
    # 步骤2: 预留固定位置
    remaining_positions = compressed_length - sink_size - recent_size
    print(f"🔒 SINK缓存 (位置0-{sink_size-1}): {sink_size} tokens")
    print(f"🔒 RECENT缓存 (最后{recent_size}个): {recent_size} tokens") 
    print(f"🔄 可分配位置: {remaining_positions} tokens")
    print()
    
    # 步骤3: 分配策略
    topk_positions = int(remaining_positions * topk_ratio)
    recent_positions = int(remaining_positions * recent_ratio)
    
    print(f"⭐ 重要性分配 (TOPK): {topk_positions} tokens")
    print(f"🕒 时间分配 (RECENT): {recent_positions} tokens")
    
    # 注意：topk和recent可能有重叠
    overlap = topk_positions + recent_positions - remaining_positions
    if overlap > 0:
        print(f"🔄 重叠部分: {overlap} tokens (既重要又最近)")
    
    print()
    print("📋 最终分配总结:")
    print(f"   - SINK缓存: {sink_size} tokens (永久保留)")
    print(f"   - RECENT缓存: {recent_size} tokens (最新状态)")
    print(f"   - 重要token: ~{topk_positions} tokens (高注意力)")
    print(f"   - 较新token: ~{recent_positions} tokens (时间局部性)")
    print(f"   - 总计: {compressed_length} tokens")

def compare_strategies():
    """比较不同策略的效果"""
    print("\n🔄 不同策略对比")
    print("=" * 50)
    
    strategies = [
        {"name": "保守策略", "compress": 0.2, "topk": 0.7, "recent": 0.3},
        {"name": "激进策略", "compress": 0.05, "topk": 0.3, "recent": 0.7}, 
        {"name": "当前策略", "compress": 0.1, "topk": 0.5, "recent": 0.5},
    ]
    
    original_length = 10000
    
    for strategy in strategies:
        compressed = int(original_length * strategy["compress"])
        remaining = compressed - 32 - 32  # 减去SINK和RECENT
        topk_tokens = int(remaining * strategy["topk"])
        recent_tokens = int(remaining * strategy["recent"])
        
        print(f"\n📊 {strategy['name']}:")
        print(f"   总tokens: {compressed} ({strategy['compress']*100}%)")
        print(f"   重要性导向: {topk_tokens} tokens")
        print(f"   时间导向: {recent_tokens} tokens")
        print(f"   显存节省: {(1-strategy['compress'])*100:.1f}%")

if __name__ == "__main__":
    demonstrate_cache_allocation()
    compare_strategies() 