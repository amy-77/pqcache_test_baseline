#!/usr/bin/env python3
"""
具体例子：KV缓存token选择机制
演示TOPK和RECENT_RATIO如何工作
"""

import numpy as np

def create_concrete_example():
    """创建一个具体的例子"""
    print("🎯 具体例子：科学论文问答场景")
    print("=" * 60)
    
    # 模拟一个科学论文的token序列
    tokens = [
        # 位置0-31: SINK区域 (永远保留)
        "Abstract:", "This", "paper", "presents", "a", "novel", "approach", "...",
        "[SINK区域共32个token - 永远保留]",
        
        # 位置32-8999: 中间的长文档内容
        "Introduction", "section", "describes", "the", "problem", "...",
        "Method", "section", "explains", "our", "algorithm", "...", 
        "Results", "show", "significant", "improvement", "...",
        "Discussion", "reveals", "important", "insights", "...",
        "[中间8968个token的长文档内容]",
        
        # 位置9000-9967: 问题和开始生成的回答
        "Question:", "What", "is", "the", "main", "contribution", "of", "this", "paper?",
        "Answer:", "The", "main", "contribution", "is", "a", "novel", "...",
        
        # 位置9968-9999: RECENT区域 (最近32个token - 永远保留)
        "algorithm", "that", "significantly", "improves", "performance", "by", "...",
        "[RECENT区域最后32个token - 永远保留]"
    ]
    
    # 模拟注意力权重 (数值越大表示越重要)
    attention_weights = {
        # SINK区域 - 不参与选择，直接保留
        0: 0.0,   # "Abstract:" - SINK
        5: 0.0,   # "novel" - SINK  
        
        # 中间区域 - 参与重要性选择
        32: 0.95,   # "Introduction" - 非常重要!
        45: 0.85,   # "algorithm" - 很重要
        123: 0.90,  # "Method" - 很重要  
        234: 0.30,  # "the" - 不重要
        345: 0.25,  # "and" - 不重要
        456: 0.88,  # "Results" - 很重要
        567: 0.20,  # "of" - 不重要
        678: 0.82,  # "significant" - 重要
        789: 0.15,  # "a" - 不重要
        890: 0.87,  # "improvement" - 很重要
        
        # 问题和答案区域
        9005: 0.92,  # "contribution" - 问题关键词
        9010: 0.88,  # "Answer:" - 重要
        9015: 0.85,  # "novel" - 重要
        
        # RECENT区域 - 不参与选择，直接保留
        9968: 0.0,  # "algorithm" - RECENT
        9999: 0.0,  # 最后一个token - RECENT
    }
    
    return tokens, attention_weights

def demonstrate_selection_process():
    """演示选择过程"""
    tokens, attention_weights = create_concrete_example()
    
    # 参数设置
    total_length = 10000
    compress_ratio = 0.1
    sink_size = 32
    recent_size = 32
    topk_ratio = 0.5
    recent_ratio = 0.5
    
    compressed_length = int(total_length * compress_ratio)  # 1000
    available_positions = compressed_length - sink_size - recent_size  # 936
    
    print(f"📊 原始长度: {total_length} tokens")
    print(f"📊 压缩后: {compressed_length} tokens")
    print(f"📊 可分配位置: {available_positions} tokens")
    print()
    
    # 步骤1: 固定保留区域
    print("🔒 第1步: 固定保留区域")
    print(f"   SINK区域 (位置0-{sink_size-1}): 永远保留")
    print(f"   RECENT区域 (位置{total_length-recent_size}-{total_length-1}): 永远保留") 
    print()
    
    # 步骤2: 候选token池 (除了SINK和RECENT的所有token)
    candidate_positions = list(range(sink_size, total_length - recent_size))
    print(f"🏊 第2步: 候选token池")
    print(f"   候选范围: 位置{sink_size}到{total_length-recent_size-1}")
    print(f"   候选总数: {len(candidate_positions)} tokens")
    print()
    
    # 步骤3: 按重要性排序 (TOPK选择)
    print("⭐ 第3步: 重要性选择 (TOPK)")
    
    # 模拟计算每个候选位置的注意力权重
    weighted_candidates = []
    for pos in candidate_positions[:20]:  # 只显示前20个作为例子
        weight = attention_weights.get(pos, np.random.random() * 0.5)  # 随机权重作为示例
        weighted_candidates.append((pos, weight))
    
    # 按权重排序
    weighted_candidates.sort(key=lambda x: x[1], reverse=True)
    
    topk_count = int(available_positions * topk_ratio)  # 468个
    print(f"   需要选择最重要的: {topk_count} tokens")
    print(f"   最重要的几个token:")
    
    for i, (pos, weight) in enumerate(weighted_candidates[:8]):
        if pos in attention_weights:
            token_desc = f"位置{pos} (权重{weight:.2f}) - 重要关键词"
        else:
            token_desc = f"位置{pos} (权重{weight:.2f}) - 普通token"
        print(f"     #{i+1}: {token_desc}")
    
    print(f"   ... (总共选择{topk_count}个最重要的)")
    print()
    
    # 步骤4: 按时间选择 (RECENT_RATIO选择)
    print("🕒 第4步: 时间局部性选择 (RECENT_RATIO)")
    
    recent_count = int(available_positions * recent_ratio)  # 468个
    print(f"   需要选择最近的: {recent_count} tokens")
    
    # 从候选池中选择最近的token (位置越大越近)
    recent_candidates = sorted(candidate_positions, reverse=True)[:recent_count]
    
    print(f"   选择的最近token范围:")
    print(f"     从位置{recent_candidates[-1]}到位置{recent_candidates[0]}")
    print(f"     这些是问题和答案开始部分，以及文档结尾部分")
    print()
    
    # 步骤5: 合并和去重
    print("🔄 第5步: 合并结果")
    
    # 重要性选择的token
    topk_selected = [pos for pos, _ in weighted_candidates[:topk_count]]
    
    # 时间选择的token  
    recent_selected = recent_candidates[:recent_count]
    
    # 合并并去重
    all_selected = list(set(topk_selected + recent_selected))
    overlap_count = len(topk_selected) + len(recent_selected) - len(all_selected)
    
    print(f"   重要性选择: {len(topk_selected)} tokens")
    print(f"   时间选择: {len(recent_selected)} tokens") 
    print(f"   重叠部分: {overlap_count} tokens (既重要又最近)")
    print(f"   最终选择: {len(all_selected)} tokens")
    print()
    
    # 步骤6: 最终分配总结
    print("📋 第6步: 最终缓存分配")
    total_kept = sink_size + recent_size + len(all_selected)
    print(f"   🔒 SINK缓存: {sink_size} tokens (文档开头)")
    print(f"   🔒 RECENT缓存: {recent_size} tokens (最新生成)")
    print(f"   ⭐ 重要token: ~{len([p for p in topk_selected if p in all_selected])} tokens (高注意力)")
    print(f"   🕒 较新token: ~{len([p for p in recent_selected if p in all_selected])} tokens (时间局部性)")
    print(f"   📊 总计: {total_kept} tokens (目标{compressed_length})")

def show_intuitive_example():
    """更直观的例子"""
    print("\n" + "="*60)
    print("🎯 更直观的例子：对话场景")
    print("="*60)
    
    print("假设有一个10000 token的对话，压缩到1000个token:")
    print()
    
    conversation = [
        "位置0-31: [系统提示] 你是一个AI助手... (SINK区域，永远保留)",
        "位置32-5000: [用户] 请详细解释量子计算的原理...",
        "位置5001-8000: [AI] 量子计算是基于量子力学原理... (长回答)",
        "位置8001-9000: [用户] 那么量子纠缠是怎么工作的?",  
        "位置9001-9968: [AI] 量子纠缠是一种奇特的现象...",
        "位置9968-9999: [AI] ...因此量子纠缠... (RECENT区域，永远保留)"
    ]
    
    print("📚 对话内容:")
    for line in conversation:
        print(f"  {line}")
    print()
    
    print("🔍 选择策略:")
    print("  1. SINK (32个): 系统提示永远保留")
    print("  2. RECENT (32个): 最新回答永远保留") 
    print("  3. 剩余936个位置分配:")
    print("     - 重要性50% (468个): 选择'量子计算'、'量子纠缠'等关键词")
    print("     - 时间性50% (468个): 选择最近的问答内容")
    print("     - 重叠: 一些token既重要又最近，所以总数<936")
    print()
    
    print("💡 这样既保留了:")
    print("  ✓ 对话上下文 (SINK)")
    print("  ✓ 最新状态 (RECENT)")  
    print("  ✓ 关键信息 (TOPK)")
    print("  ✓ 局部连贯性 (RECENT_RATIO)")

if __name__ == "__main__":
    demonstrate_selection_process()
    show_intuitive_example() 