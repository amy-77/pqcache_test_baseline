#!/usr/bin/env python3
"""
分析SCBench数据结构的脚本
详细检查第一条数据的所有key和value
"""

import json

def analyze_scbench_data():
    """分析SCBench数据的第一条记录"""
    
    # 读取数据
    data_file = '/home/pai/data/PQCache/data/scbench/scbench_repoqa.json'
    
    with open(data_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        data = json.loads(first_line)
    
    # 创建分析结果
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("SCBench数据结构分析 - 第一条记录")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    def analyze_recursive(obj, prefix="", level=0):
        """递归分析数据结构"""
        indent = "  " * level
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                # 写key
                output_lines.append(f"{indent}Key: {prefix}{key}")
                
                # 分析value的类型和内容
                if isinstance(value, str):
                    # 字符串类型，显示长度和前200字符
                    output_lines.append(f"{indent}Value (str, len={len(value)}): {repr(value[:200])}{'...' if len(value) > 200 else ''}")
                elif isinstance(value, (int, float, bool)):
                    # 基本类型
                    output_lines.append(f"{indent}Value ({type(value).__name__}): {value}")
                elif isinstance(value, list):
                    # 列表类型
                    output_lines.append(f"{indent}Value (list, len={len(value)}):")
                    if len(value) > 0:
                        output_lines.append(f"{indent}  List item type: {type(value[0]).__name__}")
                        # 分析列表的每个元素
                        for i, item in enumerate(value):
                            output_lines.append(f"{indent}  === List Item {i+1} ===")
                            analyze_recursive(item, f"{prefix}{key}[{i}].", level+2)
                    else:
                        output_lines.append(f"{indent}  (empty list)")
                elif isinstance(value, dict):
                    # 嵌套字典
                    output_lines.append(f"{indent}Value (dict, len={len(value)}):")
                    analyze_recursive(value, f"{prefix}{key}.", level+1)
                else:
                    # 其他类型
                    output_lines.append(f"{indent}Value ({type(value).__name__}): {repr(value)}")
                
                output_lines.append("")  # 空行分隔
                
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                output_lines.append(f"{indent}List Item {i+1}:")
                analyze_recursive(item, f"{prefix}[{i}].", level+1)
        else:
            output_lines.append(f"{indent}Value ({type(obj).__name__}): {repr(obj)}")
    
    # 开始分析
    analyze_recursive(data)
    
    # 特别分析multi_turns
    output_lines.append("\n" + "=" * 60)
    output_lines.append("特别分析 multi_turns 结构")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    if 'multi_turns' in data:
        multi_turns = data['multi_turns']
        output_lines.append(f"multi_turns 包含 {len(multi_turns)} 个轮次")
        output_lines.append("")
        
        for i, turn in enumerate(multi_turns):
            output_lines.append(f"=== 轮次 {i+1} ===")
            if isinstance(turn, dict):
                for key, value in turn.items():
                    output_lines.append(f"Key: {key}")
                    if isinstance(value, str):
                        if len(value) > 500:
                            output_lines.append(f"Value (str, len={len(value)}): {repr(value[:200])}...{repr(value[-100:])}")
                        else:
                            output_lines.append(f"Value (str, len={len(value)}): {repr(value)}")
                    else:
                        output_lines.append(f"Value ({type(value).__name__}): {repr(value)}")
                    output_lines.append("")
            output_lines.append("")
    
    # 写入文件
    output_file = '/home/pai/data/PQCache/scbench_data_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"分析完成！结果已保存到: {output_file}")
    print(f"总共写入 {len(output_lines)} 行")
    
    # 也打印一些关键信息到控制台
    print("\n=== 关键信息概览 ===")
    print(f"顶级keys: {list(data.keys())}")
    if 'multi_turns' in data:
        print(f"multi_turns轮数: {len(data['multi_turns'])}")
        print(f"每轮的keys: {[list(turn.keys()) if isinstance(turn, dict) else type(turn) for turn in data['multi_turns']]}")




if __name__ == "__main__":
    analyze_scbench_data()