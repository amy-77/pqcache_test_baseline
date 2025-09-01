#!/bin/bash
# SCBench 所有任务管理脚本 - PQCache评估
# 用于运行所有分类的SCBench任务

set -e

echo "================================================================"
echo "SCBench PQCache 完整评估脚本"
echo "================================================================"
echo "本脚本将按任务类别运行所有SCBench任务评估"
echo ""

# 任务分类定义
declare -A TASK_CATEGORIES
TASK_CATEGORIES[long_generation_decode_shift]="run_scbench_summary.sh run_scbench_summary_with_needles.sh"
TASK_CATEGORIES[strong_drift]="run_scbench_qa_eng.sh run_scbench_qa_chn.sh run_scbench_choice_eng.sh run_scbench_mf.sh"
TASK_CATEGORIES[multiturn_kv_drift]="run_scbench_kv.sh run_scbench_prefix_suffix.sh run_scbench_vt.sh"
TASK_CATEGORIES[global_processing]="run_scbench_many_shot.sh"

# 显示可用选项
show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  all                        - 运行所有任务"
    echo "  long_generation_decode_shift - 长生成导致解码内注意力迁移"
    echo "    ├── scbench_summary (En.Sum)"
    echo "    └── scbench_summary_with_needles (Mix.Sum+NIAH)"
    echo ""
    echo "  strong_drift              - 强漂移任务"
    echo "    ├── scbench_qa_eng (En.QA)"
    echo "    ├── scbench_qa_chn (Zh.QA)"
    echo "    ├── scbench_choice_eng (En.MultiChoice)"
    echo "    └── scbench_mf (Math.Find)"
    echo ""
    echo "  multiturn_kv_drift        - 多轮/多请求下KV漂移"
    echo "    ├── scbench_kv (Retr.KV)"
    echo "    ├── scbench_prefix_suffix (Retr.Prefix-Suffix)"
    echo "    └── scbench_vt (Retr.MultiHop)"
    echo ""
    echo "  global_processing         - 全局信息处理"
    echo "    └── scbench_many_shot (Many-shot ICL)"
    echo ""
    echo "  单个任务脚本:"
    echo "    ./run_scbench_<task_name>.sh"
    echo ""
}

# 运行单个脚本
run_script() {
    local script=$1
    local task_name=$(echo $script | sed 's/run_scbench_//' | sed 's/.sh//')
    
    echo ""
    echo "▶ 开始运行: $script"
    echo "任务: $task_name"
    echo "时间: $(date)"
    echo "----------------------------------------"
    
    if [ -f "$script" ]; then
        chmod +x "$script"
        ./"$script"
        if [ $? -eq 0 ]; then
            echo "✅ $script 完成"
        else
            echo "❌ $script 失败"
        fi
    else
        echo "❌ 脚本文件不存在: $script"
    fi
    
    echo "----------------------------------------"
}

# 运行任务类别
run_category() {
    local category=$1
    local scripts=${TASK_CATEGORIES[$category]}
    
    if [ -z "$scripts" ]; then
        echo "❌ 未知的任务类别: $category"
        show_usage
        exit 1
    fi
    
    echo ""
    echo "🚀 开始运行任务类别: $category"
    echo "包含脚本: $scripts"
    echo ""
    
    for script in $scripts; do
        run_script "$script"
        echo "等待3秒后继续下一个任务..."
        sleep 3
    done
    
    echo ""
    echo "🎉 任务类别 $category 全部完成！"
}

# 运行所有任务
run_all() {
    echo ""
    echo "🚀 开始运行所有SCBench任务..."
    echo ""
    
    for category in long_generation_decode_shift strong_drift multiturn_kv_drift global_processing; do
        echo ""
        echo "==============================================="
        echo "运行任务类别: $category"
        echo "==============================================="
        run_category "$category"
        echo ""
        echo "等待10秒后运行下一个类别..."
        sleep 10
    done
    
    echo ""
    echo "🎉🎉🎉 所有SCBench任务评估完成！ 🎉🎉🎉"
    echo "结果保存在: pred_generic/llama-3.1/*/pqcache_official/"
}

# 主逻辑
case "${1:-help}" in
    "all")
        run_all
        ;;
    "long_generation_decode_shift"|"strong_drift"|"multiturn_kv_drift"|"global_processing")
        run_category "$1"
        ;;
    "help"|"-h"|"--help"|"")
        show_usage
        ;;
    *)
        echo "❌ 未知选项: $1"
        show_usage
        exit 1
        ;;
esac