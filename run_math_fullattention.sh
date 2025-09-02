#!/bin/bash
# Full-Attention AIME 2024 数学推理评估
# 对比 PQCache 和 Full-Attention 的性能差异

set -e

# 确保激活正确的conda环境
source /home/pai/data/miniconda3/etc/profile.d/conda.sh
conda activate pqcache

echo "=========================================="
echo "Full-Attention Mathematics CoT - AIME 2024"
echo "数据集: AIME 2024"
echo "模型: Llama-3.1-8B-Instruct"
echo "任务: Full-Attention Chain-of-Thought Mathematical Reasoning"
echo "=========================================="

# 任务配置
DATASET="aime_2024"
MODEL="llama-3.1"
OUTPUT_DIR="math_cot_results"
MAX_SEQ_LENGTH=32768
MAX_NEW_TOKENS=2048
NUM_EVAL_EXAMPLES=-1  # 所有样本
TEMPERATURE=0.0  # 确定性生成

# Method configuration - 重点：使用full-attention
METHOD="fullattention"
COMPRESSOR="original"  # original表示不压缩
DEVICE_ID=2

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

echo "模型: ${MODEL}"
echo "数据集: ${DATASET}"
echo "设备: CUDA:${DEVICE_ID}"
echo "方法: ${METHOD} (compressor: ${COMPRESSOR})"
echo "最大新token数: ${MAX_NEW_TOKENS}"
echo "评估样本数: ${NUM_EVAL_EXAMPLES}"
echo ""

# 运行评估
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
TOKENIZERS_PARALLELISM=false \
python run_math_cot_fullattention.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --method ${METHOD} \
    --compressor ${COMPRESSOR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p 1.0 \
    --num_eval_examples ${NUM_EVAL_EXAMPLES} \
    --device_id ${DEVICE_ID} \
    --output_dir ${OUTPUT_DIR} \
    --verbose

echo ""
echo "✅ Full-Attention评估完成！"

# 显示结果文件位置
RESULT_FILE="${OUTPUT_DIR}/${MODEL}/${DATASET}/fullattention_cot/cot_results_fullattention.jsonl"
if [ -f "$RESULT_FILE" ]; then
    echo "📊 结果文件: $RESULT_FILE"
    echo "📈 样本数量: $(wc -l < "$RESULT_FILE")"
    
    # 快速统计正确率
    python -c "
import json
correct = 0
total = 0
with open('$RESULT_FILE', 'r') as f:
    for line in f:
        result = json.loads(line.strip())
        total += 1
        if result.get('is_correct', False):
            correct += 1
print(f'🎯 快速统计 - 正确: {correct}/{total}, 准确率: {correct/total:.4f} ({correct/total*100:.2f}%)')
"
else
    echo "❌ 结果文件未找到: $RESULT_FILE"
fi

echo ""
echo "🔍 要进行详细分析，请运行:"
echo "python eval_aime_2024_results.py --results_file $RESULT_FILE --detailed_analysis"

# nohup ./run_math_fullattention.sh > math_fullattention.log 2>&1 &

#4003186