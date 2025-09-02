#!/bin/bash
# Full-Attention MATH-500 数学推理评估
# 对比AIME 2024的难度，验证模型在相对简单数学题上的表现

set -e

# 确保激活正确的conda环境
source /home/pai/data/miniconda3/etc/profile.d/conda.sh
conda activate pqcache

echo "=========================================="
echo "Full-Attention Mathematics CoT - MATH-500"
echo "数据集: MATH-500 (高中标准数学)"
echo "模型: Llama-3.1-8B-Instruct"
echo "任务: Full-Attention Chain-of-Thought Mathematical Reasoning"
echo "=========================================="

# 任务配置
DATASET="math_500"
MODEL="llama-3.1"
OUTPUT_DIR="math_cot_results"
MAX_SEQ_LENGTH=32768
MAX_NEW_TOKENS=2048
NUM_EVAL_EXAMPLES=20  # 先测试20个样本，验证效果
TEMPERATURE=0.0  # 确定性生成

# Method configuration - 使用full-attention
METHOD="fullattention"
COMPRESSOR="original"  # original表示不压缩
DEVICE_ID=3

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
echo "题目难度: 高中标准数学 (比AIME竞赛简单)"
echo ""

# 运行评估
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
TOKENIZERS_PARALLELISM=false \
python run_math500_cot_fullattention.py \
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
echo "✅ MATH-500 Full-Attention评估完成！"

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
print(f'🎯 MATH-500快速统计 - 正确: {correct}/{total}, 准确率: {correct/total:.4f} ({correct/total*100:.2f}%)')
print(f'📊 对比AIME 2024: 之前AIME准确率仅为3.33% (1/30)')
"
else
    echo "❌ 结果文件未找到: $RESULT_FILE"
fi

echo ""
echo "🔍 要进行详细分析，请运行:"
echo "python eval_aime_2024_results.py --results_file $RESULT_FILE --detailed_analysis"

echo ""
echo "📝 实验目的验证："
echo "1. MATH-500相对AIME 2024是否更容易？"
echo "2. Llama-3.1-8B在标准高中数学上的基线表现如何？"
echo "3. 验证PQCache vs Full-Attention在合适难度数据集上的对比效果"
