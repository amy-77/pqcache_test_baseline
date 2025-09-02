#!/bin/bash
# PQCache MATH-500 数学推理评估
# 对比Full-Attention在MATH-500数据集上的表现

set -e

# 确保激活正确的conda环境
source /home/pai/data/miniconda3/etc/profile.d/conda.sh
conda activate pqcache

echo "=========================================="
echo "PQCache Mathematics CoT - MATH-500"
echo "数据集: MATH-500 (高中标准数学)"
echo "模型: Llama-3.1-8B-Instruct"
echo "任务: PQCache Chain-of-Thought Mathematical Reasoning"
echo "=========================================="

# 任务配置
DATASET="math_500"
MODEL="llama-3.1"
OUTPUT_DIR="math_cot_results"
MAX_SEQ_LENGTH=32768
MAX_NEW_TOKENS=2048
NUM_EVAL_EXAMPLES=-1  # 与Full-Attention保持一致
TEMPERATURE=0.0  # 确定性生成

# Method configuration - 使用PQCache
METHOD="pqcache"
COMPRESS_RATIO=0.1
IMPORTANT_RATIO=0.5
RECENT_RATIO=0.5
N_SUBVEC_PER_HEAD=2
N_SUBBITS=6
DEVICE_ID=3

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export SUBVEC=${N_SUBVEC_PER_HEAD}
export SUBBITS=${N_SUBBITS}
export METRIC=euc

echo "模型: ${MODEL}"
echo "数据集: ${DATASET}"
echo "设备: CUDA:${DEVICE_ID}"
echo "压缩比例: ${COMPRESS_RATIO}"
echo "重要token比例: ${IMPORTANT_RATIO}"
echo "最近token比例: ${RECENT_RATIO}"
echo "子向量数: ${N_SUBVEC_PER_HEAD}, 子比特数: ${N_SUBBITS}"
echo "方法: ${METHOD}"
echo "最大新token数: ${MAX_NEW_TOKENS}"
echo "评估样本数: ${NUM_EVAL_EXAMPLES}"
echo ""

# 运行评估
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
TOKENIZERS_PARALLELISM=false \
python run_math_cot.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --compress_ratio ${COMPRESS_RATIO} \
    --important_ratio ${IMPORTANT_RATIO} \
    --recent_ratio ${RECENT_RATIO} \
    --n_subvec_per_head ${N_SUBVEC_PER_HEAD} \
    --n_subbits ${N_SUBBITS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p 1.0 \
    --num_eval_examples ${NUM_EVAL_EXAMPLES} \
    --device_id ${DEVICE_ID} \
    --output_dir ${OUTPUT_DIR} \
    --enable_vq_cache \
    --verbose

echo ""
echo "✅ MATH-500 PQCache评估完成！"

# 显示结果文件位置
RESULT_FILE="${OUTPUT_DIR}/${MODEL}/${DATASET}/pqcache_cot/cot_results_compress_${COMPRESS_RATIO}_ratio_${IMPORTANT_RATIO}_${RECENT_RATIO}.jsonl"
if [ -f "$RESULT_FILE" ]; then
    echo "📊 结果文件: $RESULT_FILE"
    echo "📈 样本数量: $(wc -l < "$RESULT_FILE")"
    
    # 快速统计正确率
    python -c "
import json
correct = 0
total = 0
total_time = 0
with open('$RESULT_FILE', 'r') as f:
    for line in f:
        result = json.loads(line.strip())
        total += 1
        if result.get('is_correct', False):
            correct += 1
        if 'generation_time' in result:
            total_time += result['generation_time']

avg_time = total_time / total if total > 0 else 0
print(f'🎯 PQCache MATH-500统计 - 正确: {correct}/{total}, 准确率: {correct/total:.4f} ({correct/total*100:.2f}%)')
print(f'⏱️  平均生成时间: {avg_time:.2f}秒')
print(f'📊 对比基线: AIME 2024准确率仅为3.33%, 生成时间441秒')
"
else
    echo "❌ 结果文件未找到: $RESULT_FILE"
fi

echo ""
echo "🔍 要对比Full-Attention结果，请运行:"
echo "python -c \""
echo "import json"
echo "# 比较两种方法在MATH-500上的表现"
echo "print('=== MATH-500 方法对比 ===')"
echo "\""
