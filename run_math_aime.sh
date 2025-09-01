#!/bin/bash
# PQCache Mathematics CoT Reasoning - AIME 2024 Dataset
# 使用 PQCache 方法在 Llama-3.1-8B 上进行 AIME 2024 数学推理

set -e

# 确保激活正确的conda环境
source /home/pai/data/miniconda3/etc/profile.d/conda.sh
conda activate pqcache

echo "=========================================="
echo "PQCache Mathematics CoT - AIME 2024"
echo "数据集: HuggingFaceH4/aime_2024"
echo "模型: Llama-3.1-8B-Instruct"
echo "任务: Chain-of-Thought Mathematical Reasoning"
echo "=========================================="

# 任务配置
DATASET="aime_2024"
MODEL="llama-3.1"
OUTPUT_DIR="math_cot_results"
MAX_SEQ_LENGTH=32768
MAX_NEW_TOKENS=2048  # 足够长的CoT推理
NUM_EVAL_EXAMPLES=-1  # 所有样本，可以设置为较小数字进行测试
TEMPERATURE=0.0  # 确定性生成，用于数学推理

# Method configuration
METHOD="pqcache"

# PQCache参数
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
    --method ${METHOD} \
    --num_eval_examples ${NUM_EVAL_EXAMPLES} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --compress_ratio ${COMPRESS_RATIO} \
    --important_ratio ${IMPORTANT_RATIO} \
    --recent_ratio ${RECENT_RATIO} \
    --n_subvec_per_head ${N_SUBVEC_PER_HEAD} \
    --n_subbits ${N_SUBBITS} \
    --compressor pq_search \
    --enable_vq_cache \
    --device_id ${DEVICE_ID} \
    --output_dir ${OUTPUT_DIR} \
    --verbose

echo ""
echo "=========================================="
echo "AIME 2024 数学推理评估完成！"
echo "结果保存在: ${OUTPUT_DIR}/${MODEL}/${DATASET}/pqcache_cot/"
echo "=========================================="

#  nohup ./run_math_aime.sh > math_aime.log 2>&1 &
#[8] 4041575  cuda 3