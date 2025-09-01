#!/bin/bash
# SCBench Chinese QA task evaluation script for PQCache
# Task: scbench_qa_chn - Chinese question answering (strong drift)

set -e

# 确保激活正确的conda环境
source /home/pai/data/miniconda3/etc/profile.d/conda.sh || source /home/pai/data/miniconda3/etc/profile.d/conda.sh
conda activate pqcache

echo "=========================================="
echo "SCBench Chinese QA Task (scbench_qa_chn)"
echo "任务类型: 强漂移 - 中文问答"
echo "数据集大小: 35 samples"
echo "多轮对话: 2-4 turns"
echo "上下文类型: Chinese texts"
echo "=========================================="

# 任务配置
TASK="scbench_qa_chn"
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="pred_scbench_qa_chn"
MAX_SEQ_LENGTH=131072
MAX_NEW_TOKENS=40  # 中文问答的标准长度（简短回答）
NUM_EVAL_EXAMPLES=-1  # 测试用，可以设置为-1运行全部
MAX_TURNS=-1  # -1表示使用数据集中的所有轮次

# PQCache参数
COMPRESS_RATIO=0.1
IMPORTANT_RATIO=0.5
RECENT_RATIO=0.5
SINK_SIZE=16
N_SUBVEC_PER_HEAD=2
N_SUBBITS=6
DEVICE_ID=4
PP_SIZE=1

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export SUBVEC=${N_SUBVEC_PER_HEAD}
export SUBBITS=${N_SUBBITS}
export METRIC=euc

echo "模型路径: ${MODEL_NAME_OR_PATH}"
echo "设备: CUDA:${DEVICE_ID}"
echo "压缩比例: ${COMPRESS_RATIO}"
echo "重要token比例: ${IMPORTANT_RATIO}"
echo "最近token比例: ${RECENT_RATIO}"
echo "子向量数: ${N_SUBVEC_PER_HEAD}, 子比特数: ${N_SUBBITS}"
echo "评估样本数: ${NUM_EVAL_EXAMPLES}"
echo ""

# 运行评估

PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
TOKENIZERS_PARALLELISM=false \
python vq_pred_scbench_generic.py \
    --model "llama-3.1" \
    --task "${TASK}" \
    --num_eval_examples ${NUM_EVAL_EXAMPLES} \
    --max_turns ${MAX_TURNS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --compress_ratio ${COMPRESS_RATIO} \
    --important_ratio ${IMPORTANT_RATIO} \
    --recent_ratio ${RECENT_RATIO} \
    --n_subvec_per_head ${N_SUBVEC_PER_HEAD} \
    --n_subbits ${N_SUBBITS} \
    --pp-size ${PP_SIZE} \
    --device_id ${DEVICE_ID} \
    --output_dir "${OUTPUT_DIR}" \
    --enable_vq_cache \
    --verbose

echo ""
echo "=========================================="
echo "SCBench 中文问答任务评估完成！"
echo "结果保存在: ${OUTPUT_DIR}/llama-3.1/${TASK}/pqcache_official/"
echo "=========================================="
