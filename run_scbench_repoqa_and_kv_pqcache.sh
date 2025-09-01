#!/bin/bash
set -x

# 确保激活正确的conda环境
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate pqcache

# 验证环境
echo "当前conda环境: $CONDA_DEFAULT_ENV"
which python

# SCBench测试配置
TASK="scbench_repoqa_and_kv"  # 或 scbench_repoqa_and_kv
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="results_scbench_repoqa_and_kv"
MAX_SEQ_LENGTH=100000
MAX_NEW_TOKENS=1024
NUM_EVAL_EXAMPLES=88  # 先用2个样本测试
MAX_TURNS=8  # 先用1轮对话测试

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
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
export SUBVEC=${N_SUBVEC_PER_HEAD}
export SUBBITS=${N_SUBBITS}
export METRIC=euc

# 运行测试
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
TOKENIZERS_PARALLELISM=false \
python vq_pred_scbench_official.py \
    --task ${TASK} \
    --model llama-3.1 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --num_eval_examples ${NUM_EVAL_EXAMPLES} \
    --max_turns ${MAX_TURNS} \
    --compress_ratio ${COMPRESS_RATIO} \
    --important_ratio ${IMPORTANT_RATIO} \
    --recent_ratio ${RECENT_RATIO} \
    --sink-size ${SINK_SIZE} \
    --n_subvec_per_head ${N_SUBVEC_PER_HEAD} \
    --n_subbits ${N_SUBBITS} \
    --enable_vq_cache \
    --compressor pq_search \
    --fp16 \
    --device_id ${DEVICE_ID} \
    --pp-size ${PP_SIZE}

echo "SCBench PQCache测试完成！"
echo "结果保存在: ${OUTPUT_DIR}/${TASK}_results.jsonl"




#  nohup ./run_scbench_repoqa_and_kv_pqcache.sh > scbench_run_repoqa_and_kv.log 2>&1 &
#[2] 2758090