#!/bin/bash
set -x

# 确保激活正确的conda环境
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate pqcache

# 验证环境
echo "当前conda环境: $CONDA_DEFAULT_ENV"
which python

# SCBench RepoQA测试配置（支持增量保存）
TASK="scbench_repoqa"
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="results_scbench"
MAX_SEQ_LENGTH=100000
MAX_NEW_TOKENS=1024
NUM_EVAL_EXAMPLES=88  # 完整评估
MAX_TURNS=5  # 完整轮次

# PQCache参数
COMPRESS_RATIO=0.1
IMPORTANT_RATIO=0.5
RECENT_RATIO=0.5
SINK_SIZE=16
N_SUBVEC_PER_HEAD=2
N_SUBBITS=6
DEVICE_ID=5  # 使用GPU 5
PP_SIZE=1

# 设置所有必需的环境变量
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
export SUBVEC=${N_SUBVEC_PER_HEAD}
export SUBBITS=${N_SUBBITS}
export METRIC=euc

echo "启动SCBench RepoQA评估（支持增量保存）"
echo "评估参数:"
echo "  - 任务: ${TASK}"
echo "  - 样本数: ${NUM_EVAL_EXAMPLES}"
echo "  - 轮次数: ${MAX_TURNS}"
echo "  - GPU: ${DEVICE_ID}"
echo "  - 增量保存: 启用"

# 运行完整评估
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

echo "增量结果文件: pred/llama-3.1/${TASK}/scbench_official/incremental_compress_${COMPRESS_RATIO}_important_${IMPORTANT_RATIO}_recent_${RECENT_RATIO}_subvec_${N_SUBVEC_PER_HEAD}_subbits_${N_SUBBITS}.jsonl"

# 显示结果统计
if [ -f "pred/llama-3.1/${TASK}/scbench_official/incremental_compress_${COMPRESS_RATIO}_important_${IMPORTANT_RATIO}_recent_${RECENT_RATIO}_subvec_${N_SUBVEC_PER_HEAD}_subbits_${N_SUBBITS}.jsonl" ]; then
    echo "  总轮次数: $(wc -l < pred/llama-3.1/${TASK}/scbench_official/incremental_compress_${COMPRESS_RATIO}_important_${IMPORTANT_RATIO}_recent_${RECENT_RATIO}_subvec_${N_SUBVEC_PER_HEAD}_subbits_${N_SUBBITS}.jsonl)"
    echo "  完成样本数: $(jq -r '.id' pred/llama-3.1/${TASK}/scbench_official/incremental_compress_${COMPRESS_RATIO}_important_${IMPORTANT_RATIO}_recent_${RECENT_RATIO}_subvec_${N_SUBVEC_PER_HEAD}_subbits_${N_SUBBITS}.jsonl 2>/dev/null | sort -u | wc -l)"
fi

# nohup ./run_scbench_repoqa_incremental.sh > scbench_run.log 2>&1 &
# cuda 5