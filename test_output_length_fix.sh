#!/bin/bash
set -x

# 确保激活正确的conda环境
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate pqcache

echo "测试修复后的输出长度计算..."

# 设置环境变量
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=5
export SUBVEC=2
export SUBBITS=4
export METRIC=euc

# 快速测试参数
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
TOKENIZERS_PARALLELISM=false \
python vq_pred_scbench_official.py \
    --task scbench_repoqa \
    --model llama-3.1 \
    --max_seq_length 20000 \
    --max_new_tokens 100 \
    --num_eval_examples 1 \
    --max_turns 1 \
    --compress_ratio 0.2 \
    --important_ratio 0.5 \
    --recent_ratio 0.5 \
    --sink-size 8 \
    --n_subvec_per_head 2 \
    --n_subbits 4 \
    --enable_vq_cache \
    --compressor pq_search \
    --fp16 \
    --device_id 5 \
    --pp-size 1 \
    --exp_name "output_length_test"

echo "测试完成！检查结果文件..."
cat pred/llama-3.1/scbench_repoqa/output_length_test/compress_0.2_important_0.5_recent_0.5_subvec_2_subbits_4.jsonl
