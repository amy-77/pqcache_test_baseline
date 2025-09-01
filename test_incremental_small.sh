#!/bin/bash
# 小规模测试增量保存功能

echo "🧪 测试增量保存功能（1个样本，1轮对话）"

export CUDA_VISIBLE_DEVICES=7
export SUBVEC=2
export SUBBITS=6
export METRIC=euc

python vq_pred_scbench_official.py \
    --task scbench_repoqa \
    --model llama-3.1 \
    --max_seq_length 100000 \
    --max_new_tokens 200 \
    --num_eval_examples 1 \
    --max_turns 1 \
    --compress_ratio 0.1 \
    --important_ratio 0.5 \
    --recent_ratio 0.5 \
    --sink-size 16 \
    --n_subvec_per_head 2 \
    --n_subbits 6 \
    --enable_vq_cache \
    --compressor pq_search \
    --fp16 \
    --device_id 7 \
    --pp-size 1

echo "✅ 测试完成！检查增量文件是否生成:"
ls -la pred/llama-3.1/scbench_repoqa/scbench_official/incremental_*.jsonl 2>/dev/null || echo "❌ 增量文件未生成"
