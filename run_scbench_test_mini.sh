#!/bin/bash
# 轻量级测试版本 - 快速验证增量保存功能
set -x

echo "🧪 启动SCBench轻量级测试（增量保存验证）"

# 确保激活正确的conda环境
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate pqcache

# 验证环境
echo "当前conda环境: $CONDA_DEFAULT_ENV"
which python

# 轻量级测试配置
TASK="scbench_many_shot"
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="pred/test"

# 🔥 关键：大幅减少参数进行快速测试
MAX_SEQ_LENGTH=32768   # many_shot需要较长序列长度，设置为32K
MAX_NEW_TOKENS=10      # many_shot任务的官方配置
NUM_EVAL_EXAMPLES=1    # 只测试1个样本
MAX_TURNS=3            # 只测试1轮对话

# PQCache参数（保持不变）
COMPRESS_RATIO=0.1
IMPORTANT_RATIO=0.5
RECENT_RATIO=0.5
SINK_SIZE=16
N_SUBVEC_PER_HEAD=2
N_SUBBITS=6
DEVICE_ID=7            # 使用GPU 7
PP_SIZE=1

# 设置环境变量
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
export SUBVEC=${N_SUBVEC_PER_HEAD}
export SUBBITS=${N_SUBBITS}
export METRIC=euc

echo "📊 测试配置:"
echo "  - 任务: ${TASK}"
echo "  - 样本数: ${NUM_EVAL_EXAMPLES} (轻量)"
echo "  - 轮次数: ${MAX_TURNS} (轻量)"
echo "  - 最大序列长度: ${MAX_SEQ_LENGTH} (轻量)"
echo "  - 最大生成长度: ${MAX_NEW_TOKENS} (轻量)"
echo "  - GPU: ${DEVICE_ID}"
echo "  - 增量保存: ✅ 启用"

echo "⏱️ 预计测试时间: 2-5分钟"

# 记录开始时间
start_time=$(date +%s)

# 运行轻量级测试
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
TOKENIZERS_PARALLELISM=false \
python vq_pred_scbench_generic.py \
    --task ${TASK} \
    --model llama-3.1 \
    --output_dir ${OUTPUT_DIR} \
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

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "✅ SCBench轻量级测试完成！"
echo "⏱️ 耗时: ${duration} 秒"

# 检查增量保存文件
INCREMENTAL_FILE="${OUTPUT_DIR}/llama-3.1/${TASK}/pqcache_general/incremental_compress_${COMPRESS_RATIO}_important_${IMPORTANT_RATIO}_recent_${RECENT_RATIO}_subvec_${N_SUBVEC_PER_HEAD}_subbits_${N_SUBBITS}.jsonl"

echo ""
echo "📁 检查增量保存结果:"
if [ -f "${INCREMENTAL_FILE}" ]; then
    echo "✅ 增量文件已生成: ${INCREMENTAL_FILE}"
    echo "📊 文件大小: $(du -h "${INCREMENTAL_FILE}" | cut -f1)"
    echo "📝 结果行数: $(wc -l < "${INCREMENTAL_FILE}")"
    echo ""
    echo "📋 结果预览 (前3行):"
    head -3 "${INCREMENTAL_FILE}" | jq -r '. | "样本\(.id) 轮次\(.turn_idx): 生成\(.output_length)字符 耗时\(.generation_time)s"' 2>/dev/null || head -3 "${INCREMENTAL_FILE}"
else
    echo "❌ 增量文件未生成，可能测试失败"
fi

echo ""
echo "🎯 测试完成总结:"
echo "  - 配置: ${NUM_EVAL_EXAMPLES}样本 × ${MAX_TURNS}轮 × ${MAX_NEW_TOKENS}token"
echo "  - 耗时: ${duration}秒"
echo "  - 增量保存: $([ -f "${INCREMENTAL_FILE}" ] && echo "✅ 成功" || echo "❌ 失败")"

if [ -f "${INCREMENTAL_FILE}" ]; then
    echo ""
    echo "🚀 增量保存功能验证成功！可以使用完整配置运行："
    echo "   nohup ./run_scbench_repoqa_incremental.sh > scbench_full.log 2>&1 &"
fi




#./run_scbench_test_mini.sh > debug_test.log 2>&1
