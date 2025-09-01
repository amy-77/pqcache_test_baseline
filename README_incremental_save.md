# SCBench 增量保存功能说明

## 📊 问题诊断结果

**您的SCBench RepoQA任务运行正常，但极其缓慢**

### 当前状态
- ✅ **程序正常运行** - 已完成 **16/88 样本** (18.2%)
- ⏱️ **已运行**: 23.6小时  
- 🎯 **预计完成**: 2025年9月5日 16:30 (还需106小时 ≈ **4.4天**)
- 📊 **平均每样本**: 1.5小时 (89分钟)

### 为什么结果文件为空？
原始的 `vq_pred_scbench_official.py` 只在**所有88个样本完成后**才保存结果，这就是为什么 `results_scbench/scbench_repoqa_results.jsonl` 为空的原因。

## 🛠️ 解决方案：增量保存功能

### 修改内容

1. **修改了 `evaluate_scbench_official` 函数**
   - 添加了 `incremental_save_path` 参数
   - 每完成一个样本立即保存结果
   - 即使出错也会记录到增量文件

2. **修改了主函数**
   - 生成增量保存文件路径
   - 传递增量保存路径给评估函数
   - 添加了详细的日志输出

### 文件路径规则

**增量保存文件路径**:
```
pred/{model_name}/{task}/{exp_name}/incremental_{config_str}.jsonl
```

**示例**:
- `pred/llama-3.1/scbench_repoqa/scbench_official/incremental_compress_0.1_important_0.5_recent_0.5_subvec_2_subbits_6.jsonl`
- `pred/llama-3.1/scbench_repoqa_and_kv/scbench_official/incremental_compress_0.1_important_0.5_recent_0.5_subvec_2_subbits_6.jsonl`

## 📈 当前进展提取

已从日志中提取了当前的16个完成样本，保存在：
- **文件**: `extracted_scbench_progress.jsonl`
- **格式**: 与SCBench标准输出格式兼容
- **内容**: 32个轮次的结果（每样本2轮）

### 性能统计
- **平均生成时间**: 1008.0秒 (16.8分钟/轮)
- **平均生成字符数**: 4410字符
- **时间范围**: 170.5s - 1943.8s
- **最新完成**: 样本#15 (2025-09-01 05:51:00)

## 🚀 如何使用新功能

### 方案1: 继续等待当前进程
- 当前进程会继续运行（无增量保存）
- 预计还需4.4天完成
- 可以使用 `extracted_scbench_progress.jsonl` 查看已完成的结果

### 方案2: 重新启动进程（推荐）
```bash
# 1. 终止当前进程
kill 3026285  # scbench_repoqa
kill 2758451  # scbench_repoqa_and_kv

# 2. 使用修改后的脚本重新启动
nohup python vq_pred_scbench_official.py --task scbench_repoqa ... > scbench_repoqa_incremental.log 2>&1 &
```

### 方案3: 小规模测试
```bash
# 减少样本数量进行测试
python vq_pred_scbench_official.py --task scbench_repoqa --num_eval_examples 5 ...
```

## 📝 增量文件格式

每行一个JSON对象，包含：
```json
{
  "id": 0,
  "turn_idx": 0,
  "prediction": "生成的回答文本",
  "ground_truth": "标准答案",
  "lang": "编程语言",
  "repo": "仓库名",
  "generation_time": 1150.86,
  "input_length": 45000,
  "output_length": 4156
}
```

## 🎯 优势

1. **实时可见**: 每完成一个样本立即保存
2. **容错性**: 进程意外中断不会丢失已完成的结果
3. **进度监控**: 可以随时查看当前进展
4. **向下兼容**: 仍然会在最后生成完整的结果文件

---
**生成时间**: 2025-09-01  
**修改文件**: `vq_pred_scbench_official.py`  
**提取工具**: `extract_current_progress.py`
