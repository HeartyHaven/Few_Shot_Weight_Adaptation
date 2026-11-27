# 主表蒸馏后台任务

授权：用户 2026-11-26 要求先蒸馏，GPU 2、4、5、6、7 尽量跑满，并构造后台 watchdog，直到主表这一组结果填满。看板任务 `BL-DISTILL-MAIN-20261126`，owner `codex-distill@masdar`。

范围：Qwen3 1.7B / 4B / 14B × t1/t2/t4/t5 × seeds 0/1/2。每任务同样 100 条可见真题，不添加合成题。GPT-6.2 Astra high 生成可检查的步骤示范，正确性过滤。教师只有题目、可见答案/参考解释，没有工具、验证或测试。一次生成的示范在所有学生和种子间共享。未执行 RL。

训练：bf16 LoRA，r=16，alpha=32，dropout=.05，lr=2e-4，5 epochs，effective batch=8；仅回复计算 loss。长样本不截断答案，每个 epoch 的最后不足批次也更新。按训练种子固定打乱；长度分桶减少 padding。OOM 只减 microbatch，不改 effective batch。每轮保存 adapter、优化器、scheduler、随机状态，故障从整轮断点恢复。无隐藏验证调参，无测试选优。

评测：vLLM bf16，关闭思考，贪心、2048 新 token；同任务、同精度重新测 Base。BBH 750；MedCalc 主指标教过组 160；MATH 277；CRUXEval 留出 550（100 dev + 150 val 从官方 800 中排除）。PPL、ARC-C、HellaSwag、MMLU、GSM8K；GSM8K 512 token。副作用行用四项分类/解题准确率相对配套 Base 的平均变化。原报告其他行的口径未改动，不能把 550 题和旧 800 题直接比较。

进程层级：`supervisor.py` 持久运行，watchdog 异常退出后重启；`watchdog.py` 每 15 秒检查、补位。进程采用独立 session，不依赖交互终端。重启按 PID+创建时间找回存活 worker，也扫描本任务专属 worker 路径以收回极小启动窗口中的进程。每卡最多 3 个 worker，按静态显存预留与当前空闲显存同时准入；不触碰其他项目进程。

训练最多 4 次尝试；评测最多 8 次，包含初始环境故障的尝试，日志全部保留。45 分钟无阶段/step 进展或总计 4 小时则清理本任务自己的进程树并重试。训练有 OOM 时降低 microbatch。教师调用有总量上限；401/403 停止这一轮。失败写入 `ATTENTION.json`，其余可做任务继续。只有 36 次蒸馏评测及配套 Base 全部通过完整性检查才写 `COMPLETE.json` 并把看板标 done。

评测协议 2 固定 KV cache（1.7B/4B 16 GiB，14B 24 GiB）、max_num_seqs=64、max_num_batched_tokens=8192，避免其他并行进程释放显存干扰 vLLM 的自动显存测量。首批协议 1 评测仅保留历史，统一重测协议 2 后才回填；不重新训练、不按测试结果修改训练。临时 IPC 路径在 `/tmp/fswa_distill_<uid>`，编译缓存仍在项目 cache；PATH 包含当前 venv 的 ninja。

查看状态（项目根目录）：

```bash
.venv-vllm/bin/python outputs/engineer/distill_main/status.py
```

主要文件：`watchdog_state.json` 实时状态；`watchdog.log` 调度日志；`events.jsonl` 生命周期；`logs/` 每次尝试日志；`data/usage.json` 教师 token 使用量；`runs/*/train_done.json` 训练配置、成本和指纹；`evaluations/` 验证后的指标；`RESULTS.md` / `summary.json` 汇总。逐题测试输出只在项目 `_secret/eval_distill_main/`。

自动回填用户指定 `report/slides/GPT-Guided Model Editing.md` 的主表，增加蒸馏与配套 Base 行，不改其他行。初始备份为 `source_report.before_distill.md`。不足 3 次显示完成数，齐后填均值 ± 标准差。每次更新做内容指纹检查，检测到同时修改就留到下一轮。

暂停新任务：在本目录创建 `PAUSE` 文件，现有任务完成后不会补位；删除后恢复。停止调度：创建 `STOP` 文件，supervisor/watchdog 退出，现有 worker 保留。恢复时移除 `STOP`，用当前 venv 后台启动 `supervisor.py`；它会接管存活 worker。全部完成后自动退出，不会循环重跑成功任务。

维护时不要清空状态、删除已训练 adapter 或用全局 pkill。失败修复后，先确认原 PID 已退出，再把对应 job 设回 retry，并保留尝试记录；已完成结果无需重跑。
