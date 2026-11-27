# 主表方法 × 数据量对照

> 当前状态：2026-11-26 21:11 Dubai，用户明确要求停止全部进程。STOP/PAUSE已保留，GPU worker、教师生成和调度器均已退出。除非用户明确要求恢复，不执行下方恢复操作。详见`USER_STOP.json`。

授权来自本会话：千级 GRPO、千级蒸馏，以及与已有 Distill-100 同题的 GRPO-100；继续只使用 GPU 2、4、5、6、7，持久后台并行、检测与自动回填。看板 owner 为 `codex-grpo@masdar`，三个任务分别为 `BL-GRPO-MAIN-20261126`、`BL-DISTILL-K-20261126`、`BL-GRPO-100-20261126`，仅本地更新，不提交或推送。

## 实验矩阵和数据

每种新增方法均为 Qwen3 1.7B/4B/14B × 医疗 t1/追踪 t2/数学 t4/代码 t5 × seeds 0/1/2，即 36 次训练与 36 次完整评测。新增共 108 次训练、108 次评测。12 个同协议 Base 已校验并复用，来源路径与文件指纹保存在评测 JSON 中。旧 Distill-100 无需重跑。

GRPO-100 使用原可见 100 题，逐题 ID、问题、答案均与 Distill-100 一致。K 数据医疗、追踪、代码各 2,000，数学 1,488；GRPO-K 与 Distill-K 共用同一组题，原 100 题是其子集。`data_validation.json` 核验逐题配对、去重、隐藏数据排除和正确/错误答案奖励。

- 医疗：MedCalc-Bench-Verified 公开 train 中规则类题，排除隐藏验证；计算器覆盖比原 100 题更广。
- 追踪：原 100 题 + 1,900 道独立生成题，生成器沿用项目实现、独立种子 261126。
- 数学：两个既有领域全部难度的公开训练题，排除隐藏验证；1,490 题中 2 条参考答案不能被现有评分类判正确，冻结为 1,488。
- 代码：原 100 道可见 CRUXEval 题 + 1,900 道有界语法独立生成的 Python 程序，执行求答案、过滤常量函数、按常量归一化 AST 去重。这不是声称复现不存在于本地的旧 LoRA 代码训练语料。

只使用隐藏题的 ID/哈希进行排除，不把隐藏问题、答案或输出传入训练或教师请求。跨大小比较同时包含覆盖范围变化，须如实解释。相同数据量不代表相同算力。

## 训练与评测

GRPO 使用 TRL 0.25.1 的 GRPOTrainer，bf16、LoRA r16/alpha32/dropout0，lr1e-5，cosine/3% warmup，beta .04、epsilon .2、group std、1 iteration、1 epoch、每题采样 8 个回答。最终答案正确给 1，否则 0；原有任务 grader，与主表评测相同。prompt 上限 4096，completion 上限 2048，temperature1/top_p1；截断 completion 不计策略损失。100 档每步 4 题/32 回答，共 25 步；K 档每步 8 题/64 回答，共 250 步（数学 186）。记录此 batch 差异，不把它当成严格相同优化预算实验。

vLLM 与训练 actor 同卡，固定 KV 16 GiB（14B 为 24 GiB），重要性修正开启、cap2。为避免 bf16 merge/unmerge 累积污染参考模型，同步采样模型时用临时合并张量，HF 冻结基座不做原位 merge。每步记录奖励、梯度、KL、长度和截断率，保留训练采样记录；每 25 步保存两个最近检查点。最终检查完整题目覆盖、步数、有限数值、冻结基座采样指纹和 adapter 指纹。真实 3 步试跑已验证 1.7B/14B 的非零适配器参与后续采样，峰值分别 40.84/90.42 GiB；pilot 不接触隐藏测试，不计正式结果。

千级蒸馏沿用已完成 Distill-100 的方法：Astra high 写可检查步骤，最终答案过滤；原 400 条示范直接复用，新增 7,088 条。仅训练集问题/参考答案传给教师，无工具或隐藏数据。全体学生与种子共享一套教师示范；某任务全量题目通过校验才发布 ready，不静默删掉失败题。LoRA r16/alpha32/dropout .05、lr2e-4、5 epochs、batch8、回复 token loss，固定最终检查点，所有样本每轮覆盖。K 档 1,250 步，数学 930 步；按整轮保存 adapter/优化器/随机状态，OOM 只减少 microbatch。

评测复用已验证的协议 2：vLLM bf16、关闭思考、贪心，任务 2048 新 token、GSM8K 512；固定 KV、max_num_seqs64、max_num_batched_tokens8192。医疗主指标教过组160、BBH750、MATH277、代码留出550；PPL、ARC-C300、HellaSwag300、算术200、MMLU1140、GSM8K500。副作用为 ARC-C/HellaSwag/MMLU/GSM8K 相对各任务 matched Base 的平均 pp，再平均任务和种子。未按隐藏验证/测试调参或选最好 seed。旧 Ours 有不同选择口径，不能直接严格排序。

## 后台进程和恢复

统一 `supervisor.py` → `watchdog.py` → worker。按显存预算在 GPU 2/4/5/6/7 动态补位：每卡最多 2 个训练、总共最多 3 个 worker；14B GRPO 的 128 GiB 预算使其独占。只管理本目录 worker 的 PID+创建时间及其子进程，不碰其他项目/显卡。优先保持约 3 个百条训练和 2 个千级训练；同时保留两个 14B 优先卡位，避免小模型长期占满所有卡而拖延大模型队列。其他卡也可运行 14B；某卡没有可容纳的高优先作业时补入能容纳的作业。教师数据 ready 后自动接入蒸馏。

资源策略 `resource_policy.py`（v2）：新 GRPO 1.7B/4B/14B 分别预留 64/66/128 GiB，进程内 PyTorch allocator 上限 58/60/122 GiB；1.7B 在普通预算无法补位时用 56 GiB 预留、50 GiB allocator、microbatch≤4。梯度累积保持 32/64 回答的有效 batch，G、步数、学习率、训练集和评测协议不变。SFT 沿用 48/64/100 GiB 预留、各减 4 GiB 的 allocator 上限。评测沿用固定 KV 和 44/72 GiB 预留；评测 vLLM 的子进程没有新增 allocator 硬限额。

准入同时计算本任务每个进程树的实际占用与预留的较大值、未归属本任务的占用，并保留至少 8 GiB 余量。刚启动还未分配显存的作业也按完整预算计入。原有未加 allocator 上限的训练继续保留旧的 75/90/128 GiB 预算，不能用新预算回算。新作业在环境变量和队列中记录资源档位，GRPO 训练配置另记 allocator 上限，崩溃后可按原档位接管；OOM 作业缩小 microbatch，独占重试。进程脱离终端后持续运行，重启仅接管，不重复启动存活 worker。

watchdog 每轮等待 5 秒后检查（加上 GPU/文件检查通常约 8–10 秒）；训练最多 4 次尝试、评测最多 8 次；45 分钟无进展或单次总计 24 小时才清理本任务进程树并重试。OOM 减少 microbatch、保持有效 batch；日志和断点保留。超限写 `ATTENTION.json`，其他任务继续。每组 36 个完整结果齐后单独标记对应看板 done；全部 108 个评测以及配套 Base 验证通过才写 `COMPLETE.json`，随后后台退出。

教师线程上限 4，实际起始并发 2，proxy 过载后自动减半至 1；每条最多 4 次实际生成尝试，总实际生成调用上限 2,400，token 保守计费上限 1,200 万。明确即时 proxy 503 拒绝按传输故障单独计数，不算生成成功或生成 token；旧无退避阶段的 600 次拒绝保留在 `transport_reconciliation.json`。总传输错误上限 700（含这 600 次）；遇到账号暂不可用全局等待 5 分钟。失败不会填成结果。`usage.json` 同时保留实际返回 token 和保守预算计数。若出现 `AUTH_ERROR.json`、预算上限或最终数据失败，需要查看原因；本地 GRPO 不受教师阻塞。

暂停补位与新教师请求：创建本目录 `PAUSE`；已有 GPU 作业继续跑，删除该文件后恢复。停止调度：创建 `STOP`；supervisor/watchdog 退出，已启动 GPU worker 保留，便于恢复时接管。恢复时移除 STOP，用项目 venv 在后台启动 `supervisor.py`。不要清空状态、全局 pkill 或重启旧 main_sched。

## 状态与证据

在项目根目录运行：

```bash
.venv-vllm/bin/python outputs/engineer/grpo_main/status.py
```

`gpu_utilization.jsonl` 连续记录利用率、功率、显存和作业分布，`watchdog_state.json` 还记录近 5 分钟采样均值。`resource_validation.json` 是预算、旧进程接管、有效 batch、OOM 隔离和 14B 优先策略的只读回归校验。`watchdog_state.json` 是实时队列，`supervisor.log` 是调度输出，`logs/` 保存每次尝试，`events.jsonl` 保存生命周期。`teacher_data/heartbeat.json` 与 `usage.json` 为教师进度和成本；`runs/*/train_done.json` 为训练证据，`evaluations/` 为完整指标，`RESULTS.md` / `summary.json` 为聚合结果。测试逐题输出仅在 `_secret/eval_grpo_main/`。

主表 `report/slides/GPT-Guided Model Editing.md` 自动增加三类方法，未齐时展示已完成种子数，三种子齐后写均值 ± 样本标准差。原文备份 `source_report.before_grpo_variants.md`；更新前后检查文件指纹避免覆盖并发修改。后续改代码应记录 `code_history/` 与 `CHANGELOG.md`，不可根据隐藏成绩改训练方案。
