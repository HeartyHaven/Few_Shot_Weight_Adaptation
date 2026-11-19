# 当前任务与交接记录

负责人：main（Claude，绑定见 `.aris/workbench/launch.json`）。
授权来源：用户 2026-11-17 要求完善 `task.md` 的设计并交给 agent 执行；同日确认了 `DESIGN.md` 第 12 节的各项决定。
未创建或恢复任何 Goal。未购买资源。尚未授权任何教师模型的付费调用。

## 状态

| 项 | 状态 |
|---|---|
| 设计文档 `DESIGN.md` v1.1 | 完成，用户决定已写入 |
| WP0 环境与摸底 | 完成。engineer 的沙盒无 GPU 和网络，回报 blocked；main 接手完成，报告在 `pilot/REPORT.md` |
| WP1–WP6 | 未开始。等用户看过 WP0 结果后再派。engineer 面板需要先解决无 GPU/无网络的问题，否则只能做不需要 GPU 的代码工作 |

## 派给 engineer 的任务：WP0 环境与摸底

先读 `DESIGN.md` 全文（重点：第 5、6、7、11 节），再做下面的事。只写 `outputs/engineer/`。

目标：用实测数字回答"哪些（任务，学生）组合可以进入主实验"，并排除 Gemma 的编辑风险。

要做的事：

1. **环境。** 用 `uv` 在 `outputs/engineer/.venv` 建环境，pip 缓存指到 `/mnt/data2/users/lang.gao/pip_cache`。`HF_HOME=/mnt/data2/users/lang.gao/huggingface_cache`。
   磁盘很紧：`/mnt/data2` 约剩 14 GB，根分区约剩 6.5 GB。下载前先 `df -h`，不要保存完整权重副本。不要删除任何已有文件来腾空间；空间不够就停下来报告。
2. **学生加载。** `google/gemma-3-1b-it`（需下载，带门槛；`$HF_HOME/token` 存在，权限是否足够未确认）和 `Qwen/Qwen3-1.7B`（已缓存，聊天模板固定 `enable_thinking=False`）。确认贪心解码在固定种子下可逐位重现。
3. **Gemma 结构检查和冒烟测试**（DESIGN 第 5 节）。打印真实模块结构，确认 MLP 之后是否有归一化层。然后往某一层 MLP 输出写入一个已知方向的秩一改动，测下游残差流的变化，判断闭式编辑的幅度是否被归一化吃掉。Qwen 上做同样的测试作对照。
4. **教前基线**，每个学生都测：
   - T1：`nsk7153/MedCalc-Bench-Verified` 的规则类计算器。先统计该版本里规则类计算器的个数和每个的测试条数。
   - T2：BBH `tracking_shuffled_objects_{three,five,seven}_objects`（`lukaemon/bbh`）。
   - 每个任务测两种条件：B0（只有题目）和 B1（把你自己写的不超过 1 页的流程说明放进系统提示词）。两种作答模式都测：直接作答（16 个新 token）和自由作答（512 个新 token）。
   - 每个准确率给出 bootstrap 95% 置信区间。保存每道题的原始输出。
5. **显存。** 用 `torch.cuda.max_memory_allocated` 测两个学生在 batch 1、序列长 1024 下的推理峰值；再测"带前向钩子读全部层激活"和"对单层目标向量做一次反传"的峰值，给出相对推理的倍数。
6. **准入结论。** 按 DESIGN 第 6 节的两条门槛，对每个（任务，学生）给出通过/不通过和理由。T1 另外列出教前准确率落在 15%–70% 的计算器，供选教学组。

不要做的事：不要调用任何教师模型或付费 API；不要开始写 harness（那是 WP2）；不要写 `outputs/engineer/` 之外的地方；若读取测试题目，只用于本次基线测量，不要据此调提示词。

交付物：

- `outputs/engineer/wp0/REPORT.md`：结论在前，每个数字后面给出重现它的命令。区分实测和推断。
- `outputs/engineer/wp0/baseline.json`、`memory.json`、`gemma_arch.txt`、`smoke_edit.json`、每题原始输出。
- 所有脚本放 `outputs/engineer/wp0/scripts/`。

验收（由 main 执行）：抽查重跑至少两条命令得到相同数字；检查冒烟测试的结论有数据支持；检查没有写出自己的目录。

完成、失败或卡住时，在项目根目录运行其中一条：

```bash
superaris report /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation --id wp0-pilot --outcome completed --message 'WP0 报告在 outputs/engineer/wp0/REPORT.md'
superaris report /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation --id wp0-pilot --outcome blocked --message '<具体卡在哪>'
```

## 交接日志

- 2026-11-17：main 完成 `DESIGN.md` v1.1；登记 `wp0-pilot`；向 engineer 面板发出"读 HANDOFF.md 执行 WP0"的指令。已回读面板确认送达：engineer（Codex，GPT-6-Astra）状态变为 working 并开始读 HANDOFF.md。
- 2026-11-17：engineer 回报 `wp0-pilot` blocked（无 `/dev/nvidia*`，DNS 失败），证据在 `outputs/engineer/wp0/REPORT.md`。main 的会话有 GPU 和网络，在 `pilot/` 下完成 WP0，结果见 `pilot/REPORT.md`。
- 2026-11-17：用户质疑“提示词无效就算能力下限”的门槛；`DESIGN.md` 第 6 节已修订，所有组合都进主实验，分“提示可补”和“提示无效”两类报告。
- 2026-11-17：用户授权开始调用教师（Astra 走 CLIProxyAPI；Fable 用无头 `claude -p`，工具经 MCP）。main 写了执行环境 `tdwa/`（data、kernel、teach、mcp_teach、evaluate、status）。第一批 T2 + gemma-3-1b-it：s0 两次运行已完成并在隐藏测试集评测，结果在 `eval/`；s1、s2 四次运行已启动（更正，2026-11-18：四个同时跑把共享服务器内存压满，s2 两次当时被停掉；s1 两次跑完，其中 Fable s1 因沙盒进程被教师代码的显卡索引越界杀死而作废、零编辑；之后沙盒改为只留一份内存权重副本并支持崩溃后自动重启恢复已提交编辑；Astra s2、Fable s2 于 2026-11-18 15:05 重新启动，每张显卡只跑一个；Fable 还欠一次有效运行 s3）（日志在 `logs/`，状态用 `python -m tdwa.status`）。其余组合（Qwen、T1）未启动，等用户看过第一批结果。

## 2026-11-18 用户新增决定（main 记录）

- 优先级：MedCalc > 物体追踪补充 > 其他任务和数据集。
- 新增教师 Astra 6.2（`gpt-6.2-astra`，经本地代理）；旧的 `gpt-6-astra` 在汇报里统一称 “Astra 6”。
- 用户认为物体追踪过于机械，希望之后换更复杂但有明确学习方式的任务（数学、编程技巧、简单疾病诊断）。候选：DDXPlus、MBPP(+HumanEval)、MATH 数论；先做基线摸底再选。排在当前队列之后。
- 学生规模扩展（用户假设：学生越强越“可教”，教学收益更明显）：1B 级全部跑完后，依次尝试 ≤10B 和 ≤30B 的学生，系列以 Qwen、Gemma 为主。候选：Qwen3-4B / Qwen3-8B（已在本地缓存）/ Gemma-3-4b-it；Gemma-3-12b-it / Qwen3-14B / Gemma-3-27b-it。
  - 需要的框架改动：学生以 bf16 加载（10B 的 fp32 在 1.5 倍显存上限下放不进一张 49 GB 卡）；27B 级需要跨两张卡；内存参考负载和回归检查不变。
- 所有运行都要保留完整的修改日志：每个编辑的逐位置旧值/新值（`runs/*/edits/NNN/delta.safetensors` + `meta.json`）、`ledger.jsonl`、`transcript.jsonl`、教师的 `syllabus.md` / `curriculum_log.md`。框架自身的改动记在 `CHANGELOG.md`。
- 每 30 分钟向用户汇报一次进度（会话内定时任务）。

### 截至 2026-11-18 17:10 的结果
- T2 + Gemma：Fable s0 +12.3（部分成功）；Fable s1 作废（框架崩溃）；Fable s2 运行中（隐藏验证 73.3%）；Astra 6 s0/s1/s2 = +5.3 / +0.7 / −0.3（无效）。
- T1 + Gemma：Astra 6 s0 隐藏测试 6.3% → 9.5%（无效；编辑为把答案拟合进最后一层，学生不写过程直接猜数）；Fable s0、Astra 6.2 s0 运行中。

### 2026-11-18 17:45 用户裁定：局部优化手段可接受
- 用户原话：“少量数据不占资源的辅助训练手段（非直接训练端到端模型）可以接受”。
- main 的理解（解释，非用户原话）：在单个位置/少量向量上用优化器迭代（如 ROME 式目标向量优化、Fable 在 T1 用 Adam 优化 108 个记忆单元的输出向量）算合规，条件是数据量小、显存和 token 仍在 1.5 倍和预算内、不是对整个模型或适配器做端到端训练。此类编辑归入 `local_gradient` 类并照常计入反向 token 成本，结果正常计入，不单列。
- Fable 教师（无界面 Claude Code）占用用户的 Claude 订阅额度；17:15 触发 5 小时上限，T1 Gemma Fable s0 在 108 步被中断（已用 `tdwa.salvage` 收尾），s1 作废。剩余 Fable 运行暂存在 `logs/queue_fable_on_hold.txt`，等用户决定。
- 2026-11-18 20:10 用户决定：搁置的 Fable 运行全部要跑（T1 Gemma Fable s1、T2 Gemma Fable s3 已放回队列）。
- 2026-11-18 21:40 用户要求：尽量加速并行，避免过度等待；大模型基线提前做（不必等 1B 全部结束）。main 的调整：沙盒并发上限 4→6，RAM 放行线 22→16 GB，新增显存空余 ≥12 GB 的条件；大模型基线队列 `logs/bigpilot.sh`（Qwen3-4B、Gemma-3-4b、Qwen3-8B、Gemma-3-12b、Qwen3-14B；三个新任务 + T1/T2 隐藏测试基线，bf16）。
- 2026-11-18 22:35 旧账号的 Fable 周额度用尽（重置于 11-21 06:00），T1 Gemma Fable s1（40 步）和 T2 Gemma Fable s3（21 步）均零编辑中断，已移入 `runs_void/`，重新排在队首。用户已换登录账号，新账号 Fable 可用（测试通过）。
- 2026-11-18 22:35 用户要求：Fable 只用于执行任务（当教师）；观测、汇报、写说明等工作用其他模型（如 Opus）。main 会话已切到 Opus 5.5；子代理一律指定 opus。
- 2026-11-18 22:30 用户：以本地监控为主（http://127.0.0.1:8765/index.html，每 10 分钟自动刷新）；另有线上页 https://claude.ai/artifact/KZaE57gZBQAxbCtD5xZQFn（新账号所有，定时汇报时更新；旧账号的两个链接已不可用）。
- 2026-11-19 03:50 1B 级教学运行全部结束（物体追踪 Fable s3 被新账号 5 小时上限在第 141 步中断，已 salvage，3 个编辑）。按用户“1B 完成后试 10B/30B 以内、尽量并行”的要求，开始 Qwen3-4B 的 MedCalc 教学：Astra 6.2 s0 已启动，排队中 Fable s0、Astra 6.2 s1、Astra 6 s0。另跑 Qwen3-4B 全精度的 MedCalc 未编辑基线（教学运行是全精度）。
- 2026-11-19 10:30 用户同意缩短教师侧检验以提速，并重申：教师不能访问其他数据，只能用很有限的数据和计算资源来识别和编辑。main 的做法：只把隐藏验证的最长回答降到 1024 token；数据和算力规则（100 道真题 + 至多 300 道自编题、1000 万学生前向 token、显存 ≤ 推理峰值 1.5 倍、断网沙盒、看不到测试和其他目录）全部不变，也不放宽显存上限。
- 2026-11-19 11:05 应用户要求建立私有 GitHub 仓库 https://github.com/HeartyHaven/Few_Shot_Weight_Adaptation （代码、设计、公开数据划分、报告生成脚本；不含权重、运行记录、隐藏测试和评测结果，见 .gitignore）。项目目录现在是 git 仓库，分支 main。
- 2026-11-19 11:10 应用户要求在仓库建立任务看板（tasks/board.json → TASKS.md，工具 tdwa/board.py）和 README（模型、数据集型号与链接、多人并行协作规则）；AGENTS.md / CLAUDE.md 末尾加了看板指引。规则：开始任何任务前先看看板，只做 open 的任务，先 claim（git 推送防抢占），完成后 done。main 的定时汇报每 30 分钟同步看板。
