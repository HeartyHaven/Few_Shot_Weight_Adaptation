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

## 派给 engineer 的任务：编辑图谱子页面（句柄 atlas-page，看板 ATLAS-PAGE）

用户要求：把每个模型、每次运行“每一步定位并修改了哪些参数、教师认为它们是干什么的、怎么改的”可视化；并能在运行之间、教师之间对照，看修改的位置或教师的理解有没有共性（共性部位很可能就是小模型的症结）。做成报告网站的一个独立子页面。用户指定编程交给 GPT Astra（engineer 面板）。

**只写 `outputs/engineer/atlas/`。** 交付 `outputs/engineer/atlas/atlas.html`（单个自包含 HTML 文件，内联 CSS/JS；如需库只能从 cdnjs.cloudflare.com 加载并写死版本号）。main 会把它复制成 `report/atlas.html` 并在主报告导航里加链接。不需要 GPU，不需要网络，不要调用任何模型 API。

**数据（只读）**，页面用相对路径 `fetch('site/atlas.json')` 和 `fetch('site/atlas_roles.json')` 读取：
- `report/site/atlas.json`（已存在，由 `report/build_atlas.py` 生成）：`{"runs":[{run, task("t1"=MedCalc 临床评分 | "t2"=物体追踪), student, teacher("claude-fable-5-2"=Fable 5.2 | "gpt-6-astra"=Astra 6 | "gpt-6.2-astra"=Astra 6.2), seed, arch{layers, heads, kv_heads, head_dim, hidden, mlp, layer_types}, n_steps, exams[[after_edit, 验证正确率%]], edits[{id, step(第几次工具调用), minute, accepted(是否通过副作用检查), rolled_back, kept_final(最终是否保留), op_class, n_values, val_after(该编辑后的隐藏验证%或null), hypothesis, method, prediction, data_used(英文原文), regression{ppl,canary...}, targets[{param, layer, module(q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj/bias:..), unit_kind("head"|"neuron"|"tensor"|"vector"), n_units, n_units_changed, units[[单元编号, 该单元改动占本矩阵改动的比例(范数比)]]（最多 24 个，按大小排序）, top_share, rel_norm(改动幅度/原矩阵), n_values, rank, top_sv}]}]}]}`。k/v_proj 的 head 编号是 KV 头编号；attention 的 q/o 头编号是查询头编号（Gemma 1B 有 4 个查询头、1 个 KV 头；Qwen3-1.7B 16/8；Qwen3-4B 32/8）。
- `report/site/atlas_roles.json`（main 稍后生成，结构如下；在它出现之前请自己造 2 条假数据测试，交付时去掉）：`{"<run>": {"summary_zh": "...", "items": [{label("L17H0"等), layer, kind("head"|"mlp"|"neurons"|"layer"|"embedding"|"norm"|"bias"|"other"), module, index, role_zh(教师认为它做什么), evidence_zh, action("edited"|"probed"|"ablated"|"rejected"), edit_ids[], step, source}]}}`。

**页面要有的内容**（中文界面，通俗直白）：
1. 顶部筛选：任务（物体追踪 / 医疗评分）、学生、开关“只看最终保留的编辑 / 包含撤回和被拒的”。
2. **部位总图**：选定任务+学生后，画“层 × 部件”网格：每层一行；列是每个注意力头（q/k/v/o 合并到头）和一个 MLP 格（显示被改神经元数）。格子颜色 = 改过它的运行数，标记出自几位不同教师。再叠加“只被定位/探测、没被改”的部件（来自 roles 的 probed/ablated/rejected），用不同样式。点击格子：列出所有涉及它的运行、编辑、步号、教师给出的作用解释（role_zh）和改法（module、改动幅度、改了多少单元）。
3. **逐步时间线**：选一个运行，按步号排列：每个“识别/探测”条目和每个编辑；编辑卡片显示步号、分钟、改了哪些层/模块/头/神经元（用小条形图显示单元占比）、操作类别中文名（analytic 解析计算 / forward_observation 前向观察 / closed_form 公式直接求解 / zeroth_order_search 小范围试探 / local_gradient 局部梯度 / new_bias_parameter 新增偏置）、编辑后验证分数、是否保留/撤回/被拒；英文 hypothesis/method 折叠展开。顶部显示该运行的 summary_zh 和验证分数曲线。
4. **共性分析**：对选定任务+学生，按“部件”汇总：被多少次运行、多少位教师改过/定位过，排序成表；给出自动生成的一句结论（例如“第 17 层 0 号头被 3 位教师中的 3 位、8 次运行中的 7 次改过”）。MLP 神经元：统计在不同运行里重复出现的神经元编号（同一层同一模块），列出重复次数最多的。再加一张“相对深度”分布图：不同学生的改动按 层号/总层数 分箱，便于跨学生比较。
5. 页面标题用 `<title>编辑图谱</title>`；顶部放一个“返回主报告”的链接 `index.html`（线上版 main 会改成正确地址）。

**样式**：沿用主报告的配色和字体（参考 `report/tdwa_report.html` 开头的 CSS 变量：--bg、--panel、--ink、--muted、--line、--soft、--base、--astra、--fable、--bad 等，浅色/深色两套），手机宽度可用，表格和图在自己的容器里横向滚动。这是一个独立 HTML 页面：要写自己的 `<!doctype html>`、charset、viewport。

**验收**：在本机用 `python3 -m http.server` 打开（数据路径用相对路径），四个部分在“物体追踪 + Gemma 1B”和“医疗评分 + Qwen3 1.7B”两种组合下都正常显示，控制台无报错。完成后：
`superaris report /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation --id atlas-page --outcome completed --message '页面在 outputs/engineer/atlas/atlas.html'`；卡住则 `--outcome blocked` 并写原因。
- 2026-11-19 13:00（main@ws006343，第二台机器，2×49 GB）：用户要求“灵活调用模型、尽量让 GPU 利用率到 100%、用 vLLM 加速生成”，大文件只放数据盘的项目目录（`cache/`），不写系统盘。数据盘用户配额 1 TB 已接近用满，LoRA 只存适配器。完成 DDX-SPLITS、BL-LORA-FEW、BL-LORA-FULL（结果见 `pilot/BL_LORA.md`：LoRA 任务分数远高于教师编辑，但通用能力全部大幅下降）。Gemma 27B 基线（BASE-G27）正在本机两卡上跑。本机调度：`logs/sched.py` + `logs/jobs_ws006343.jsonl`，环境 `logs/env_ws006343.sh`。
- 2026-11-19 17:57 用户提供 callback，代理的 Codex 重新登录成功（新凭证在 ~/.cli-proxy-api/），Astra 6 和 Astra 6.2 恢复可用。登录时因 1455 端口被用户自己的 Codex 占用，用 -oauth-callback-port 1458 起本地监听、手动转交 callback。Qwen3-4B 上 Astra 6 重新排队。
- 2026-11-19 19:40 新账号的 Fable 周额度也已用尽（重置于 11-22 15:00）：T1 Qwen3-4B Fable s1（53 步）和 T3 Qwen 1.7B Fable s0（30 步）零编辑中断，移入 runs_void/。显卡 0 因此空了约 20 分钟，用户指出后改排 Astra 6.2（T1 Qwen3-4B s2）和 Astra 6（T3 Qwen 1.7B）。在 Fable 恢复前只用 Astra 教学。
- 2026-11-20 16:00（main@ws006343）：本机工作告一段落，用户要求暂不开新工作。
  - 完成：全部对照组（B1/B2/B3/B3t/B4/B5，Gemma 1B、Qwen 1.7B、Qwen 4B、Gemma 4B，T1/T2），MMLU/GSM8K 保留检查（RET-FULL 前半），框架对照（Astra 走 Codex 明显弱于接口循环），Gemma 27B 基线，DDXPlus 划分。结果在网页“对照组”标签和 `pilot/BL_LORA.md`。
  - 关键结论：Qwen3 4B MedCalc 规则类上，提示烘焙 B5 71.6±0.5%（通用能力无损）高于教师编辑 55.6%；物体追踪 B3 失败是因为教师写倒推捷径，要求逐步写表后（B3t）与 B2 持平。
  - 任务与模型方案：`report/TASK_MODEL_PLAN.md`，已写入用户的项目文档（Experiments → Setup 后三个小节）。主表建议：物体追踪 + MedCalc（规则类 + 公式类）+ MATH Level 5 + CRUXEval-O；学生 Qwen3 1.7B/4B/14B，全部可从 ModelScope 免申请下载。
  - 未完成：T1-G4 的 Fable 运行（Fable 额度 11-22 15:00 恢复）；Codex 框架是否加“提前结束自动续轮”待用户决定；RET-FULL 的 IFEval 和教师编辑模型的 MMLU/GSM8K。
  - 本机不提交 `.aris/workbench/context/*/LATEST_PREPARED.md`（远程版本指向 ws006344 路径）。
