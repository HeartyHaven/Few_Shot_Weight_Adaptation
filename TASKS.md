# 任务看板

> 由 `python -m tdwa.board render` 从 `tasks/board.json` 生成，请不要手改本文件。最后更新：2026-11-20 17:50 +0400。
> 认领和完成任务的规则见 [README.md](README.md#多人并行协作规则)。

## 进行中（4）

| ID | 任务 | 学生 | 教师 | 数据 | 负责者 | 时间 | 结果 / 备注 | 产出位置 |
|---|---|---|---|---|---|---|---|---|
| T3-DDX | 教学运行：疾病诊断 DDXPlus | Qwen3 1.7B (Qwen/Qwen3-1.7B); Qwen3 4B (Qwen/Qwen3-4B) | Astra 6.2 (gpt-6.2-astra) | DDXPlus | main@ws006344 | 2026-11-19 18:05 +0400 | Astra 6.2 运行在第 46 步因框架漏洞（超长调用被接口拒收）中止，1 个编辑保留，验证 65.3→78.7%，隐藏测试评测中；漏洞已修。Astra 6 运行中 |  |
| DECODE-SPEED | 在 1.5 倍显存上限内实现更快的解码（如小页缓存的连续批处理），并验证输出与现有一致 |  |  |  | main@ws006344 | 2026-11-20 07:14 +0400 | 19:30 试 transformers 5.22 自带连续批处理 generate_batch（Qwen3-1.7B，32 道 MedCalc 题，512 token）：耗时 42 秒对 43 秒无提速，显存 10.9 对 7.0 GB 更高，且仅 1/32 输出与现有 generate 相同，不能直接替换。14B 一次 100 题小测已超过 60 分钟 |  |
| T1-Q14 | 教学运行：Qwen3 14B 学 MedCalc（检验更大模型是否更可教） | Qwen3 14B (Qwen/Qwen3-14B, https://huggingface.co/Qwen/Qwen3-14B，半精度) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-20 09:45 +0400 | 19:30 s0 运行 79 分钟、6 次调用；100 题全量小测超过 60 分钟单步上限，沙盒自动重启（重启修复生效） |  |
| T1-G12 | 教学运行：Gemma 3 12B 学 MedCalc（换模型家族检验 8B/14B 的结论） | Gemma 3 12B (google/gemma-3-12b-it, https://huggingface.co/google/gemma-3-12b-it，半精度) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-20 16:44 +0400 | 18:50 s0 已在 GPU1 启动（bf16），沙盒加载完成 |  |

## 未开始（可认领）（5）

| ID | 任务 | 学生 | 教师 | 数据 | 负责者 | 时间 | 结果 / 备注 | 产出位置 |
|---|---|---|---|---|---|---|---|---|
| T1-G4 | 教学运行：Gemma 3 4B 学 MedCalc（三位教师各一次） | Gemma 3 4B (google/gemma-3-4b-it) | Astra 6.2 (gpt-6.2-astra); Fable 5.2 (claude-fable-5-2); Astra 6 (gpt-6-astra) | MedCalc-Bench Verified (8 种规则评分) |  |  | 两个 Astra（Codex 框架）已跑：6.2 零保留编辑，6 教过的 16.3→18.1（无效）；Fable 待额度（11-22 15:00）；Gemma 4B 对照 B0/B1/B2/B5 已测，B4 训练中。交还 |  |
| RET-FULL | 更完整的通用能力检查：MMLU、GSM8K、IFEval（lm-eval） |  |  |  |  |  | MMLU(1140)+GSM8K(500) 已测全部对照组（结果在网页“对照组”标签）；未做：IFEval、教师编辑后的模型。用户要求暂停新工作，交还 |  |
| CHECK-HS | 核查 T1-Q17-A62-s0 编辑后 HellaSwag 54.7→64.3 的异常上升 | Qwen3 1.7B (Qwen/Qwen3-1.7B) |  |  |  |  |  |  |
| CAPTIONS-ZH | 为最早三次以外的运行补中文逐步说明和日志翻译（用 Opus，不用 Fable） |  |  |  |  |  |  |  |
| T1-Q4-F52-s1 | 教学运行：MedCalc（重复运行，Astra 额度用尽期间用 Fable） | Qwen3 4B (Qwen/Qwen3-4B) | Fable 5.2 (claude-fable-5-2) | MedCalc-Bench Verified (8 种规则评分) |  |  | 18:5x 新账号 Fable 周额度用尽（重置于 11-22 15:00），在第 53 步零编辑中断，已移入 runs_void/；额度恢复后重跑 |  |

## 已完成（51）

| ID | 任务 | 学生 | 教师 | 数据 | 负责者 | 时间 | 结果 / 备注 | 产出位置 |
|---|---|---|---|---|---|---|---|---|
| SETUP-DESIGN | 设计文档与实验规则 |  |  |  | main@ws006344 | 2026-11-17 | DESIGN.md v1.1 | DESIGN.md |
| SETUP-HARNESS | 教学沙盒、教师接入、评测、数据划分 |  |  |  | main@ws006344 | 2026-11-18 | tdwa/ 全部模块；崩溃自动重启、懒加载原始权重 | tdwa/ |
| PILOT-WP0 | WP0 先导：学生结构、编辑可行性、提示基线、显存比 | Gemma 3 1B (google/gemma-3-1b-it); Qwen3 1.7B (Qwen/Qwen3-1.7B) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-17 | 见 pilot/REPORT.md | pilot/REPORT.md |
| T2-G1-F52-s0 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Fable 5.2 (claude-fable-5-2) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 70.3%（+12.3） | runs/ 与 eval/（不在仓库） |
| T2-G1-F52-s2 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Fable 5.2 (claude-fable-5-2) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 72.0%（+14.0） | runs/ 与 eval/（不在仓库） |
| T2-G1-F52-s3 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Fable 5.2 (claude-fable-5-2) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 63.0%（+5.0，额度中断于第 141 步） | runs/ 与 eval/（不在仓库） |
| T2-G1-A6-s0 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6 (gpt-6-astra) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 63.3%（+5.3，不显著） | runs/ 与 eval/（不在仓库） |
| T2-G1-A6-s1 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6 (gpt-6-astra) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 58.7% | runs/ 与 eval/（不在仓库） |
| T2-G1-A6-s2 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6 (gpt-6-astra) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 57.7% | runs/ 与 eval/（不在仓库） |
| T2-G1-A62-s0 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6.2 (gpt-6.2-astra) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 97.3%（+39.3；BBH 96.7；更难题目 84.0） | runs/ 与 eval/（不在仓库） |
| T2-G1-A62-s1 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6.2 (gpt-6.2-astra) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 隐藏测试新出题目 58.0% → 85.3%（+27.3；BBH 89.3；更难题目 57.3） | runs/ 与 eval/（不在仓库） |
| T1-G1-A6-s0 | 教学运行：MedCalc | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6 (gpt-6-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 5.6% → 10.0%（不显著） | runs/ 与 eval/（不在仓库） |
| T1-G1-A6-s1 | 教学运行：MedCalc | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6 (gpt-6-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 5.6% → 15.0%（算术 44.5→34.0） | runs/ 与 eval/（不在仓库） |
| T1-G1-A62-s0 | 教学运行：MedCalc | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 5.6% → 21.3%（+15.7） | runs/ 与 eval/（不在仓库） |
| T1-G1-A62-s1 | 教学运行：MedCalc | Gemma 3 1B (google/gemma-3-1b-it) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 5.6% → 20.0%（+14.4） | runs/ 与 eval/（不在仓库） |
| T1-G1-F52-s0 | 教学运行：MedCalc | Gemma 3 1B (google/gemma-3-1b-it) | Fable 5.2 (claude-fable-5-2) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 5.6% → 14.4%（额度中断于第 108 步，1 个编辑） | runs/ 与 eval/（不在仓库） |
| T1-G1-F52-s1 | 教学运行：MedCalc | Gemma 3 1B (google/gemma-3-1b-it) | Fable 5.2 (claude-fable-5-2) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 零编辑结束（教师报告：找到可行流程但未能写入），等同未编辑 | runs/ 与 eval/（不在仓库） |
| T1-Q17-A62-s0 | 教学运行：MedCalc | Qwen3 1.7B (Qwen/Qwen3-1.7B) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 21.3% → 49.4%（+28.1） | runs/ 与 eval/（不在仓库） |
| T1-Q17-A62-s1 | 教学运行：MedCalc | Qwen3 1.7B (Qwen/Qwen3-1.7B) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 21.3% → 44.4%（+23.1） | runs/ 与 eval/（不在仓库） |
| T1-Q17-F52-s0 | 教学运行：MedCalc | Qwen3 1.7B (Qwen/Qwen3-1.7B) | Fable 5.2 (claude-fable-5-2) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 教过的 8 种评分 21.3% → 33.1%（+11.9） | runs/ 与 eval/（不在仓库） |
| T1-Q17-A6-s0 | 教学运行：MedCalc | Qwen3 1.7B (Qwen/Qwen3-1.7B) | Astra 6 (gpt-6-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-18/19 | 所有编辑都被教师撤回，等同未编辑 | runs/ 与 eval/（不在仓库） |
| T1-Q4-A62-s0 | 教学运行：MedCalc | Qwen3 4B (Qwen/Qwen3-4B) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-19 13:15 +0400 | 教过的 8 种评分 26.9% → 55.6%（+28.7，区间 48.1–63.1）；没教过的 16.8% → 10.5%（下降）；困惑度 10.33→10.93 | eval/t1__Qwen3-4B__gpt-6.2-astra__s0.json（不在仓库） |
| T1-Q4-F52-s0 | 教学运行：MedCalc | Qwen3 4B (Qwen/Qwen3-4B) | Fable 5.2 (claude-fable-5-2) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-19 15:44 +0400 | 教过的 8 种评分 26.9% → 52.5%（+25.6，区间 44.4–59.4）；没教过的 16.8% → 13.6%；困惑度 10.33→10.34，算术 78→71 | eval/t1__Qwen3-4B__claude-fable-5-2__s0.json（不在仓库） |
| T1-Q4-A62-s1 | 教学运行：MedCalc | Qwen3 4B (Qwen/Qwen3-4B) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-19 17:15 +0400 | 教过的 8 种评分 26.9% → 56.3%（+29.4，区间 48.8–64.4）；没教过的 16.8% → 10.0%（下降）；困惑度 10.33→10.60；vLLM 全精度评测 | eval/t1__Qwen3-4B__gpt-6.2-astra__s1.json（不在仓库） |
| T1-Q4-A6-s0 | 教学运行：MedCalc | Qwen3 4B (Qwen/Qwen3-4B) | Astra 6 (gpt-6-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-20 01:19 +0400 | Astra 6 重跑：已教 32.5%（基线 26.9），未教 10.0（基线 16.8）；通用能力基本不变 | eval/t1__Qwen3-4B__gpt-6-astra__s0.json |
| BASE-G1 | 未编辑基线（追踪、MedCalc、三个新任务） | Gemma 3 1B (google/gemma-3-1b-it) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006344 | 2026-11-18 | 追踪新出题目 58.0；MedCalc 教过 5.6；诊断 37.3 / 编程 58.7 / 数学 51.3 | eval/、pilot/out2/（不在仓库） |
| BASE-Q17 | 未编辑基线（MedCalc、三个新任务） | Qwen3 1.7B (Qwen/Qwen3-1.7B) |  | MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006344 | 2026-11-18 | MedCalc 教过 21.3；诊断 52.0 / 数学 75.3（编程待重测） | eval/、pilot/out2/（不在仓库） |
| BASE-Q4 | 未编辑基线（全部，含全精度 MedCalc） | Qwen3 4B (Qwen/Qwen3-4B) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006344 | 2026-11-19 | MedCalc 教过 26.9（全精度）；追踪 BBH 85.2；诊断 56.0 / 编程 70.0 / 数学 77.3 | eval/、pilot/out2/（不在仓库） |
| BASE-G4 | 未编辑基线（半精度） | Gemma 3 4B (google/gemma-3-4b-it) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006344 | 2026-11-19 | MedCalc 教过 17.5；追踪 BBH 95.5；诊断 60.0 / 编程 72.7 / 数学 77.3 | eval/、pilot/out2/（不在仓库） |
| BASE-Q8 | 未编辑基线（半精度，三个新任务 + 追踪 + MedCalc） | Qwen3 8B (Qwen/Qwen3-8B) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006344 | 2026-11-19 18:19 +0400 | 半精度：诊断 68.0 / 编程 80.0 / 数学 73.3；MedCalc 教过 48.8、没教过 12.3；追踪 新出题目 96.3、BBH 94.5、更难题目 91.3（追踪已无教学空间） | eval/base__t*__Qwen3-8B__bf16.json（不在仓库） |
| BASE-G12 | 未编辑基线（半精度，三个新任务 + 追踪 + MedCalc） | Gemma 3 12B (google/gemma-3-12b-it) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006344 | 2026-11-20 02:16 +0400 | Gemma-12B 未编辑（bf16）：MedCalc 已教46.2/未教16.4；追踪 BBH 99.5、新题 99.3、难题 98.0（无可教空间）；ddx 72.7, code 80.7, math 84.0；ARC 90.7, HS 85.0 | eval/base__t2__gemma-3-12b-it__bf16.json |
| BASE-Q14 | 未编辑基线（半精度，三个新任务 + 追踪 + MedCalc） | Qwen3 14B (Qwen/Qwen3-14B) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006344 | 2026-11-20 07:14 +0400 | Qwen3-14B 未编辑（bf16）：MedCalc 已教45.0/未教18.6（不高于 8B 的 48.8）；追踪 BBH 98.7、新题 99.7、难题 87.3；ddx 68.7, code 82.7, math 84.7；ARC 92.3, HS 89.0 | eval/base__t1__Qwen3-14B__bf16.json |
| BASE-G27 | 未编辑基线：Gemma 3 27B（需两张卡） | Gemma 3 27B (google/gemma-3-27b-it) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分); 三个候选新任务：DDXPlus / MBPP / MATH 数论 | main@ws006343 | 2026-11-19 14:58 +0400 | vLLM bf16 两卡：MedCalc 教过的 53.8%/全部 30.8%；追踪 BBH 99.3%；DDXPlus 71.5%；摸底 ddx 70.0 / 编程 84.7 / 数学 84.7 | report/site/baselines.json |
| BASE-REGRADE-CODE | 用修好的代码提取逻辑重测 Gemma 1B 与 Qwen 1.7B 的编程基线 | Gemma 3 1B (google/gemma-3-1b-it); Qwen3 1.7B (Qwen/Qwen3-1.7B) |  | MBPP sanitized | main@ws006344 | 2026-11-19 17:46 +0400 | Gemma 1B：诊断 37.3 / 编程 58.7 / 数学 51.3；Qwen 1.7B：诊断 52.0 / 编程 60.0（原 28.0，代码提取问题已修）/ 数学 75.3 | pilot/out2/ |
| KERNEL-BF16 | 让沙盒支持半精度学生（8B 以上必需），并检验半精度对微小编辑的影响 |  |  |  | main@ws006344 | 2026-11-20 01:47 +0400 | 沙盒支持 --dtype bfloat16：Qwen3-8B 自检通过（提交/守门/测验/回滚正常，显存峰值 1.05 倍）；修复 bf16 下 SVD 摘要报错 | tdwa/kernel.py |
| T1-Q8 | 教学运行：Qwen3 8B 学 MedCalc | Qwen3 8B (Qwen/Qwen3-8B) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-20 15:15 +0400 | Qwen3-8B（bf16）Astra 6.2 三个种子均无提升：s0 零编辑；s1 已教 48.8→46.9、未教 12.3→13.2；s2 已教 48.8→46.2、未教 12.3→13.2；通用能力不变 | eval/t1__Qwen3-8B__gpt-6.2-astra__s2.json |
| DDX-SPLITS | 为 DDXPlus 建立 100 道真题 / 隐藏验证 / 隐藏测试划分 |  |  | DDXPlus (aai530-group6/ddxplus) | main@ws006343 | 2026-11-19 11:19 +0400 | t3: 100 道真题 / 150 隐藏验证 / 600 隐藏测试，按疾病轮流抽样；tdwa/data_ddx.py；grade.py 加 t3 | data/t3/dev.jsonl |
| BL-LORA-FEW | 对照 B2：只用 100 道真题做 LoRA 微调 | Gemma 3 1B (google/gemma-3-1b-it); Qwen3 1.7B (Qwen/Qwen3-1.7B) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分) | main@ws006343 | 2026-11-19 12:59 +0400 | 3 种子：T2 G1 56.9→92.9、Q17 58.7→98.5；T1 G1 6.3→16.2、Q17 13.4→22.9；ARC-C 降 14–35 点 | pilot/BL_LORA.md |
| BL-DISTILL | 对照 B3：用教师写的解题过程做蒸馏 | Gemma 3 1B (google/gemma-3-1b-it); Qwen3 1.7B (Qwen/Qwen3-1.7B) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分) | main@ws006343 | 2026-11-20 01:28 +0400 | B3（Astra 6.2 经隔离 Codex 写解题过程，100 真题+≤300 自编，LoRA）：追踪 G1 56.9→18.3、Q17 58.7→29.2（教师写倒推捷径，小模型学不会）；MedCalc 教过的 G1 4.4→18.1、Q17 21.9→25.0、Q4 26.9→32.5；通用能力均下降 | report/site/baselines.json |
| BL-LORA-FULL | 对照 B4：用全部训练数据做 LoRA（上限参考） | Gemma 3 1B (google/gemma-3-1b-it); Qwen3 1.7B (Qwen/Qwen3-1.7B) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成); MedCalc-Bench Verified (8 种规则评分) | main@ws006343 | 2026-11-19 12:59 +0400 | T2 两学生 99.9；T1 G1 6.3→26.3、Q17 13.4→36.0；ARC-C 降 17–42 点，全部过不了副作用标准 | pilot/BL_LORA.md |
| BL-PROMPT-BAKE | 对照 B5：提示烘焙（把解题步骤蒸进权重） | Gemma 3 1B (google/gemma-3-1b-it) |  | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006343 | 2026-11-19 14:04 +0400 | B5 物体追踪：Gemma 1B 57.7→26.0（B1 提示本身 23.3，烘焙学到了有害的提示行为）；Qwen 1.7B 57.7→64.7（B1 61.0）；保留能力 ARC 降 0–10 点 | report/site/baselines.json |
| ATLAS-DATA | 编辑图谱数据：每个编辑落在哪些层/头/神经元（report/build_atlas.py → report/site/atlas.json） |  |  |  | main@ws006344 | 2026-11-19 11:40 | 20 次运行、112 个编辑已解析 | report/build_atlas.py |
| ATLAS-ROLES | 从教学日志提取每次运行识别过的部件、教师认为它的作用、证据和处理方式（Opus 子任务） |  |  |  | main@ws006344 | 2026-11-19 11:42 +0400 | 20 次运行、297 个部件条目（Opus 从教学日志提取） | report/site/atlas_roles.json |
| ATLAS-PAGE | 编辑图谱子页面：逐步可视化每个运行定位和修改的参数，跨教师/运行对照找共性（GPT Astra / engineer 面板编写） |  |  |  | engineer@ws006344 | 2026-11-19 11:50 +0400 | 已接入网站：本地 http://127.0.0.1:8765/atlas.html，线上主报告顶部“编辑图谱”；29 项逻辑检查通过，未做浏览器实测 | report/atlas.html |
| BL-LORA-Q4 | 对照 B2/B4 扩到 Qwen3 4B 的 MedCalc（与 Astra 6.2 的 4B 教学结果对比） | Qwen3 4B (Qwen/Qwen3-4B) |  | MedCalc-Bench Verified (8 种规则评分) | main@ws006343 | 2026-11-19 23:06 +0400 | Qwen3 4B MedCalc 教过的8种：未编辑 26.9；B1 规则提示 71.9；B2 55.2(3种子)；B4 见网页；B5 提示烘焙 71.2 且通用能力无损 | report/site/baselines.json |
| EVAL-VLLM-WS006344 | 在 ws006344 上装 vLLM，并让教学运行（含编辑后权重）的隐藏测试改用 vLLM；先验证与现有 float32 结果的一致性 |  |  |  | main@ws006344 | 2026-11-19 17:15 +0400 | vLLM 0.11.0 全精度评测可用：与原评测 98% 输出相同、判分 99%+ 相同；Qwen3-4B 一次完整评测约 20 分钟（原来 3 小时以上） | tdwa/eval_fast.py |
| T1-Q4-A62-s2 | 教学运行：MedCalc（第 3 次重复） | Qwen3 4B (Qwen/Qwen3-4B) | Astra 6.2 (gpt-6.2-astra) | MedCalc-Bench Verified (8 种规则评分) | main@ws006344 | 2026-11-20 00:14 +0400 | 教过的 8 种评分 26.9% → 61.9%（+35.0，区间 54.4–69.4，三次中最高）；没教过的 16.8% → 11.4%；困惑度 10.33→10.80，ARC/HellaSwag/算术不变 | eval/t1__Qwen3-4B__gpt-6.2-astra__s2.json（不在仓库） |
| T3-Q17-A6-s0 | 教学运行：疾病诊断 DDXPlus | Qwen3 1.7B (Qwen/Qwen3-1.7B) | Astra 6 (gpt-6-astra) | DDXPlus (aai530-group6/ddxplus) | main@ws006344 | 2026-11-19 23:45 +0400 | 隐藏测试 66.7% → 67.3%（无效，保留 1 个编辑）；通用能力不变（算术 49→55.5） | eval/t3__Qwen3-1.7B__gpt-6-astra__s0.json（不在仓库） |
| SCAFFOLD-CODEX | 框架对照：同一配置下 Astra 6.2 走 Codex 无界面 vs 接口循环（T1 Qwen 1.7B、T2 Gemma 1B 各一次） | Qwen3 1.7B; Gemma 3 1B | Astra 6.2 (gpt-6.2-astra) via codex exec | MedCalc; 物体追踪 | main@ws006343 | 2026-11-19 23:28 +0400 | Codex 框架明显弱于接口循环：T2 Gemma1B 测试 BBH 62.7/新题 63.0/难题 33.3（接口循环 96.7/97.3/84.0，未编辑 57.2/58.0/26.7）；T1 Qwen1.7B 零保留编辑（接口循环 49.4）。两次都在 47–66 步主动结束 | eval/t2__gemma-3-1b-it__codex-gpt-6.2-astra__s0.json |
| T3-Q17-A62-s0 | 教学运行：疾病诊断 DDXPlus | Qwen3 1.7B (Qwen/Qwen3-1.7B) | Astra 6.2 (gpt-6.2-astra) | DDXPlus (aai530-group6/ddxplus) | main@ws006344 | 2026-11-19 20:45 +0400 | 隐藏测试 66.7% → 81.0%（+14.3，区间 77.7–84.0）；直接选答案 60.8% → 81.2%；但三位数算术 49% → 12%（副作用检查只用两位数算术，没拦住），ARC 73.7→69.0；第 46 步因框架漏洞中止 | eval/t3__Qwen3-1.7B__gpt-6.2-astra__s0.json（不在仓库） |
| T3-Q17-A62-s1 | 教学运行：疾病诊断 DDXPlus（重复运行） | Qwen3 1.7B (Qwen/Qwen3-1.7B) | Astra 6.2 (gpt-6.2-astra) | DDXPlus (aai530-group6/ddxplus) | main@ws006344 | 2026-11-19 22:45 +0400 | 零保留编辑（唯一的编辑使验证 65.3→58.0%，已撤回），教师在 40 步主动结束；学生等同未编辑（隐藏测试 66.7%）。与第 1 次（81.0%）差别很大，说明这个任务上的成功依赖教师在那次运行里找到的具体改法 |  |

## 作废（1）

| ID | 任务 | 学生 | 教师 | 数据 | 负责者 | 时间 | 结果 / 备注 | 产出位置 |
|---|---|---|---|---|---|---|---|---|
| T2-G1-F52-s1 | 教学运行：物体追踪 | Gemma 3 1B (google/gemma-3-1b-it) | Fable 5.2 (claude-fable-5-2) | 物体追踪 (BBH tracking_shuffled_objects + 自生成) | main@ws006344 | 2026-11-18/19 | 作废：沙盒被显卡错误杀死、当时无恢复机制，零编辑 | runs/ 与 eval/（不在仓库） |

