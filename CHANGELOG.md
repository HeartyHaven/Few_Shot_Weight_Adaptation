# 框架修改日志

记录实验框架（`tdwa/`、排队脚本、报告）的每次改动，以及它影响哪些运行。学生权重的修改记录不在这里，在各运行目录的 `edits/`、`ledger.jsonl` 里。

## 2026-11-19（ws006343，main@ws006343）
- **路径可按机器配置**（`tdwa/evaluate.py`、`tdwa/teach.py`、`pilot/scripts/baseline2.py`）：`TDWA_HF`、`TDWA_VENV`、`TDWA_USER_ROOT` 环境变量可覆盖原来写死的 `/mnt/data2/...`；不设时与原来完全相同，ws006344 上的运行不受影响。
- **vLLM 评测**（`tdwa/evaluate_vllm.py`，新增）：与 `evaluate.py` 相同的提示、判分、隐藏测试和保留能力检查，生成用 vLLM（bf16），强制选择用限定选项字母的一步贪心，困惑度用提示 logprob。Gemma 1B 的 T1 隐藏测试与 HF 全精度结果相同（6.3%）。用它测的编辑/LoRA 学生只与同样用它测的基线比较（文件名带 `__vllm`）。
- **LoRA 对照**（`tdwa/baseline_lora.py`，新增）：B2（100 道真题）、B4（T1 全部 MedCalc 训练集去掉隐藏验证；T2 新生成 5000 题）。超参事先固定，不看验证集调。产出在 `baselines/`（不入库）。
- **任务 T3：DDXPlus**（`tdwa/data_ddx.py`，新增；`tdwa/grade.py` 加 `t3`，与 T2 同为选项题，5 个选项）：100 道真题 + 150 道隐藏验证（validate.csv）+ 600 道隐藏测试（test.csv，排除了摸底用过的题）。按疾病轮流抽样。教学沙盒（`kernel.py`/`teach.py`）尚未检查是否有只认 T2 的地方。
- **Gemma 3 4B 及以上能进沙盒**（`tdwa/kernel.py`）：这些检查点按 Gemma3ForConditionalGeneration 加载，解码层在 `model.model.language_model.layers`，参数名与权重文件键名（`language_model.model.*`）不同。反向记账钩子改为自动找解码层；原始权重的按需读取加了名称映射（4B 上 883 个参数全部对上，内存里不留副本）。之前 4B 沙盒启动即崩溃，ws006343 上三次 T1-G4 启动因此作废（`runs_void/`，零编辑）。
- **Codex 无界面教师**（`tdwa/codex_teach.py`，新增，ws006343）：Astra 教师可用 `codex exec` 运行，工具与 Fable 的 `mcp_teach` 相同（同一 MCP 服务，经本机 TCP 中继接入）。Codex 自带 shell 无法全部关闭，所以整个 codex 进程放在挂载命名空间里，项目目录、`~/.cache`、`~/.claude`、`~/.codex` 等都被空目录遮住；CODEX_HOME 是只含登录凭据的临时目录（结束后删除），shell 沙盒只读，网页搜索关闭，三个教学工具自动批准。运行名的教师字段为 `codex-<模型>`，与接口循环的 Astra 运行区分。框架不同，结果与接口循环的 Astra 运行不直接可比。
- **本机调度**（`logs/sched.py`、`logs/env_ws006343.sh`）：按显存把多个作业装进两张卡；所有缓存和大文件放项目目录下的 `cache/`（数据盘），不写系统盘。

## 2026-11-18
- **沙盒崩溃自动重启**（`tdwa/kernel.py`、`tdwa/teach.py`）：教师代码触发显卡错误或超时后，沙盒自动重启，从 `kstate.json` 和 `edits/` 恢复已提交的编辑、新增偏置位置和成本计数；未提交改动和教师的 Python 变量丢失，并在返回信息里告知教师。起因：T2 Fable s1 因显卡索引越界后进程无法恢复而作废。影响：15:00 之后启动的所有运行。
- **内存占用**（`tdwa/kernel.py`）：不再在 RAM 里保留原始权重和已提交权重两份完整副本。原始权重改为从模型文件按需读取（bf16 转 fp32，与加载结果逐张量核对一致）；已提交状态只保存与原始不同的张量。新沙盒启动时约 6 GB（之前 12–14 GB）。影响：15:09 之后启动的运行。15:02 启动的 T2 Astra 6 s2、T2 Fable s2 用的是只去掉一份副本的中间版本。
- **排队脚本**（`logs/queue.sh`、`logs/queue.txt`）：可用内存 ≥ 22 GB 且沙盒少于 4 个时逐个启动待跑的运行，选显存占用少的那张卡。启动记录在 `logs/queue.log`。
- **新教师** `gpt-6.2-astra`：与 `gpt-6-astra` 用同一套接口循环、提示和预算。
- **报告**（`report/`）：`export.py` 导出运行记录，`build.py` 合并中文说明，`tdwa_report.html` 为页面。

## 2026-11-17
- 首版框架：`tdwa/data.py`、`grade.py`、`kernel.py`、`teach.py`（接口循环教师）、`mcp_teach.py`（无界面 Claude Code 教师）、`evaluate.py`、`status.py`。
- 内存参考负载定为批量 1 × 1024 token（最初为 8 × 1024，过宽）。
- 沙盒增加模型 `blobs` 目录的挂载（否则权重文件的链接失效）。

## 已知未解决
- 无界面 Fable 教师没有教师 token 的硬上限（只有工具调用次数上限）。
- 教师可在沙盒 RAM 里缓存大量激活，单个沙盒可涨到 20 GB 以上，没有上限。
- 两类教师的运行方式不同（接口循环 / Claude Code）。
- 崩溃重启时，出事那一次调用里已消耗的学生 token 不计入成本。
- 2026-11-18 21:05 `tdwa/evaluate.py`：生成时遇到显存不足自动减半批量重试（共享显卡上评测曾因此失败一次）。贪心解码，输出不受批量影响。
- 2026-11-18 22:20 大模型基线：评测和摸底脚本加了显存硬上限（TDWA_MEM_GB）、可调批量（TDWA_BS）和显存不足自动减批；起因是 Qwen3-4B 的基线评测把显卡 1 占到 47.8 GB，挤压了教学运行。已有结果的步骤会跳过。

## 2026-11-19
- 排队脚本（`logs/queue.sh`）：每行可带第 6 列“所需显存（MB）”，选空闲显存最多的卡；所需显存 > 20 GB 的运行（4B 学生）单次调用时限设为 40 分钟；已存在的运行目录自动跳过。
- 开始 4B 学生（Qwen3-4B，MedCalc）。学生仍按全精度加载（推理峰值 16.4 GB，1.5 倍上限 24.5 GB，可放进单卡），与 1B 级结果设置一致；8B 及以上全精度放不下，届时再改半精度。
- `report/build_site.py`：被订阅额度中断的运行统一显示中文说明。
- 2026-11-19 04:45 排队脚本：大学生（所需显存 > 20 GB）还要求目标显卡上没有其他大学生沙盒（沙盒会长到上限）。起因：两个 Qwen3-4B 沙盒被放到同一张卡上（合计上限超过显卡容量），已停掉后启动的 Astra 6.2 s1（仅 5 步）并移入 runs_void/，重新排队。
- 2026-11-19 05:45 大模型基线队列：不再把基线任务放到已有 4B 教学沙盒的显卡上（避免显存超量）。
- 2026-11-19 10:30 教师的隐藏验证：每题最长回答从 2048 降到 1024 token（新运行的 config 里记为 `val_max_new_tokens`）。依据：已有 12,750 份验证回答中，超过 1024 token 的只占 0.5–3.3%，其中答对的只占全部的 0.2–0.3%；超长回答多是循环输出，却决定了整批生成的耗时。验证题数不变（MedCalc 120、物体追踪 150）；教师看不到的最终隐藏测试仍用 2048，所以所有运行的最终结果可比。10:30 前启动的运行仍用 2048。用户已同意（“可以，因为现在速度太慢了”）。
- 2026-11-19 13:45 `tdwa/evaluate.py`：教师新增但始终为零的偏置向量不再挂到模型上（零向量不改变输出）。起因：Qwen3-4B Astra 6.2 s1 在 gate_proj 输出上加了一个从未赋值、形状也不匹配的零偏置，评测时报形状错误。
- 2026-11-19 16:30 评测加速（用户要求“测试时用 vLLM 加速”）：本机装 vLLM 0.11.0（`.venv-vllm`，torch 2.8 cu128；0.34 需要更新的显卡驱动），transformers 4.57.1、datasets 4.0.0。全精度需要 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`、关闭 CUDA 图（`TDWA_VLLM_EAGER=1`）、`VLLM_USE_FLASHINFER_SAMPLER=0`（本机无 nvcc）。新脚本 `tdwa/merge_run.py`（把编辑合进全精度模型目录）和 `tdwa/eval_fast.py`（合并→vLLM 评测→清理）。一致性：未编辑 Gemma 1B 物体追踪 1200 题 98% 输出完全相同、99.3–99.7% 判分相同；Astra 6.2 s0 编辑后两套测试结果完全相同，第三套 83.3 对 84.0。速度约 15 倍。之后的评测默认用它，结果文件里 engine=vllm、dtype=float32。
- 2026-11-19 20:15 `tdwa/teach.py`：教师单次调用参数超过 20 万字符时不执行，并在历史里用占位符替代、回给教师错误说明。起因：T3 Qwen 1.7B Astra 6.2 s0 的教师在第 46 步塞入 169 万字符的代码，接口拒收含它的整段历史（单项上限 1 MiB），框架原样重试 6 次后中止（已提交的 1 个编辑保留，按正常结束评测）。只对之后启动的运行生效。

## 2026-11-20 01:40 main@ws006344
- tdwa/kernel.py commit_edit: change summary (norms, SVD rank) computed in float32; bf16 students failed on `svdvals` (no bf16 CUDA kernel). Stored old/new values unchanged. Qwen3-8B bf16 scripted self-test passed (commit, guard, quiz, rollback; peak 1.05x of inference). Started T1-Q8: Qwen3-8B MedCalc, Astra 6.2 s0, --dtype bfloat16.

## 2026-11-20 03:15 main@ws006344
- T1-Q8 Qwen3-8B (bf16) Astra 6.2 s0 finished with zero committed edits (50 turns, 1 hidden-validation look: 39.2 on the clean student); student = base, no separate test eval. Started s1 on GPU1.

## 2026-11-20 08:00 main@ws006344
- report/build_site.py, report/build_atlas.py: skip harness self-test runs (teacher "selftest").
- DECODE-SPEED measurements (Qwen3-8B bf16): sandbox quiz slowness is not the allocator (0 retries; expandable_segments no gain). Plain HF generate on real MedCalc prompts: 157 ms/step (16 longest, 2573 tok, padded), 98 ms/step unpadded, 35 ms/step short prompts. Fix needs continuous batching / early exit of finished sequences.
- Qwen3-14B baselines done; started T1-Q8 s2 (Astra 6.2) on GPU0.
- Observation (not changed): kernel.py caps GPU memory at 1.5 x 1.08 of the batch-1/1024-token inference peak; 17 runs peaked at 1.505-1.595x (largest: t1__Qwen3-4B__gpt-6-astra__s0 1.595). README states a strict 1.5x. Raised with the user.
- 10:45 T1-Q8 s1 hidden test (HF bf16): taught 48.8 -> 46.9, untaught 12.3 -> 13.2, retention unchanged. Added board task T1-Q14 (Qwen3-14B MedCalc, Astra 6.2) and started s0 on GPU1.

## 2026-11-20 13:00 main@ws006344
- tdwa/kernel.py: official validation now also refuses while teacher-made hooks or patches are attached to the student (module forward/pre/backward hooks other than the harness's own, instance or class `forward` overrides, patched `generate`, torch global module hooks). Before, it only refused uncommitted weight changes, so hooks (never saved) could change validation scores. Found in T1-Q8 s2: two looks on the same clean weights gave 39.2 and 34.2 (generation is otherwise deterministic: s0/s1/s2 clean looks all 39.17); the teacher's note says earlier probes were contaminated by stale hooks. A refused validation costs no look. Hidden-test results are unaffected (they use saved weights only). Running sandboxes pick this up only on restart. Tested with a scripted Qwen3-1.7B session (6 cases detected, harness bias hooks allowed).
- 16:25 T1-Q8 done: Qwen3-8B Astra 6.2 s2 hidden test taught 48.8 -> 46.2, untaught 12.3 -> 13.2 (s1: 46.9 / 13.2; s0 zero edits). Started same-protocol rule-in-prompt control for 8B on GPU0 (logs/b1_q8.sh, vLLM bf16: base and base+B1).
- 16:55 tdwa/teach.py Sandbox._restart: wait up to 300 s for the killed kernel to exit (was 30 s) and retry the launch up to 3 times with 90 s pauses. T1-Q14 s0 (Qwen3-14B) died because the restarted kernel ran out of GPU memory while the killed one still held 35 GB. That run (5.8 h, zero edits, 1 validation look at 35.8) moved to runs_void/...__restart_oom; s0 restarted with --exec-timeout 3600.
- 16:01 T1-Q14 s0 restart did not start: heavy disk I/O from another user's job made the 14B sandbox load exceed the 900 s startup limit (zero calls, zero edits); moved to runs_void/...__start_timeout. Paused at the user's request (no new work for now); a rerun needs a longer startup limit or normal disk I/O.
- 16:08 8B rule-in-prompt control done (logs/b1_q8.sh, vLLM bf16, same protocol): untaught-student test taught 48.1 / untaught 15.0 (free 29.0); with the B1 procedure text before each question taught 73.1 / untaught 44.6 (free 56.6); retention unchanged (ppl 8.13). As with 4B (26.9 -> 71.2), giving the rules in the prompt helps a lot, while weight editing by Astra 6.2 did not (s1 46.9, s2 46.2).

## 2026-11-20 23:30 engineer@ws006343
- New `tdwa/agy_teach.py`: teaching runs with the Antigravity CLI (`agy --print`, stream-json in and out; the prompt goes in on stdin as one user event) as the teacher. Run name `{task}__{student}__agy-{model}[_{tag}]__s{seed}`, with the same budget flags as codex_teach plus `--dtype` and `--nudges`. If agy ends a turn before calling the tdwa `finish`, the driver sends up to 3 "continue" messages. Isolation has three layers. (1) agy runs in a user, mount and pid namespace: /tmp is a fresh tmpfs holding only agy's private HOME, an empty cwd, the relay and a read-only bind of the agy binary, and /mnt/data/users/Lang_Gao (project, _secret, runs, TMPDIR, caches), the whole home directory, /var/tmp and /dev/shm are covered by empty tmpfs mounts. The network stays on. (2) The private HOME holds only a copy of the OAuth token file, a settings.json that allows `mcp(tdwa/*)` and denies `command/unsandboxed/read_file/write_file/read_url/execute_url/download_url(*)`, an mcp_config.json naming only tdwa (served outside the namespace over a TCP relay, as in codex_teach) and a hooks.json. The user's global agy config is untouched, and the staged token copy is deleted as soon as the namespace has read it. (3) A PreToolUse hook denies every tool except `mcp_tdwa_*`, which also blocks `search_web`: the permission rules alone did not. A leak test (gemini-3.8-flash-low with a stub tdwa server) showed `_secret` listing and reading, `find`, `ls /`, curl and web search all refused, and the tdwa run_python and finish tools working. A 4-call smoke run (Qwen3-1.7B t1, GPU1) completed with finish and a report and was moved to runs_void/.
