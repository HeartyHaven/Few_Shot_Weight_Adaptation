# 千级数据 Distill / GRPO 对照：执行流程

这个分支要做的事只有一件：给主表的每个模型（Qwen3 1.7B / 4B / 14B）在每个 bench（通用 t2、医疗 t1、数学 t4、代码 t5）上各补一组**千级训练数据的蒸馏**和一组**千级训练数据的 GRPO**，一共 12 × 2 = 24 组。结果表在 [`outputs/engineer/grpo_main/RESULTS_K.md`](outputs/engineer/grpo_main/RESULTS_K.md)，现在是空的，跑完一组自动填一格。

分支内容来自 masdar（8×H200）项目副本在 2026-11-27 的状态：蒸馏和 GRPO 的全部脚本、训练题、已生成的教师示范、已有的评测记录，加上本文件、一个顺序执行脚本和空结果表。

## 1. 要跑的 24 组

| | 通用 t2 | 医疗 t1 | 数学 t4 | 代码 t5 |
|---|---|---|---|---|
| 训练题数（两种方法共用同一批题） | 2,000 | 2,000 | 1,488 | 2,000 |
| Qwen3 1.7B | Distill + GRPO | Distill + GRPO | Distill + GRPO | Distill + GRPO |
| Qwen3 4B | Distill + GRPO | Distill + GRPO | Distill + GRPO | Distill + GRPO |
| Qwen3 14B | Distill + GRPO | Distill + GRPO | Distill + GRPO | Distill + GRPO |

默认每组跑 1 个种子（seed 0）。要 3 个种子时给执行脚本加 `--seeds 0 1 2`，结果表会自动改写成均值 ± 标准差。

两种方法都用最常规的做法，都从原始 Base 出发训练一个 LoRA（r=16，alpha=32，注意力和 MLP 的 7 个投影层），都只取固定预算跑完后的最后一个检查点，不看隐藏验证或测试来调参、选模型。

**Distill（教师示范蒸馏）**：教师 GPT-6.2 Astra（high）给每道训练题写一份带步骤的解答，最终答案判对才保留；学生在这些解答上做监督微调，只对回复部分算 loss。

| 项目 | 取值 |
|---|---|
| 轮数 / 每步题数 | 5 轮 / 8 题（每轮覆盖全部题） |
| 步数 | 1,250（数学 930） |
| 学习率 | 2e-4，cosine，3% warmup，AdamW，无 weight decay |
| LoRA dropout / 精度 / 长度上限 | 0.05 / bf16 / 8,192 token |

**GRPO**：TRL 0.25.1 的 `GRPOTrainer`，`loss_type='grpo'`。学生对每道题采样 8 个回答，最终答案对的奖励 1、错的奖励 0（和评测用同一个评分函数 `tdwa/grade.py`），奖励在同一道题的 8 个回答内做标准化。

| 项目 | 取值 |
|---|---|
| 轮数 / 每步 | 1 轮 / 8 题 × 8 回答 = 64 个回答 |
| 步数 | 250（数学 186） |
| 学习率 | 1e-5，cosine，3% warmup |
| 裁剪 epsilon / KL 系数 beta | 0.2 / 0.04（参考策略是不带适配器的原始 Base） |
| 采样 | temperature 1.0，top_p 1.0，提示上限 4,096、回答上限 2,048 token |
| LoRA dropout / 精度 | 0 / bf16 |

和 TRL 0.25.1 默认值不同的地方有四处：`loss_type` 默认是 `dapo`，这里用 `grpo`；`beta` 默认 0，这里 0.04；`learning_rate` 默认 1e-6，这里 1e-5；`mask_truncated_completions` 默认关，这里开（被截断的回答不算损失）。采样用和训练同卡的 vLLM，重要性修正保持 TRL 默认（开，上限 2）。完整配置见 `outputs/engineer/grpo_main/TRAIN_PROTOCOL.json`。

**评测**（两种方法相同）：vLLM bf16、关闭思考、贪心解码、最多 2,048 个新 token。主指标：医疗为规则类教过组 160 题，通用为 BBH 追踪 750 题，数学为 MATH 277 题，代码为 CRUXEval 留出 550 题。副作用为 ARC-C、HellaSwag、MMLU、GSM8K 四项相对同协议 Base 的平均变化。

## 2. 现状（2026-11-27）

- **训练题已齐并冻结**：`outputs/engineer/grpo_main/data/t{1,2,4,5}.jsonl`。每个 bench 都包含原来 100 道可见题；隐藏验证和测试题按题目哈希排除，这次拷贝后重新跑过 `validate_data.py`，四个 bench 与隐藏题重合数都是 0。
- **GRPO 的 12 组现在就能跑**，不依赖教师。
- **蒸馏的 12 组还不能跑**，因为教师示范没生成完：

  | bench | 已通过校验的示范 | 还缺 |
  |---|---:|---:|
  | 医疗 t1 | 1,794 / 2,000 | 206 |
  | 通用 t2 | 1,995 / 2,000 | 5 |
  | 数学 t4 | 405 / 1,488 | 1,083 |
  | 代码 t5 | 100 / 2,000 | 1,900 |

  合计还缺 3,194 条。某个 bench 的示范全部通过校验后才会生成 `teacher_data/<bench>.ready.json`，蒸馏脚本只认这个文件，不会用不完整的示范训练。
- **12 个同协议 Base 分数已有**（masdar 上测的），结果表里已经填好，可以直接当比较基线。
- masdar 上千级 GRPO 跑完过 1 组（Qwen3 1.7B、医疗、seed 0：23.13%，同协议 Base 25.00%），另有 8 组中途停止。这些适配器和检查点没有进 git，留在 `Few_Shot_Weight_Adaptation_masdar_20261127/outputs/engineer/grpo_main/runs/`。在新目录里这 12 组都按“未完成”处理、从头训练。把旧检查点搬到新目录续跑这件事我没有验证过。

## 3. 前置条件

以下路径都相对仓库根目录。除第 2 项外都不在 git 里，需要自己准备。

1. `.venv-vllm`：torch 2.8.0、vLLM 0.11.0、transformers 4.57.1 的 Python 环境。
2. 训练库：运行 `bash outputs/engineer/grpo_main/setup_deps.sh`（需要 `uv` 和网络），把 TRL 0.25.1、PEFT 0.17.1、accelerate 1.11.0 装到 `outputs/engineer/grpo_main/deps/`，不改动 `.venv-vllm`。
3. `cache/hf`：HuggingFace 缓存目录（可以是软链接）。脚本以离线模式运行，里面必须已有 `Qwen/Qwen3-1.7B`、`Qwen3-4B`、`Qwen3-14B`，以及评测用的数据集 `cais/mmlu`、`openai/gsm8k`、`allenai/ai2_arc`、`Rowan/hellaswag`。
4. `_secret/t1`、`t2`、`t4`、`t5`：隐藏测试题。文件指纹必须和已有 Base 记录里的一致，否则 Base 分数不能复用。
5. 教师密钥 `../.secrets/cliproxy.env`（仓库的上一级目录），内含 `CLIPROXY_BASE_URL` 和 `CLIPROXY_API_KEY`。只有第 4 步补教师示范时需要。
6. 显卡。masdar 的 H200（141 GiB）上实测的训练显存峰值如下，执行脚本每张卡同时只跑一组：

   | | Qwen3 1.7B | Qwen3 4B | Qwen3 14B |
   |---|---:|---:|---:|
   | GRPO 训练（含同卡 vLLM） | 约 55 GiB | 约 48 GiB | 约 98 GiB |
   | 蒸馏训练 | 未记录 | 未记录 | 35.6 GiB |

   这套设置只在 H200 上跑通过。GRPO 里 vLLM 的 KV 缓存固定为 16 GiB（14B 为 24 GiB），评测同样固定。48 GB 的卡装不下 14B 的 GRPO（峰值 98 GiB）；1.7B 和 4B 的 GRPO 在 48 GB 卡上没有试过。

准备好后先跑检查，全部 `OK` 再继续：

```bash
P=.venv-vllm/bin/python; R=outputs/engineer/grpo_main/run_k.py
$P $R --check
```

## 4. 执行步骤

所有命令都在仓库根目录运行。

**第 1 步：看清单。** 列出 24 组和各自的状态（`train` 待训练、`evaluate` 已训练待评测、`done` 已完成、`blocked` 等教师示范），不启动任何进程。

```bash
$P $R --dry-run
```

**第 2 步：跑 GRPO 的 12 组。** `--gpus` 写要用的显卡编号，每张卡一次跑一组（先训练、再评测），按 1.7B → 4B → 14B 的顺序。

```bash
nohup $P $R --gpus 0,1,2 --methods grpo_k > outputs/engineer/grpo_main/run_k.grpo.log 2>&1 &
```

只跑一部分时用 `--models`、`--tasks` 限定，例如 `--gpus 0 --methods grpo_k --models Qwen3-14B`。

**第 3 步：补齐教师示范（要调用付费的教师接口，先确认预算）。** 脚本会跳过已通过校验的示范，只生成缺的 3,194 条，每次请求 5 道题。

```bash
nohup $P outputs/engineer/grpo_main/generate.py >> outputs/engineer/grpo_main/generator.log 2>&1 &
cat outputs/engineer/grpo_main/teacher_data/heartbeat.json    # 各 bench 已通过校验的条数
```

脚本里写死的上限和已用量（上限在 `generate.py` 开头，要改请自己决定）：

| 上限 | 已用 | 说明 |
|---|---:|---|
| 生成请求 2,400 次 | 831 | |
| token 1,200 万（按最坏情况预留计） | 336 万 | 已新生成的 3,894 条实际用了输入 238 万 + 输出 91 万 token |
| 传输错误 700 次 | 605 | **只剩 95 次余量**，代理再频繁报 503 会很快触顶停止 |
| 每道题最多 4 次尝试 | 没有题用完 | |

四个 bench 都生成 `teacher_data/<bench>.ready.json` 就是完成。脚本退出时如果留下 `teacher_data/INCOMPLETE.json` 或 `AUTH_ERROR.json`，说明触到上限或密钥被拒，里面有原因；修好后重跑同一条命令即可续上。

**第 4 步：跑蒸馏的 12 组。** 哪个 bench 的示范先齐，哪个 bench 就可以先跑。

```bash
nohup $P $R --gpus 0,1,2 --methods distill_k > outputs/engineer/grpo_main/run_k.distill.log 2>&1 &
```

**第 5 步：看结果表。** 每完成一组，`RESULTS_K.md` 自动重写。也可以随时手动刷新：

```bash
$P outputs/engineer/grpo_main/collect_k.py
```

一个格子只有在评测记录通过校验（协议一致、隐藏测试指纹一致、适配器指纹一致）后才会填数；没跑完或校验不过的格子保持为空，不会记成 0。

**可选。**
- 补到 3 个种子：`$P $R --gpus ... --seeds 0 1 2`。
- 在本机重测 12 个 Base（不复用 masdar 的分数）：加 `--with-base`。
- 重新复核训练题：`PYTHONPATH=outputs/engineer/grpo_main/deps:. $P outputs/engineer/grpo_main/validate_data.py`。

**中断和续跑。** 同一条命令重复执行就是续跑：已通过校验的组会跳过；GRPO 每 25 步存一次检查点，蒸馏每轮存一次，训练从最近的检查点接着跑。显存不足时脚本自动把 microbatch 减半重试（有效批量不变），每一步最多试 3 次，仍失败的组标为 `failed`，不影响其他组。要停止时：在 `outputs/engineer/grpo_main/` 下建一个名为 `STOP` 的空文件，当前这组跑完后不再开新的；或者直接终止 `run_k.py`，它会连同正在跑的训练一起结束。每次训练和评测的日志在 `outputs/engineer/grpo_main/logs/`。

## 5. 耗时参考（masdar 实测）

- 千级 GRPO 完整跑完的只有 1 组：Qwen3 1.7B 医疗，250 步用了 3.35 小时（当时同一张卡上还并行着别的作业）。
- 停止时的进度：4B 医疗 90 / 250 步用了 2.88 小时；4B 通用 101 / 250 步用了 2.59 小时；14B 通用 74 / 250 步用了 0.97 小时。
- 百条蒸馏（100 题、5 轮）的训练时间：1.7B 平均 1.2 分钟，4B 平均 2.3 分钟，14B 平均 3.9 分钟。千级蒸馏的题量是它的 20 倍（数学约 15 倍），还没有实测。
- 一次评测（主指标 + 通用能力）用时 2 到 9 分钟。

## 6. 文件说明

都在 `outputs/engineer/grpo_main/` 下。

| 文件 | 作用 |
|---|---|
| `run_k.py` | **执行入口**：检查、列清单、按显卡顺序训练并评测 24 组 |
| `collect_k.py` → `RESULTS_K.md` | 生成结果表 |
| `setup_deps.sh` | 安装 TRL / PEFT / accelerate |
| `worker.py` | 单次训练或单次评测（被 `run_k.py` 调用） |
| `grpo_training.py`、`distill_training.py` | 两种方法的训练实现 |
| `generate.py` | 调教师生成蒸馏示范 |
| `prepare.py`、`validate_data.py` | 生成并冻结训练题、复核训练题（题已冻结，一般不用重跑 `prepare.py`） |
| `data/` | 千级训练题和各自的指纹 |
| `teacher_data/` | 教师示范（每题一个文件）、请求原始记录 `calls/`、用量和尝试次数账本 |
| `evaluations/` | 已有评测记录：12 个同协议 Base、GRPO-100 的 36 次、千级 GRPO 的 1 次 |
| `TRAIN_PROTOCOL.json`、`data_manifest.json`、`data_validation.json` | 训练设置、数据来源、数据复核结果 |

历史保留、这次流程不需要用的：`supervisor.py`、`watchdog.py`、`resource_policy.py`、`validate_resources.py`、`sample_utilization.py`、`status.py`、`collect.py` 的主体、`audit_report_20261127.py` 是 masdar 上的后台调度器（按 H200 的显存做多作业同卡、只允许 GPU 2/4/5/6/7、自动改报告文档），`OPERATIONS.md`、`RESULTS.md`、`summary.json` 是它停止时的说明和结果，`data_100/` 是 GRPO-100 用的 100 题。`outputs/engineer/distill_main/` 是已完成的百条蒸馏：它的 `data/` 里 400 条教师示范被千级蒸馏直接复用，`evaluations/` 里的 12 个 Base 记录是同协议 Base 的来源。

没有放进分支的：训练出的适配器和检查点（`runs/`，约 50 GB）、试跑目录、每次运行的日志和调度器状态、隐藏测试题 `_secret/`、模型缓存、教师密钥。

相对 masdar 副本，改动只有一处：`distill_training.py` 设置进程显存上限时加了 `min(1.0, …)`，否则显存小于预留值的显卡会直接报错；训练设置没有变。

## 7. 看板

masdar 本地看板里，这两件事对应任务 `BL-GRPO-MAIN-20261126`（千级 GRPO）和 `BL-DISTILL-K-20261126`（千级蒸馏），状态是 running、owner 是 `codex-grpo@masdar`，当时只在本地更新、没有推送。接手运行前请按 `README.md` 的多人并行协作规则处理这两个任务的归属。
