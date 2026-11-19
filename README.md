# 教师直接编辑权重的小模型领域适配（TDWA）

一个很强的大模型当“教师”，不做端到端训练，只用很少的数据和算力，通过分析学生模型内部、直接计算并写入权重，让小模型在固定流程、要求高精度的任务上变好。原始想法见 [task.md](task.md)，完整设计见 [DESIGN.md](DESIGN.md)，决定和进展记录见 [HANDOFF.md](HANDOFF.md)，框架改动见 [CHANGELOG.md](CHANGELOG.md)。

**哪些任务做完了、哪些正在做、哪些还没做：见 [TASKS.md](TASKS.md)（任务看板，实时更新）。**

## 多人并行协作规则

这个项目同时有多个进程在做事（不同的 Claude / Codex 会话、ARIS 的不同角色、人）。为了不重复劳动、不互相覆盖，所有人都按下面的规则使用任务看板。看板的唯一数据源是本仓库里的 `tasks/board.json`，[TASKS.md](TASKS.md) 由它自动生成。

1. **开始任何任务之前先看看板。** 运行 `python -m tdwa.board list`（会先从 GitHub 拉取最新状态），或直接在 GitHub 上打开 TASKS.md。只能做状态为“未开始（可认领）”的任务。
2. **先认领，再动手。** `python -m tdwa.board claim <ID> --owner <名字@机器名>`。这条命令会拉取、检查、写入、提交并推送；如果同时有人抢先认领，推送会失败，命令会重新读取看板并拒绝，这时换一个任务。
3. **真正开始运行时标为进行中**：`python -m tdwa.board start <ID> --owner ...`；过程中可以用 `note` 写进展。
4. **完成后立刻标记**：`python -m tdwa.board done <ID> --owner ... --result "一句话结果" --output <产出路径>`；失败用 `fail`；做不下去要交还用 `release`，并写明原因。
5. **只改自己名下的任务。** 脚本会拒绝修改别人认领的任务。看板上没有的新任务，在 `tasks/board.json` 里加一条（状态 `open`）再按上面流程认领。
6. **不要手改 TASKS.md**，它会被重新生成。
7. 负责者统一写成 `名字@机器名`，例如本仓库建立者的会话是 `main@ws006344`。

## 实验规则（所有教学运行都必须遵守）

| 限制 | 内容 |
|---|---|
| 训练 | 不许对整个模型或适配器做端到端梯度训练。允许：权重解析、前向观察、闭式求解（最小二乘等）、对少数选定参数的零阶搜索、单个位置上的局部梯度（少量数据、预算内的局部优化可接受）、新增偏置向量 |
| 数据 | 只能看 100 道真题（含答案）；自编题最多 300 道；任何写进学生的数值都不能靠大量数据拟合得到 |
| 算力 | 显存不超过学生正常推理峰值的 1.5 倍；学生总共最多处理 1000 万个前向 token；反向 token 按 2 倍计入成本 |
| 隔离 | 教师代码在断网的沙盒里运行，看不到验证题、测试题和服务器上的其他目录 |
| 验证 | 隐藏验证集只返回分数，每次运行最多 10 次；最终用教师从未见过的隐藏测试集评估 |
| 副作用 | 每次提交编辑自动检查通用文本困惑度（上升不超过 10%）和 100 道检查题（下降不超过 3 个百分点），不过就自动撤销 |

## 模型

### 学生（被编辑的小模型）

| 简称 | 型号 | 链接 | 状态 |
|---|---|---|---|
| Gemma 1B | google/gemma-3-1b-it | https://huggingface.co/google/gemma-3-1b-it | 已教学 |
| Qwen 1.7B | Qwen/Qwen3-1.7B（关闭思考模式） | https://huggingface.co/Qwen/Qwen3-1.7B | 已教学 |
| Qwen 4B | Qwen/Qwen3-4B（关闭思考模式） | https://huggingface.co/Qwen/Qwen3-4B | 教学中 |
| Gemma 4B | google/gemma-3-4b-it | https://huggingface.co/google/gemma-3-4b-it | 已测基线 |
| Qwen 8B | Qwen/Qwen3-8B | https://huggingface.co/Qwen/Qwen3-8B | 待测基线 |
| Gemma 12B | google/gemma-3-12b-it | https://huggingface.co/google/gemma-3-12b-it | 待测基线 |
| Qwen 14B | Qwen/Qwen3-14B | https://huggingface.co/Qwen/Qwen3-14B | 待测基线 |
| Gemma 27B | google/gemma-3-27b-it | https://huggingface.co/google/gemma-3-27b-it | 未开始 |

1B 到 4B 的学生按全精度（float32）加载，和全部已完成运行一致；8B 以上需要改用半精度（见看板 KERNEL-BF16）。

### 教师（做编辑的大模型）

| 简称 | 型号 ID | 接入方式 |
|---|---|---|
| Fable 5.2 | `claude-fable-5-2`（Anthropic Claude Fable 5.2，https://www.anthropic.com/claude/fable） | 无界面 Claude Code（`claude -p`），只开放三个工具（运行代码、隐藏验证、结束），见 `tdwa/mcp_teach.py`；占用登录账号的订阅额度 |
| Astra 6.2 | `gpt-6.2-astra` | 本地 CLIProxyAPI 的 OpenAI 兼容接口，推理强度 high，见 `tdwa/teach.py` |
| Astra 6 | `gpt-6-astra` | 同上 |

代理的地址和密钥在仓库外的 `Weight_Programming_Eval/.secrets/cliproxy.env`，不在本仓库里。监控、汇报、写说明等辅助工作不用 Fable，用其他模型（如 Opus）。

## 数据集

| 用途 | 数据集 | 链接 | 本项目怎么用 |
|---|---|---|---|
| 任务 T2：物体追踪 | BBH `tracking_shuffled_objects_{three,five,seven}_objects` | https://huggingface.co/datasets/lukaemon/bbh | 750 道原题做隐藏测试；另用 `tdwa/data.py` 按同一格式生成：100 道真题、150 道隐藏验证、300 道新出测试题、150 道更难测试题（多 4 次交换和 9 人题） |
| 任务 T1：临床评分 | MedCalc-Bench Verified | https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified | 教 8 种评分（CURB-65、SIRS、PERC、Wells 肺栓塞、Centor、Child-Pugh、HAS-BLED、CHA2DS2-VASc）：训练集里抽 100 道真题、120 道隐藏验证；测试集里 19 种规则评分共 380 题做隐藏测试（另 11 种没教过的用来看连带影响） |
| 副作用检查 | ARC-Easy / ARC-Challenge | https://huggingface.co/datasets/allenai/ai2_arc | 60 道 ARC-Easy 做编辑时的检查题；300 道 ARC-Challenge 做最终通用能力检查 |
| 副作用检查 | HellaSwag | https://huggingface.co/datasets/Rowan/hellaswag | 300 道做最终通用能力检查 |
| 候选新任务：疾病诊断 | DDXPlus | https://huggingface.co/datasets/aai530-group6/ddxplus | 已测基线（5 选 1，150 题） |
| 候选新任务：编程 | MBPP（sanitized） | https://huggingface.co/datasets/google-research-datasets/mbpp | 已测基线（跑单元测试判分） |
| 候选新任务：数学 | MATH 数论 1–3 级 | https://huggingface.co/datasets/EleutherAI/hendrycks_math | 已测基线（按最终答案判分） |

另有两样不来自公开数据集：40 道两位数加减法检查题由 `tdwa/data.py` 生成；通用文本困惑度用的 `data/generic.txt` 来自本机另一个项目。

隐藏验证和测试题在 `_secret/`，不放进仓库；运行 `python -m tdwa.data` 会用固定随机种子原样重新生成。

## 相关工作

| 工作 | 链接 |
|---|---|
| Takehi 等，Sparks of Weight-Space Programming（博客） | https://rikiyat.github.io/blog/weight-space-programming |
| Exploring Weight-Space Programming | https://arxiv.org/abs/2611.09382 |
| WeightBench | https://arxiv.org/abs/2611.18553 |
| iWSP | https://arxiv.org/abs/2611.18114 |

这些条目当初由调研子代理在网上逐条核对过，写论文前需要再核对一遍。

## 目录

| 路径 | 内容 |
|---|---|
| `tdwa/` | `data.py` 数据划分；`kernel.py` 学生沙盒；`teach.py` 接口式教师循环；`mcp_teach.py` Claude Code 教师；`evaluate.py` 隐藏测试评测；`salvage.py` 被中断运行收尾；`progress.py`、`status.py` 进度；`board.py` 任务看板 |
| `tasks/board.json`、`TASKS.md` | 任务看板 |
| `data/` | 教师可见的 100 道真题、检查题 |
| `pilot/` | 先导实验脚本和报告 |
| `report/` | 报告网页和生成脚本（`build_site.py` 从运行记录生成数据，本地服务 `http://127.0.0.1:8765/index.html`） |
| `logs/queue.sh`、`logs/bigpilot.sh` | 教学运行和大模型基线的排队脚本 |
| `requirements.txt` | 运行环境的 Python 包版本（Python 3.12，torch 2.6.0+cu124，transformers 5.22.2） |
| 不在仓库 | `runs/`（每次运行的完整记录和逐位置权重改动，约 13 GB）、`eval/`、`_secret/`、模型权重 |

## 常用命令

```bash
PY=/mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python
$PY -m tdwa.board list                                  # 看任务看板
$PY tdwa/progress.py                                    # 每次运行一行的进度
$PY -m tdwa.teach --task t1 --student Qwen/Qwen3-4B --teacher gpt-6.2-astra --seed 0 --gpu 0 --effort high   # 接口式教师
$PY -m tdwa.mcp_teach run --task t1 --student Qwen/Qwen3-4B --teacher claude-fable-5-2 --seed 0 --gpu 1      # Fable 教师
$PY -m tdwa.evaluate --run runs/<运行名>                 # 隐藏测试评测
```
