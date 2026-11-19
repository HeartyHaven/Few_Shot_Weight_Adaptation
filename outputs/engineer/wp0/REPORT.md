# WP0：阻塞，尚不能判定任何组合的准入

当前执行会话没有可访问的 NVIDIA GPU，且 Hugging Face / PyPI 域名解析失败。已完成环境预检并用 uv 建立 `outputs/engineer/.venv`，尚未安装实验依赖、加载学生或执行基线。**没有准确率、置信区间、编辑效果或显存倍数可供验收。** 本报告是阻塞交接，不是 WP0 实验完成报告；任何组合都不能据此判为能力不足。

执行依据：项目当前 `HANDOFF.md` 的 WP0 指派和 `DESIGN.md`；输入文件摘要见 `preflight.json` 的 `project_input_sha256`。仅执行 WP0，没有进入后续工作包。

## 可复现的实测证据

以下命令均在项目根目录执行。命令 A/B 输出 JSON；保存的原始证据分别是 `preflight.json`、`torch_probe.json`。时间戳、空闲磁盘和网络状态会随运行时点变化，其余诊断字段可供 main 抽查。

**命令 A：预检**

```bash
PYTHONDONTWRITEBYTECODE=1 outputs/engineer/.venv/bin/python outputs/engineer/wp0/scripts/preflight.py
```

**命令 B：在已有 CUDA 版 PyTorch 中独立核验设备可用性**

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HOME=/mnt/data2/users/lang.gao/huggingface_cache HF_HUB_OFFLINE=1 /mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python outputs/engineer/wp0/scripts/probe_torch.py
```

已有环境只用作只读诊断，没有修改其依赖，没有把它冒充为本项目建好的实验环境。

| 实测项 | 结果与重现命令 |
|---|---|
| GPU 设备文件 | `/dev/nvidia*` 未发现；命令 A |
| NVIDIA 驱动访问 | `nvidia-smi` 退出码 **9**，提示无法与驱动通信；命令 A |
| CUDA 可用性 | `cuda_available=false`，`cuda_device_count=0`；命令 B |
| 独立诊断环境 | torch **2.6.0+cu124**，编译 CUDA **12.4**，transformers **5.22.2**；命令 B |
| 网络 | `huggingface.co`、`pypi.org` 均返回 `gaierror`，errno **-3**；命令 A |
| 磁盘 | `df -h` 显示根分区约 **6.5 GB**、数据盘约 **14 GB** 可用；准确字节值以 `preflight.json` 为准；命令 A |
| Gemma 缓存 | `HF_HOME` 根目录和其 `hub/` 下均未发现 `google/gemma-3-1b-it`；命令 A |
| Gemma token | token 文件存在；未读取内容，权限尚未验证，不能认定许可拒绝；命令 A |
| Qwen 缓存 | 完整权重分片位于 `HF_HOME/models--Qwen--Qwen3-1.7B`，revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；`hub/` 中对应快照仅有配置和 tokenizer 文件；命令 A |
| Qwen 配置 | 配置文件记载 **28** 层、hidden size **2048**、MLP width **6144**，聊天模板支持 `enable_thinking`；只是静态配置核验，未加载模型；命令 A |
| 数据集缓存 | 在所列 `HF_HOME` 根目录及 `hub/` 标准目录未找到 MedCalc-Bench-Verified 或 BBH；不是全磁盘不存在的断言；命令 A |

**观测与推断区分：** GPU 不可见和 DNS 失败是本执行会话的实测结果；可能与运行隔离有关，但尚未证明整台宿主机的驱动或网络损坏。不可据此建议重装驱动。Qwen 的缓存路径差异是实测，后续需显式指定完整的本地快照或正确 `cache_dir`，避免把默认 `HF_HOME/hub` 当成已有完整权重。

## 环境产物与写入边界

环境创建命令：

```bash
bash outputs/engineer/wp0/scripts/create_env.sh
```

已经执行；输出见 `environment_setup.log`。脚本只创建指定 venv 和目录内临时目录，不自动覆盖已有环境。使用已有解释器、离线、无缓存、无 seed 包；设置了指派中的 `HF_HOME` 和 `PIP_CACHE_DIR`，尚未对共享缓存执行写入。项目 venv 尚无 torch（命令 A），不应当视为依赖安装完成。

没有下载权重、保存完整权重副本、删除已有文件或调用教师。所有工程文件和临时文件均写入 `outputs/engineer/`；唯一另行授权的操作是末尾执行指派中的 `superaris report`，它由既有机制更新传输记录。未改项目目标、设计、交接记录或其他角色文件。

## 未测项目与准入状态

| 任务 | Gemma | Qwen | 原因 |
|---|---|---|---|
| T1 规则类计算器 | 未判定 | 未判定 | 未加载数据与模型，缺教前准确率及 B1 提升证据 |
| T2 三物体 | 未判定 | 未判定 | 同上 |
| T2 五物体 | 未判定 | 未判定 | 同上 |
| T2 七物体 | 未判定 | 未判定 | 同上 |

`baseline.json` 将每个要求的 B0/B1 × 直接/自由作答条件标为 `not_run`，准确率与置信区间为 `null`。MedCalc 规则类计算器数量、每类测试条数、教前准确率处于准入区间的候选教学组均未测；没有用旧版本的数量替代。没有读取测试题、写流程提示词或进行提示词调整，也没有产生每题输出。

`memory.json` 将推理、全部层激活钩子、单层目标向量反传的峰值与倍数标为未测；没有用 CUDA 不可用时的零值冒充峰值。

`gemma_arch.txt` 和 `smoke_edit.json` 明确标为未执行。未确认 Gemma 实际模块结构，未检验归一化是否抵消编辑幅度；不能给出闭式编辑安全或失效的结论。两学生的确定性生成同样未测。

## main 可执行的恢复步骤

先在具备 GPU 和网络访问权限的执行环境重跑命令 A/B，确认设备可见且 torch 能初始化 CUDA；使用当前角色既有授权恢复任务。确认 Gemma 下载许可与共享缓存写入边界后，再安装依赖、下载缺失资源，下载前重新检查磁盘。当前剩余空间不等于依赖和模型安装一定够用；本轮没有据此启动大下载。

随后继续原 WP0：先 Gemma 再 Qwen，按 `HANDOFF.md` 测确定性、实际模块结构、真实权重秩一编辑、两种提示条件和两种作答模式及对应 bootstrap 区间，以及同条件显存峰值。数据解锁前冻结流程提示词和判分规则。完成测量后再判断准入；不得根据本次环境失败把组合标记为科学负结果。

阻塞回报使用原句柄 `wp0-pilot`，不创建新任务或恢复任何 Goal。实际回报结果保存于同目录 `report_receipt.json`（若命令失败，该文件保存实际失败输出）。
