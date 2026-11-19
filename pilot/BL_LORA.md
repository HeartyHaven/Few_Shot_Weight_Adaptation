# LoRA 对照 B2 / B4（ws006343，2026-11-19）

全部用 `tdwa/evaluate_vllm.py`（vLLM，bf16）在隐藏测试上评测，只和同样方式测的未编辑基线比较。超参事先固定（r=16，lr 2e-4，B2 10 轮，B4 1 轮），没有看验证集。
主指标：T2 = BBH 原题 750 道自由作答；T1 = MedCalc 19 种规则评分 380 道。保留检查：ARC-Challenge 300、HellaSwag 300（强制选择）、200 道算术（12 token）、通用文本困惑度。

| 组合 | 种子数 | 主指标 % | ARC-C | HellaSwag | 算术 | 困惑度 |
|---|---|---|---|---|---|---|
| T2 Gemma 1B 未编辑 | – | 56.9 | 47.0 | 35.0 | 44.0 | 11.60 |
| T2 Gemma 1B B2（100 道真题） | 3 | 92.9 ±1.2 | 31.4 | 30.6 | 23.3 | 10.55 |
| T2 Gemma 1B B4（5000 道新生成） | 1 | 99.9 | 25.7 | 25.3 | 7.5 | 11.24 |
| T2 Qwen 1.7B 未编辑 | – | 58.7 | 74.3 | 54.3 | 51.5 | 12.14 |
| T2 Qwen 1.7B B2 | 2 | 98.5 ±0.1 | 60.0 | 42.3 | 1.0 | 12.62 |
| T2 Qwen 1.7B B4 | 1 | 99.9 | 57.0 | 38.7 | 0.0 | 13.10 |
| T1 Gemma 1B 未编辑 | – | 6.3 | 47.0 | 35.7 | 43.5 | 11.61 |
| T1 Gemma 1B B2 | 3 | 16.2 ±2.3 | 30.2 | 26.1 | 15.5 | 23.12 |
| T1 Gemma 1B B4（1 万条训练集） | 1 | 26.3 | 24.0 | 25.3 | 14.0 | 13.45 |
| T1 Qwen 1.7B 未编辑 | – | 13.4 | 74.3 | 54.3 | 51.5 | 12.14 |
| T1 Qwen 1.7B B2 | 3 | 22.9 ±0.9 | 39.7 | 28.6 | 0.0 | 12.91 |
| T1 Qwen 1.7B B4 | 1 | 36.0 | 32.3 | 26.7 | 24.0 | 11.90 |

要点：LoRA 在目标任务上远高于目前的教师编辑，但每一个都违反 DESIGN 9.3 的副作用标准（每项下降 ≤ 2 个百分点），ARC-C 降 7–42 点。Qwen 的 LoRA 在“只答数字”的算术题上几乎全错（原因未核实，推测是回答格式被训成先写步骤）。
产出：`eval/B*.json`、`baselines/*/adapter`（不入库）。重现：`python -m tdwa.baseline_lora --task t2 --student google/gemma-3-1b-it --data few --seed 0` 然后 `python -m tdwa.evaluate_vllm --task t2 --student google/gemma-3-1b-it --lora baselines/B2__t2__gemma-3-1b-it__s0/adapter --name B2__t2__gemma-3-1b-it__s0`。
