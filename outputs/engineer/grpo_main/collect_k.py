"""Results table for the thousand-example Distill / GRPO baselines only.

Writes RESULTS_K.md next to this file from validated evaluation records. A cell stays empty until
its evaluation exists and passes collect.valid (same protocol, same hidden-test fingerprints, adapter
fingerprint matches). It never edits the report document.

    .venv-vllm/bin/python outputs/engineer/grpo_main/collect_k.py
"""
import math
import statistics
import time

from common import HERE, TASKS, MODELS, SEEDS, BASELINE_INFO, read_json, run_name
from collect import RET_KEYS, main_metric, valid

METHODS = ('distill_k', 'grpo_k')
# Column order of the main table in the report.
COLUMNS = (('t2', '通用数据指标'), ('t1', '医疗指标（规则类）'), ('t4', '数学指标'), ('t5', '代码指标'))
TASK_NAME = {'t1': 't1 医疗', 't2': 't2 通用', 't4': 't4 数学', 't5': 't5 代码'}


def n_examples(task):
    return read_json(HERE / 'data' / f'{task}.ready.json')['n_examples']


def planned_steps(method, task):
    n = n_examples(task)
    return 5 * math.ceil(n / 8) if method == 'distill_k' else n // 8


def teacher_progress():
    return {t: (len(list((HERE / 'teacher_data' / t).glob('*.json'))), n_examples(t),
                (HERE / 'teacher_data' / f'{t}.ready.json').exists()) for t in TASKS}


def gather():
    bases = {}
    for model in MODELS:
        for task in TASKS:
            record = read_json(HERE / 'evaluations' / f'GRPO_BASE__{task}__{model}.json')
            bases[model, task] = record if valid(record, task, model, baseline='Base-matched') else None
    cells = {}
    for method in METHODS:
        for model in MODELS:
            for task in TASKS:
                base = bases[model, task]
                runs = []
                for seed in SEEDS:
                    name = run_name(task, model, seed, method)
                    record = read_json(HERE / 'evaluations' / f'{name}.json')
                    if base and valid(record, task, model, base, method) and record.get('seed') == seed:
                        done = read_json(HERE / 'runs' / name / 'train_done.json', {})
                        runs.append({
                            'seed': seed, 'score': main_metric(record),
                            'side': statistics.mean((record['results']['retention'][k]['acc']
                                                     - base['results']['retention'][k]['acc']) * 100 for k in RET_KEYS),
                            'hours': done.get('wall_seconds', float('nan')) / 3600,
                            'peak': done.get('peak_mem_gib', float('nan'))})
                cells[method, model, task] = {'base': main_metric(base) if base else None, 'runs': runs}
    return cells


def spread(values, digits, unit=''):
    if not values:
        return ''
    if len(values) == 1:
        return f'{values[0]:.{digits}f}{unit}'
    return f'{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}{unit}'


def render():
    cells = gather()
    done = {m: sum(bool(cells[m, model, task]['runs']) for model in MODELS for task in TASKS) for m in METHODS}
    teacher = teacher_progress()
    lines = [
        '# 千级数据 Distill / GRPO 结果表',
        '',
        '本文件由 `collect_k.py` 生成，不要手改；空格子表示该组还没有通过校验的评测结果。',
        f'更新时间：{time.strftime("%F %T %Z")}。'
        f'已有结果的格子：Distill-K {done["distill_k"]}/12，GRPO-K {done["grpo_k"]}/12。',
        '教师示范（蒸馏用）：' + '，'.join(
            f'{t} {have}/{total}{"（已冻结）" if ready else ""}' for t, (have, total, ready) in teacher.items()) + '。',
        '',
        '## 主表（每个模型 × 每个 bench）',
        '',
        '单位：准确率 %；单个种子直接写数值，多个种子写均值 ± 样本标准差。'
        '平均副作用 = ARC-C、HellaSwag、MMLU、GSM8K 四项相对同协议 Base 的平均变化，四个 bench 都有结果后才填。',
        '',
        '|  | ' + ' | '.join(title for _, title in COLUMNS) + ' | 平均副作用 |',
        '| :---- | :---- | :---- | :---- | :---- | :---- |',
    ]
    for model in MODELS:
        lines.append('| ' + model.replace('Qwen3-', 'Qwen3 ') + ' |  |  |  |  |  |')
        bases = [cells['grpo_k', model, task]['base'] for task, _ in COLUMNS]
        lines.append('| Base（同协议） | ' + ' | '.join('' if b is None else f'{b:.1f}%' for b in bases)
                     + (' | 0.0 pp |' if all(b is not None for b in bases) else ' |  |'))
        for method in METHODS:
            row = [spread([r['score'] for r in cells[method, model, task]['runs']], 1, '%') for task, _ in COLUMNS]
            sides = [statistics.mean(r['side'] for r in cells[method, model, task]['runs'])
                     for task, _ in COLUMNS if cells[method, model, task]['runs']]
            side = f'{statistics.mean(sides):+.1f} pp' if len(sides) == len(COLUMNS) else ''
            lines.append('| ' + BASELINE_INFO[method]['label'] + ' | ' + ' | '.join(row) + ' | ' + side + ' |')
    lines += [
        '',
        '## 明细（每行一组）',
        '',
        '| 方法 | 学生 | 任务 | 训练题数 | 训练步数 | 同协议 Base | 得分 | 相对 Base（pp） | 副作用 Δ（pp） | 完成种子 | 训练用时（h） | 峰值显存（GiB） |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|',
    ]
    for method in METHODS:
        for model in MODELS:
            for task in TASKS:
                cell = cells[method, model, task]
                runs = cell['runs']
                scores = [r['score'] for r in runs]
                gain = f'{statistics.mean(scores) - cell["base"]:+.2f}' if runs else ''
                lines.append(
                    f'| {method} | {model} | {TASK_NAME[task]} | {n_examples(task)} | {planned_steps(method, task)} | '
                    f'{"" if cell["base"] is None else format(cell["base"], ".2f")} | {spread(scores, 2)} | {gain} | '
                    f'{spread([r["side"] for r in runs], 2)} | {",".join(str(r["seed"]) for r in runs)} | '
                    f'{spread([r["hours"] for r in runs], 2)} | {spread([r["peak"] for r in runs], 1)} |')
    lines += [
        '',
        '主指标：t1 为 MedCalc 规则类教过组 160 题，t2 为 BBH 追踪 750 题，t4 为 MATH 277 题，t5 为 CRUXEval 留出 550 题。',
        '训练步数是固定预算（Distill：5 轮、每步 8 题；GRPO：1 轮、每步 8 题 × 8 个回答），不是实测值。',
        '流程与设置见仓库根目录 `BASELINES_K.md`。',
    ]
    return '\n'.join(lines) + '\n'


def write():
    text = render()
    (HERE / 'RESULTS_K.md').write_text(text)
    return text


if __name__ == '__main__':
    print(write())
