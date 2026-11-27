"""Publish only verified completed evaluations; retain the existing table contents."""
import json
import os
from pathlib import Path
import statistics
import time

from common import HERE, ROOT, MODELS, TASKS, SEEDS, EVAL_PROTOCOL, atomic_json, read_json, run_name, sha

DOC = ROOT / 'report/slides/GPT-Guided Model Editing.md'
RET_KEYS = ('arc_challenge','hellaswag','mmlu','gsm8k')


def main_metric(record):
    r=record['results']; task=record['task']
    if task=='t1': return r['test']['taught']['acc']*100
    return r['test_bbh' if task=='t2' else 'test']['free']['acc']*100


def valid(record, task, model, base=None):
    if not record or record.get('task')!=task or record.get('student')!='Qwen/'+model:
        return False
    if record.get('evaluation_protocol')!=EVAL_PROTOCOL: return False
    try:
        metric=main_metric(record)
        ret=record['results']['retention']
        if not 0<=metric<=100 or any(k not in ret for k in RET_KEYS): return False
        for rel,digest in record['test_fingerprints'].items():
            if sha(ROOT/rel)!=digest: return False
        if record['grader_sha256']!=sha(ROOT/'tdwa/grade.py'): return False
        if base and any(record.get(k)!=base.get(k) for k in ('dtype','engine','max_new_tokens','test_fingerprints','grader_sha256','evaluation_protocol','max_num_seqs')):
            return False
        if record.get('baseline')=='B3-100':
            ready=read_json(HERE/'data'/f'{task}.ready.json')
            if not ready or record['data_sha256']!=ready['sha256']: return False
        return True
    except (KeyError,TypeError,FileNotFoundError):
        return False


def collect(update_doc=True):
    data={'updated':time.time(),'protocol':{'train_examples':100,'synthetic_examples':0,'seeds':list(SEEDS),
          'teacher':'gpt-6.2-astra','dtype':'bfloat16','max_new_tokens':2048,
          'test_counts':{'t1':160,'t2':750,'t4':277,'t5':550},'selection':'all 3 fixed seeds; no test-based selection'},'models':{}}
    lines=['# 主表蒸馏进度','', '固定 100 条可见题，Astra 教师示范，LoRA，3 个种子。缺失结果不作零分。','',
           '| 学生 | 任务 | 同协议 Base | 蒸馏均值 ± SD | 完成种子 | 平均副作用 Δ pp |',
           '|---|---|---:|---:|---:|---:|']
    count=0
    for model in MODELS:
        entries={}
        for task in TASKS:
            base=read_json(HERE/'evaluations'/f'B3_BASE__{task}__{model}.json')
            if not valid(base,task,model): base=None
            records=[]
            for seed in SEEDS:
                r=read_json(HERE/'evaluations'/f'{run_name(task,model,seed)}.json')
                if valid(r,task,model,base) and base and r.get('seed')==seed: records.append(r)
            scores=[main_metric(r) for r in records]
            changes=[statistics.mean((r['results']['retention'][k]['acc']-base['results']['retention'][k]['acc'])*100 for k in RET_KEYS) for r in records] if base else []
            e={'base':main_metric(base) if base else None,'n':len(records),'seeds':[r['seed'] for r in records],
               'scores':scores,'mean':statistics.mean(scores) if scores else None,
               'sd':statistics.stdev(scores) if len(scores)>1 else None,
               'retention_delta':statistics.mean(changes) if changes else None,
               'files':[r['name'] for r in records]}
            entries[task]=e; count+=len(records)
            fmt=lambda x:'—' if x is None else f'{x:.2f}'
            lines.append(f'| {model} | {task} | {fmt(e["base"])} | {fmt(e["mean"])} ± {fmt(e["sd"])} | {e["n"]}/3 | {fmt(e["retention_delta"])} |')
        data['models'][model]=entries
    data['completed_seed_evaluations']=count; data['expected_seed_evaluations']=36
    atomic_json(HERE/'summary.json',data)
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if update_doc and DOC.exists():
        update_table(data)
    return data


def update_table(data):
    original=DOC.read_text(); fingerprint=sha(DOC)
    if not (HERE/'source_report.before_distill.md').exists():
        (HERE/'source_report.before_distill.md').write_text(original)
    start=original.find('### Results')
    stop=original.find('主表口径',start)
    if start<0 or stop<0: raise RuntimeError('Cannot locate the requested main table safely')
    section=original[start:stop]
    result=[]; current=None
    for line in section.splitlines():
        for model in MODELS:
            if line.startswith('| '+model.replace('Qwen3-','Qwen3 ')+' |'):
                current=model
        if line.startswith('| Distill（Astra，100 条）') or line.startswith('| Base（蒸馏同协议）'):
            continue
        result.append(line)
        if line.startswith('| LoRA |') and current:
            entries=data['models'][current]; order=('t2','t1','t4','t5')
            base=['—' if entries[t]['base'] is None else f'{entries[t]["base"]:.1f}%' for t in order]
            values=[]
            for task in order:
                e=entries[task]
                if e['n']==0: values.append('待测（0/3）')
                elif e['n']==3: values.append(f'{e["mean"]:.1f} ± {e["sd"]:.1f}%')
                else: values.append(f'{e["mean"]:.1f}%（{e["n"]}/3）')
            complete=all(entries[t]['n']==3 for t in TASKS)
            ret=f'{statistics.mean(entries[t]["retention_delta"] for t in TASKS):+.1f} pp' if complete else '待齐'
            result.append('| Base（蒸馏同协议） | '+' | '.join(base)+' | 0.0 pp |')
            result.append('| Distill（Astra，100 条） | '+' | '.join(values)+' | '+ret+' |')
    updated=original[:start]+'\n'.join(result).rstrip()+'\n\n'+original[stop:]
    begin='<!-- DISTILL_MAIN_PROTOCOL_BEGIN -->'; end='<!-- DISTILL_MAIN_PROTOCOL_END -->'
    note=(begin+'\n蒸馏新增行：GPT-6.2 Astra 在每任务固定 100 条可见真题上生成步骤示范（不加合成题），'
          'bf16 LoRA，固定种子 0/1/2，均值 ± 样本标准差；不足 3 次时明确标出完成数。'
          '训练 5 轮、r=16、lr=2e-4、有效 batch=8；不按隐藏测试挑种子或超参。'
          '配套 Base 与蒸馏用相同 vLLM/bf16、提示和 2048 token 上限；'
          '主指标题量为 BBH 750、MedCalc 教过组 160、MATH 277、CRUXEval 留出 550。'
          'CRUXEval 的 100 条可见题和 150 条验证题不进入蒸馏测试，不能与旧表的 800 题结果直接比较。'
          '副作用统一为 ARC-C、HellaSwag、MMLU、GSM8K 相对配套 Base 的平均变化；GSM8K 上限 512 token。'
          '明细和训练成本见 `outputs/engineer/distill_main/summary.json` 与 `RESULTS.md`。\n'+end)
    if begin in updated:
        a=updated.index(begin); b=updated.index(end,a)+len(end)
        updated=updated[:a]+note+updated[b:]
    else:
        i=updated.index('主表口径',updated.index('### Results'))
        updated=updated[:i]+note+'\n\n'+updated[i:]
    # Preserve concurrent human changes; retry on the next watchdog pass.
    if updated!=original and sha(DOC)==fingerprint:
        tmp=DOC.with_name(DOC.name+f'.distill.{os.getpid()}.tmp')
        tmp.write_text(updated); os.replace(tmp,DOC)


if __name__=='__main__':
    s=collect(); print(f'Verified seed evaluations: {s["completed_seed_evaluations"]}/36')
