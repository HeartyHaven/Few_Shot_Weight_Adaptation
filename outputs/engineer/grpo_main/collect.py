"""Publish validated method-by-data-size evaluations with shared Base provenance."""
import math
import os
import statistics
import time
from common import (HERE, ROOT, TASKS, MODELS, SEEDS, EVAL_PROTOCOL, BASELINES, BASELINE_INFO,
                    atomic_json, read_json, run_name, sha, dataset_dir)

DOC=ROOT/'report/slides/GPT-Guided Model Editing.md'
RET_KEYS=('arc_challenge','hellaswag','mmlu','gsm8k')
PROTOCOL_KEYS=('dtype','engine','max_new_tokens','gsm8k_max_new_tokens','enable_thinking','test_fingerprints',
    'grader_sha256','evaluator_sha256','evaluation_protocol','kv_cache_gib','max_num_seqs','max_num_batched_tokens')


def main_metric(record):
    r=record['results'];task=record['task']
    if task=='t1': return r['test']['taught']['acc']*100
    return r['test_bbh' if task=='t2' else 'test']['free']['acc']*100


def valid(record,task,model,base=None,baseline=None):
    if not record or record.get('task')!=task or record.get('student')!='Qwen/'+model: return False
    try:
        method=record['baseline']
        if baseline is not None and method!=baseline: return False
        if method not in (*BASELINES,'Base-matched'): return False
        expected={'dtype':'bfloat16','engine':'vllm','max_new_tokens':2048,'gsm8k_max_new_tokens':512,
            'enable_thinking':False,'evaluation_protocol':EVAL_PROTOCOL,
            'kv_cache_gib':24 if model=='Qwen3-14B' else 16,'max_num_seqs':64,'max_num_batched_tokens':8192}
        if any(record.get(k)!=v for k,v in expected.items()): return False
        if not 0<=main_metric(record)<=100: return False
        if record['grader_sha256']!=sha(ROOT/'tdwa/grade.py'): return False
        if record['evaluator_sha256']!=sha(ROOT/'tdwa/evaluate_vllm.py'): return False
        fingerprints={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'_secret'/task).glob('test*.jsonl'))}
        if record['test_fingerprints']!=fingerprints: return False
        results=record['results'];ret=results['retention']
        if not math.isfinite(ret['ppl_generic']) or ret['ppl_generic']<=0: return False
        for k,n in [('arc_challenge',300),('hellaswag',300),('mmlu',1140),('gsm8k',500),('arithmetic',200)]:
            if not 0<=ret[k]['acc']<=1 or ret[k]['n']!=n: return False
        primary=results['test']['taught'] if task=='t1' else results['test_bbh' if task=='t2' else 'test']['free']
        if primary['n']!={'t1':160,'t2':750,'t4':277,'t5':550}[task]: return False
        if task=='t2' and any(results[k]['free']['n']!=n for k,n in [('test_fresh',300),('test_ood',150)]): return False
        if base and any(record.get(k)!=base.get(k) for k in PROTOCOL_KEYS): return False
        if method in BASELINES:
            ready=read_json(dataset_dir(method)/f'{task}.ready.json')
            done=read_json(HERE/'runs'/record['name']/'train_done.json')
            if not ready or not done or record['data_sha256']!=ready['sha256']: return False
            if record['adapter_sha256']!=done['adapter_sha256']: return False
            if record['name']!=run_name(task,model,record['seed'],method): return False
            if record['adapter_sha256']!=sha(HERE/'runs'/record['name']/'adapter/adapter_model.safetensors'): return False
        else:
            if record.get('seed') is not None: return False
            if record.get('reused_from') and sha(ROOT/record['reused_from'])!=record['reused_source_sha256']: return False
        return True
    except (KeyError,TypeError,FileNotFoundError,ValueError): return False


def reuse_bases():
    for task in TASKS:
        for model in MODELS:
            src=ROOT/'outputs/engineer/distill_main/evaluations'/f'B3_BASE__{task}__{model}.json'
            data=read_json(src)
            if not valid(data,task,model,baseline='Base-matched'): raise RuntimeError(f'Invalid matched Base: {src}')
            name=f'GRPO_BASE__{task}__{model}'
            data.update(name=name,reused_from=str(src.relative_to(ROOT)),reused_source_sha256=sha(src))
            target=HERE/'evaluations'/f'{name}.json'
            if target.exists() and read_json(target)!=data: raise RuntimeError(f'Existing Base differs: {target}')
            atomic_json(target,data)


def collect(update_doc=True):
    data={'updated':time.time(),'baselines':{},'completed_seed_evaluations':0,'expected_seed_evaluations':108,
        'matched_base_count':0,'base_source':'validated protocol-2 Base from completed Distill-100 batch',
        'data':read_json(HERE/'data_manifest.json',{}),'protocol':read_json(HERE/'TRAIN_PROTOCOL.json',{}),
        'teacher_usage':read_json(HERE/'teacher_data/usage.json',{}),
        'teacher_progress':read_json(HERE/'teacher_data/heartbeat.json',{})}
    bases={}
    for model in MODELS:
        for task in TASKS:
            b=read_json(HERE/'evaluations'/f'GRPO_BASE__{task}__{model}.json')
            if valid(b,task,model,baseline='Base-matched'):
                bases[model,task]=b;data['matched_base_count']+=1
    lines=['# 主表新增 baseline 结果与进度','','方法 × 数据量：已有 Distill-100，新增 GRPO-100 / GRPO-K / Distill-K。',
        '每种新增方法：3 学生 × 4 任务 × seeds 0/1/2；均值 ± 样本标准差。未完成项不记零分。',
        '两档分别配对相同题目；K 包含原 100 题。数据量相同不代表优化步数、token 或算力相同。',
        'Base 复用同协议结果。只报告固定预算最终检查点，不根据隐藏测试选择或调参。','',
        '| 方法 | 学生 | 任务 | 训练题数 | 同协议 Base | 均值 ± SD | 完成种子 | 平均副作用 Δ pp |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.2f}'
    for baseline in BASELINES:
        group={'models':{},'completed_seed_evaluations':0,'expected_seed_evaluations':36}
        for model in MODELS:
            entries={}
            for task in TASKS:
                base=bases.get((model,task));records=[]
                for seed in SEEDS:
                    r=read_json(HERE/'evaluations'/f'{run_name(task,model,seed,baseline)}.json')
                    if base and valid(r,task,model,base,baseline) and r.get('seed')==seed: records.append(r)
                scores=[main_metric(r) for r in records]
                changes=[statistics.mean((r['results']['retention'][k]['acc']-base['results']['retention'][k]['acc'])*100
                    for k in RET_KEYS) for r in records]
                count=100 if baseline=='grpo_100' else data['data'].get('tasks',{}).get(task,{}).get('n_examples')
                entry={'base':main_metric(base) if base else None,'n':len(records),'seeds':[r['seed'] for r in records],
                    'scores':scores,'mean':statistics.mean(scores) if scores else None,
                    'sd':statistics.stdev(scores) if len(scores)>1 else None,
                    'retention_delta':statistics.mean(changes) if changes else None,'files':[r['name'] for r in records],
                    'train_examples':count}
                entries[task]=entry;group['completed_seed_evaluations']+=len(records)
                lines.append(f'| {baseline} | {model} | {task} | {count} | {fmt(entry["base"])} | '
                    f'{fmt(entry["mean"])} ± {fmt(entry["sd"])} | {entry["n"]}/3 | {fmt(entry["retention_delta"])} |')
            group['models'][model]=entries
        data['baselines'][baseline]=group
        data['completed_seed_evaluations']+=group['completed_seed_evaluations']
    lines+=['','数据：医疗/追踪/代码各 2,000，数学 1,488；[数据清单](data_manifest.json)、[配对校验](data_validation.json)。',
        '医疗为 MedCalc 公开规则类训练题，数学为两个领域全部难度训练题；追踪独立生成；代码为原 100 题 + 1,900 道有执行答案的生成程序。',
        'GRPO 固定 1 epoch、每题 8 个回答；100 档每步 4 题（25 步），K 档每步 8 题（250/186 步）。',
        'Distill 固定 5 epochs、batch 8、reply-only SFT，K 档 1,250/930 步；复用原 400 条教师示范，其余题重新生成步骤示范。',
        '两种训练均从原始 Base 初始化，LoRA r16/alpha32。更多设置及成本见 [训练协议](TRAIN_PROTOCOL.json)、[操作说明](OPERATIONS.md)。']
    atomic_json(HERE/'summary.json',data)
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if update_doc: update_table(data)
    return data


def update_table(data):
    original=DOC.read_text();fingerprint=sha(DOC)
    start=original.index('### Results');stop=original.index('<!-- DISTILL_MAIN_PROTOCOL_BEGIN -->',start)
    result=[];current=None
    for line in original[start:stop].splitlines():
        for model in MODELS:
            if line.startswith('| '+model.replace('Qwen3-','Qwen3 ')+' |'): current=model
        if any(line.startswith('| '+BASELINE_INFO[b]['label']+' |') for b in BASELINES): continue
        result.append(line)
        if line.startswith('| Distill（Astra，100 条） |') and current:
            for baseline in BASELINES:
                entries=data['baselines'][baseline]['models'][current];values=[]
                for task in ('t2','t1','t4','t5'):
                    e=entries[task]
                    values.append('待测（0/3）' if not e['n'] else
                        f'{e["mean"]:.1f} ± {e["sd"]:.1f}%' if e['n']==3 else f'{e["mean"]:.1f}%（{e["n"]}/3）')
                ret=f'{statistics.mean(e["retention_delta"] for e in entries.values()):+.1f} pp' if all(e['n']==3 for e in entries.values()) else '待齐'
                result.append('| '+BASELINE_INFO[baseline]['label']+' | '+' | '.join(values)+' | '+ret+' |')
    updated=original[:start]+'\n'.join(result).rstrip()+'\n\n'+original[stop:]
    begin='<!-- GRPO_MAIN_STATUS_BEGIN -->';end='<!-- GRPO_MAIN_STATUS_END -->'
    n=data['completed_seed_evaluations']
    details='；'.join(f"{b} {data['baselines'][b]['completed_seed_evaluations']}/36" for b in BASELINES)
    note=(begin+'\n新增对照：'+('全部完成' if n==108 else '进行中')+f'，完整种子评测 {n}/108（{details}）。'
        'GRPO-100 与上方 Distill-100 共用原 100 题；GRPO-K 与 Distill-K 共用医疗/追踪/代码各 2,000 题、数学 1,488 题，且包含这 100 题。'
        '两种方法都从原始 Base 开始；千级蒸馏使用 Astra high 的可检查步骤示范并过滤错误答案，原 400 条示范直接复用。'
        'GRPO 只按训练题最终答案给 0/1 奖励，8 回答/题、固定 1 轮；100 档 4 题/步（25 步），K 档 8 题/步（250/186 步）。'
        '蒸馏固定 5 轮、batch 8（K 档 1,250/930 步）。这是相同题目数量下的方法比较，训练 token、优化步数与算力不相等，成本另记。'
        '隐藏验证/测试不参与训练、调参或选优；评测沿用协议 2，并复用上方 12 个同协议 Base。\n\n'
        '千级数据来源：医疗为公开 MedCalc 规则类训练题，覆盖的计算器范围比原 100 题更广；数学为原两个领域的全部难度训练题；'
        '追踪独立生成；代码为原 100 道可见题加 1,900 道独立生成、经执行求解并按结构去重的程序。'
        '因此跨数据量的变化同时包含题目覆盖范围变化；旧 Ours 与这些行仍须遵守上文口径限制。'
        '明细与逐种子结果见 [新增 baseline 结果](../../outputs/engineer/grpo_main/RESULTS.md)，'
        '设置与恢复方法见 [操作说明](../../outputs/engineer/grpo_main/OPERATIONS.md)。\n'+end)
    if begin in updated:
        a=updated.index(begin);b=updated.index(end,a)+len(end);updated=updated[:a]+note+updated[b:]
    else:
        i=updated.index('主表口径',updated.index('### Results'));updated=updated[:i]+note+'\n\n'+updated[i:]
    if updated!=original and sha(DOC)==fingerprint:
        tmp=DOC.with_name(DOC.name+f'.grpo.{os.getpid()}.tmp');tmp.write_text(updated);os.replace(tmp,DOC)


if __name__=='__main__': print(collect()['completed_seed_evaluations'])
