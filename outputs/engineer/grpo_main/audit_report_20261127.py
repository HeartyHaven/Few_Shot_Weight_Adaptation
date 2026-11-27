"""Offline audit of completed artifacts for the requested report; never launches workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import HERE, ROOT, PYTHON, atomic_json, sha

CHILD = r'''
import sys,json,statistics,math
from pathlib import Path
folder=Path(sys.argv[1]);sys.path.insert(0,str(folder))
from common import HERE,ROOT,MODELS,TASKS,SEEDS,read_json,sha,run_name
from collect import valid,main_metric,RET_KEYS
from watchdog import completed
is_grpo=folder.name=='grpo_main'
methods=['grpo_100','grpo_k','distill_k'] if is_grpo else ['distill_100']
state=read_json(HERE/'watchdog_state.json')
protocol_keys=['dtype','engine','max_new_tokens','gsm8k_max_new_tokens','enable_thinking',
 'test_fingerprints','grader_sha256','evaluator_sha256','evaluation_protocol','kv_cache_gib',
 'max_num_seqs','max_num_batched_tokens']
bases={};groups={};sources={}
for model in MODELS:
 for task in TASKS:
  prefix='GRPO_BASE' if is_grpo else 'B3_BASE'
  path=HERE/'evaluations'/f'{prefix}__{task}__{model}.json';r=read_json(path)
  assert valid(r,task,model,baseline='Base-matched') if is_grpo else valid(r,task,model)
  bases[model,task]=r;sources[str(path.relative_to(ROOT))]=sha(path)
for method in methods:
 group={'completed_train':0,'completed_evaluations':0,'cells':{},'records':[],'matched_bases':12}
 for model in MODELS:
  group['cells'][model]={}
  for task in TASKS:
   base=bases[model,task];records=[]
   for seed in SEEDS:
    name=run_name(task,model,seed,method) if is_grpo else run_name(task,model,seed)
    job=state['jobs']['train:'+name]
    train_ok=bool(completed(job));group['completed_train']+=int(train_ok)
    path=HERE/'evaluations'/f'{name}.json'
    if not path.exists():continue
    r=read_json(path)
    assert valid(r,task,model,base,method) if is_grpo else valid(r,task,model,base)
    assert train_ok,(name,'evaluation without completed training')
    donepath=HERE/'runs'/name/'train_done.json';done=read_json(donepath)
    assert r['name']==name and r['seed']==seed
    assert r['adapter_sha256']==done['adapter_sha256']==sha(HERE/'runs'/name/'adapter/adapter_model.safetensors')
    assert r['data_sha256']==done['data_sha256']
    assert all(r.get(k)==base.get(k) for k in protocol_keys)
    metric=r['results']['test']['taught'] if task=='t1' else r['results']['test_bbh' if task=='t2' else 'test']['free']
    assert metric['n']=={'t1':160,'t2':750,'t4':277,'t5':550}[task]
    ret=r['results']['retention'];br=base['results']['retention']
    for key,n in [('arc_challenge',300),('hellaswag',300),('mmlu',1140),('gsm8k',500),('arithmetic',200)]:
     assert ret[key]['n']==n and 0<=ret[key]['acc']<=1
    assert math.isfinite(ret['ppl_generic']) and ret['ppl_generic']>0
    record={'name':name,'model':model,'task':task,'seed':seed,'n_examples':done['n_examples'],
     'steps':done['steps'],'expected_steps':done['expected_steps'],'base_score':main_metric(base),
     'score':main_metric(r),'task_delta_pp':main_metric(r)-main_metric(base),
     'retention_deltas':{k:(ret[k]['acc']-br[k]['acc'])*100 for k in RET_KEYS},
     'retention_mean_delta_pp':statistics.mean((ret[k]['acc']-br[k]['acc'])*100 for k in RET_KEYS),
     'ppl_relative_change_pct':(ret['ppl_generic']/br['ppl_generic']-1)*100,
     'ppl':ret['ppl_generic'],'base_ppl':br['ppl_generic'],
     'train_finished':done['finished'],'eval_finished':r['finished'],
     'eval_path':str(path.relative_to(ROOT)),'train_path':str(donepath.relative_to(ROOT))}
    group['records'].append(record);records.append(record)
    sources[str(path.relative_to(ROOT))]=sha(path);sources[str(donepath.relative_to(ROOT))]=sha(donepath)
   scores=[r['score'] for r in records]
   group['cells'][model][task]={'n':len(records),'seeds':[r['seed'] for r in records],
    'base':main_metric(base),'mean':statistics.mean(scores) if scores else None,
    'sd':statistics.stdev(scores) if len(scores)>1 else None,
    'delta_pp':statistics.mean(r['task_delta_pp'] for r in records) if records else None,
    'retention_delta_pp':statistics.mean(r['retention_mean_delta_pp'] for r in records) if records else None}
   group['completed_evaluations']+=len(records)
 group['model_retention']={}
 for model in MODELS:
  records=[r for r in group['records'] if r['model']==model]
  if len(records)==12:
   group['model_retention'][model]={**{k:statistics.mean(r['retention_deltas'][k] for r in records) for k in RET_KEYS},
    'mean_pp':statistics.mean(r['retention_mean_delta_pp'] for r in records),
    'ppl_relative_change_pct':statistics.mean(r['ppl_relative_change_pct'] for r in records)}
 groups[method]=group
print(json.dumps({'groups':groups,'source_sha256':sources},ensure_ascii=False))
'''


def main():
    import psutil
    folders=[ROOT/'outputs/engineer/distill_main',HERE]
    assert all((d/'STOP').exists() and (d/'PAUSE').exists() for d in folders)
    scripts={str(d/f) for d in folders for f in ['worker.py','generate.py','watchdog.py','supervisor.py']}
    live=[p.pid for p in psutil.process_iter(['pid','cmdline']) if any(a in scripts for a in (p.info['cmdline'] or []))]
    assert not live,live
    all_groups={};sources={}
    for folder in folders:
        payload=json.loads(subprocess.check_output([str(PYTHON),'-c',CHILD,str(folder)],cwd=ROOT,text=True))
        all_groups.update(payload['groups']);sources.update(payload['source_sha256'])
    # Verify exact 100-question pairing and the identity of the twelve reused matched Bases.
    for task in ['t1','t2','t4','t5']:
        d=ROOT/'outputs/engineer/distill_main/data'/f'{task}.jsonl';g=HERE/'data_100'/f'{task}.jsonl'
        dd={r['id']:r for r in map(json.loads,d.read_text().splitlines())}
        gg={r['id']:r for r in map(json.loads,g.read_text().splitlines())}
        assert dd.keys()==gg.keys() and len(dd)==100
        assert all(all(dd[k][f]==gg[k][f] for f in ['question','answer','task']) for k in dd)
    for model in ['Qwen3-1.7B','Qwen3-4B','Qwen3-14B']:
        for task in ['t1','t2','t4','t5']:
            g=json.loads((HERE/'evaluations'/f'GRPO_BASE__{task}__{model}.json').read_text())
            d=json.loads((ROOT/g['reused_from']).read_text())
            assert g['results']==d['results'] and sha(ROOT/g['reused_from'])==g['reused_source_sha256']
    counts={k:{'train':g['completed_train'],'evaluation':g['completed_evaluations']} for k,g in all_groups.items()}
    assert counts=={'distill_100':{'train':36,'evaluation':36},'grpo_100':{'train':36,'evaluation':36},
                   'grpo_k':{'train':1,'evaluation':1},'distill_k':{'train':0,'evaluation':0}},counts
    total=sum(v['evaluation'] for v in counts.values())
    r={'audited_at':time.time(),'passed':True,'counts':counts,'total_completed_seed_results':total,
       'complete_three_seed_cells':sum(e['n']==3 for g in all_groups.values() for model in g['cells'].values() for e in model.values()),
       'single_seed_cells':sum(e['n']==1 for g in all_groups.values() for model in g['cells'].values() for e in model.values()),
       'exact_100_question_pairing':True,'same_12_matched_bases':True,'stop_flags_retained':True,
       'live_baseline_processes':live,'groups':all_groups,'source_sha256':sources}
    atomic_json(HERE/'report_audit_20261127.json',r)
    print(json.dumps({'passed':True,'counts':counts,'total':total,'three_seed_cells':r['complete_three_seed_cells']},ensure_ascii=False))


if __name__=='__main__':main()
