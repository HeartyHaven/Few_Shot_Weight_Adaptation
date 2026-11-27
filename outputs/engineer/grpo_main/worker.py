"""Independent training/evaluation workers. Never tune using hidden test scores."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from common import HERE, ROOT, TEACHER, EVAL_PROTOCOL, atomic_json, model_path, progress, read_json, run_name, sha
sys.path.insert(0, str(ROOT))
LOADED_SOURCE_SHA256 = sha(Path(__file__))


def train(a):
    if a.baseline=='distill_k':
        from distill_training import train as impl
    else:
        from grpo_training import train as impl
    impl(a)

def extended_retention(st, out):
    from datasets import load_dataset
    from tdwa.evaluate import ci
    from tdwa.retention_vllm import num
    rng=random.Random(1212); by={}
    for r in load_dataset('cais/mmlu','all')['test']:
        by.setdefault(r['subject'],[]).append(r)
    items=[x for s in sorted(by) for x in rng.sample(by[s],min(20,len(by[s])))]
    users=[r['question']+'\n'+'\n'.join(f"({'ABCD'[i]}) {c}" for i,c in enumerate(r['choices']))+
           '\nAnswer with the option letter in parentheses.' for r in items]
    pred=st.forced(users,[4]*len(items))
    result={'mmlu':ci([p==f"({'ABCD'[r['answer']]})" for p,r in zip(pred,items)])}
    atomic_json(out/'mmlu.json',[{'subject':r['subject'],'gold':'ABCD'[r['answer']],'pred':p} for p,r in zip(pred,items)])
    rows=rng.sample(list(load_dataset('openai/gsm8k','main')['test']),500)
    answers=st.generate([r['question']+"\n\nWork step by step, then finish with a line 'Final answer: <number>'." for r in rows],512)
    gold=[float(r['answer'].split('####')[-1].strip().replace(',','')) for r in rows]
    good=[num(o) is not None and abs(num(o)-g)<1e-6 for o,g in zip(answers,gold)]
    result['gsm8k']=ci(good)
    atomic_json(out/'gsm8k.json',[{'q':r['question'],'gold':g,'out':o,'ok':ok} for r,g,o,ok in zip(rows,gold,answers,good)])
    return result


def evaluate(a):
    import functools
    import tdwa.evaluate_vllm as engine
    from tdwa.evaluate_vllm import VStudent, evaluate as evaluate_task
    name = f'GRPO_BASE__{a.task}__{a.model}' if a.base else run_name(a.task,a.model,a.seed,a.baseline)
    target=HERE/'evaluations'/f'{name}.json'
    beat=HERE/'evaluation_progress'/f'{name}.json'
    progress(beat,phase='loading_model')
    started=time.time()
    adapter=None if a.base else HERE/'runs'/name/'adapter'
    if not a.base:
        done=read_json(HERE/'runs'/name/'train_done.json')
        if not done or done['adapter_sha256'] != sha(adapter/'adapter_model.safetensors'):
            raise RuntimeError('Adapter incomplete or fingerprint changed')
    # Keep evaluator outputs physically separate from the teacher-visible data.
    raw=ROOT/'_secret'/'eval_grpo_main'/name/f'protocol{EVAL_PROTOCOL}'
    raw.mkdir(parents=True,exist_ok=True)
    # Explicit KV allocation bypasses vLLM's global-free-memory delta estimate, which is invalid
    # when another colocated worker finishes or grows its allocator during startup.
    kv_gib=24 if a.model=='Qwen3-14B' else 16
    engine.LLM=functools.partial(engine.LLM,kv_cache_memory_bytes=kv_gib*2**30,
                                max_num_seqs=64,max_num_batched_tokens=8192)
    st=VStudent('Qwen/'+a.model,str(model_path(a.model)),dtype='bfloat16',mem=a.mem,
                max_len=8192,lora=str(adapter) if adapter else None)
    progress(beat,phase='hidden_test')
    results=evaluate_task(st,a.task,str(raw))
    progress(beat,phase='extended_retention')
    results['retention'].update(extended_retention(st,raw))
    files=sorted((ROOT/'_secret'/a.task).glob('test*.jsonl'))
    metadata={'name':name,'task':a.task,'student':'Qwen/'+a.model,'seed':None if a.base else a.seed,
              'baseline':'Base-matched' if a.base else a.baseline,'teacher':'gpt-6.2-astra' if a.baseline=='distill_k' and not a.base else None,
              'dtype':'bfloat16','engine':'vllm','max_new_tokens':2048,'gsm8k_max_new_tokens':512,
              'enable_thinking':False,'test_fingerprints':{str(p.relative_to(ROOT)):sha(p) for p in files},
              'grader_sha256':sha(ROOT/'tdwa/grade.py'),'evaluator_sha256':sha(ROOT/'tdwa/evaluate_vllm.py'),
              'worker_sha256':sha(Path(__file__)),'worker_loaded_sha256':LOADED_SOURCE_SHA256,
              'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'evaluation_protocol':EVAL_PROTOCOL,'kv_cache_gib':kv_gib,'max_num_seqs':64,'max_num_batched_tokens':8192,
              'wall_seconds':round(time.time()-started,2),'finished':time.time()}
    if not a.base:
        metadata.update(data_sha256=done['data_sha256'],adapter_sha256=done['adapter_sha256'])
    if target.exists():
        old=read_json(target)
        if old: atomic_json(HERE/'evaluation_history'/f'{name}.{int(time.time())}.json',old)
    atomic_json(target,{**metadata,'results':results})
    progress(beat,phase='complete')
    print('EVALUATION_COMPLETE',name,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['train','evaluate'])
    ap.add_argument('--task',required=True,choices=['t1','t2','t4','t5'])
    ap.add_argument('--model',required=True,choices=['Qwen3-1.7B','Qwen3-4B','Qwen3-14B'])
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--baseline',choices=['grpo_k','grpo_100','distill_k'],default='grpo_k')
    ap.add_argument('--microbatch',type=int,default=2)
    ap.add_argument('--mem',type=float,default=.4)
    ap.add_argument('--base',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--max-steps',type=int,default=3)
    a=ap.parse_args()
    try:
        train(a) if a.command=='train' else evaluate(a)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if 'out of memory' in str(exc).lower():
            raise SystemExit(42)
        raise


if __name__=='__main__':
    main()
