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
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    from tdwa import grade as G

    name = run_name(a.task, a.model, a.seed)
    out = HERE / 'runs' / name
    out.mkdir(parents=True, exist_ok=True)
    beat = out / 'train_progress.json'
    started = time.time()
    progress(beat, phase='loading_data')
    ready = read_json(HERE / 'data' / f'{a.task}.ready.json')
    data_path = HERE / 'data' / f'{a.task}.jsonl'
    if not ready or sha(data_path) != ready['sha256']:
        raise RuntimeError('Teacher dataset is not ready or its fingerprint changed')
    rows = [json.loads(l) for l in data_path.read_text().splitlines()]
    if len(rows) != 100 or any(not G.grade(r, r['solution'])['correct'] for r in rows):
        raise RuntimeError('Expected exactly 100 correctly graded teacher examples')
    torch.set_num_threads(4)
    torch.manual_seed(a.seed); random.seed(a.seed)
    # Enforce the scheduler reservation even if PyTorch caches blocks from many sequence lengths.
    reserve={'Qwen3-1.7B':34,'Qwen3-4B':48,'Qwen3-14B':76}[a.model]
    torch.cuda.set_per_process_memory_fraction((reserve-4)*2**30/torch.cuda.get_device_properties(0).total_memory)
    src = model_path(a.model)
    tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    examples = []
    for row in rows:
        messages = [{'role': 'user', 'content': G.format_prompt(row)}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': row['solution']}],
                                       tokenize=False, enable_thinking=False)
        if not full.startswith(prompt):
            # Qwen's training template may omit the disabled-thinking prefix on assistant turns.
            full = prompt + row['solution'] + tok.eos_token
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        ids = tok.encode(full, add_special_tokens=False)
        if ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f'Prompt token boundary differs for training id {row["id"]}')
        if len(ids) > 8192:
            raise RuntimeError(f'Training example {row["id"]} exceeds 8192; never truncate its final answer')
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        examples.append((ids, labels, row['id']))
    sequence_stats = {'min': min(len(x[0]) for x in examples), 'max': max(len(x[0]) for x in examples),
                      'total': sum(len(x[0]) for x in examples),
                      'reply_tokens': sum(sum(v != -100 for v in x[1]) for x in examples)}
    progress(beat, phase='loading_model', sequence_stats=sequence_stats)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16,
                attn_implementation='sdpa', local_files_only=True).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                     target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0, fused=True)
    epochs, batch = 5, 8
    total_steps = epochs * math.ceil(len(examples) / batch)
    scheduler = get_cosine_schedule_with_warmup(opt, max(1, int(.03 * total_steps)), total_steps)
    checkpoint = out / 'checkpoint.pt'
    epoch_start, step, seen_tokens = 0, 0, 0
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['data_sha256'] != ready['sha256']:
            raise RuntimeError('Resume dataset changed')
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, ck['adapter'])
        opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng_cpu']); torch.cuda.set_rng_state(ck['rng_cuda'])
        epoch_start, step, seen_tokens = ck['epoch'], ck['step'], ck['seen_tokens']
        print('Resumed after epoch', epoch_start, flush=True)
    model.train()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_start = time.time()
    for epoch in range(epoch_start, epochs):
        order = list(range(len(examples))); random.Random(a.seed * 100 + epoch).shuffle(order)
        # Small randomized buckets reduce padding, with all examples retained each epoch.
        order = [i for s in range(0,len(order),32) for i in sorted(order[s:s+32], key=lambda j:len(examples[j][0]))]
        for start in range(0, len(order), batch):
            indices = order[start:start + batch]
            reply_count = sum(sum(v != -100 for v in examples[i][1][1:]) for i in indices)
            opt.zero_grad(set_to_none=True)
            loss_total = 0.0
            for begin in range(0, len(indices), a.microbatch):
                micro = [examples[i] for i in indices[begin:begin+a.microbatch]]
                length = max(len(x[0]) for x in micro)
                ids = torch.full((len(micro), length), pad, dtype=torch.long, device='cuda')
                labels = torch.full_like(ids, -100); mask = torch.zeros_like(ids)
                for i, (tokens, lab, _) in enumerate(micro):
                    ids[i,:len(tokens)] = torch.tensor(tokens,device='cuda')
                    labels[i,:len(tokens)] = torch.tensor(lab,device='cuda')
                    mask[i,:len(tokens)] = 1
                n_reply = int((labels[:,1:] != -100).sum())
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                (loss * (n_reply / reply_count)).backward()
                loss_total += float(loss.detach()) * n_reply / reply_count
                seen_tokens += int(mask.sum())
                del loss, ids, labels, mask
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite gradients')
            opt.step(); scheduler.step(); step += 1
            rec = {'epoch':epoch+1,'step':step,'of':total_steps,'loss':loss_total,'lr':scheduler.get_last_lr()[0],
                   'seen_tokens':seen_tokens,'seconds':round(time.time()-train_start,2),
                   'peak_mem_gib':round(torch.cuda.max_memory_allocated()/2**30,2),
                   'reserved_gib':round(torch.cuda.memory_reserved()/2**30,2)}
            with (out/'train_log.jsonl').open('a') as f:
                f.write(json.dumps(rec)+'\n')
            progress(beat, phase='training', **rec)
            print(json.dumps(rec),flush=True)
        from peft import get_peft_model_state_dict
        ck = {'adapter': {k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()},
              'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
              'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),
              'epoch':epoch+1,'step':step,'seen_tokens':seen_tokens,'data_sha256':ready['sha256']}
        tmp=checkpoint.with_suffix('.tmp'); torch.save(ck,tmp); os.replace(tmp,checkpoint)
        torch.cuda.empty_cache()
    adapter = out/'adapter'; model.save_pretrained(adapter, safe_serialization=True)
    atomic_json(out/'train_done.json', {'name':name,'task':a.task,'student':'Qwen/'+a.model,'teacher':TEACHER,
        'seed':a.seed,'n_examples':100,'synthetic_examples':0,'epochs':epochs,'effective_batch':batch,
        'microbatch':a.microbatch,'rank':16,'alpha':32,'lr':2e-4,'dtype':'bfloat16','max_length':8192,
        'data_sha256':ready['sha256'],'model_snapshot':str(src),'steps':step,'expected_steps':total_steps,
        'train_tokens':seen_tokens,'forward_equivalent_tokens':3*seen_tokens,'wall_seconds':time.time()-started,
        'peak_mem_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'memory_reservation_gib':reserve,'sequence_stats':sequence_stats,
        'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'worker_sha256':sha(Path(__file__)),
        'worker_loaded_sha256':LOADED_SOURCE_SHA256,
        'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'finished':time.time()})
    progress(beat,phase='complete',step=step)


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
    name = f'B3_BASE__{a.task}__{a.model}' if a.base else run_name(a.task,a.model,a.seed)
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
    raw=ROOT/'_secret'/'eval_distill_main'/name/f'protocol{EVAL_PROTOCOL}'
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
              'baseline':'Base-matched' if a.base else 'B3-100','teacher':None if a.base else TEACHER,
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
    ap.add_argument('--microbatch',type=int,default=2)
    ap.add_argument('--mem',type=float,default=.4)
    ap.add_argument('--base',action='store_true')
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
