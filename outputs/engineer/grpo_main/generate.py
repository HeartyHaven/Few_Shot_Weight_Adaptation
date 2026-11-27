"""Teacher demonstrations on frozen nested K datasets; resumable and bounded."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time

from common import HERE, ROOT, TASKS, TEACHER, atomic_json, read_json, sha, progress
sys.path.insert(0, str(ROOT))
from tdwa import grade as G

LOCK = threading.RLock()
STOP = threading.Event()
TEACHER = "gpt-6.2-astra"
MAX_CALLS = 2400
MAX_TRANSPORT_FAILURES = 700
MAX_ATTEMPTS = 4
WORKERS = 4
MAX_TOTAL_TOKENS = 12_000_000


def client():
    from openai import OpenAI
    path = ROOT.parent / '.secrets/cliproxy.env'
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        parts = shlex.split(v, comments=True)
        values[k] = parts[0] if parts else ''
    return OpenAI(base_url=values['CLIPROXY_BASE_URL'], api_key=values['CLIPROXY_API_KEY'],
                  timeout=300, max_retries=0)


def prompt(task, items):
    style = {
        't1': 'Identify the calculator, state its criteria, extract the patient facts, show each score and their sum. Use the provided reference explanation when available.',
        't2': 'Write the initial person-to-object assignment, then the COMPLETE assignment after EVERY swap in order. Do not use a backwards shortcut. Finally map the requested object to its option letter.',
        't4': 'Explain the mathematical method and every essential algebraic/counting step. Give a concise, checkable worked solution, not merely a restatement of the answer.',
        't5': 'Trace execution in order, showing changing variables and any loop iterations or string/list operations. Derive the exact returned Python literal.',
    }[task]
    return ('Write worked solutions for distillation into a small language model. These are authorized training examples, not evaluation questions. '
            'The gold answer is provided to check your work; do not mention it was provided. '
            'Use English, concise steps, preferably under 650 tokens per solution (up to 1400 if necessary). '
            'Each solution MUST end with exactly one line "Final answer: ...", matching the provided gold answer. '
            + style + '\nReturn ONLY a JSON array of objects with exactly the fields "id" and "solution". '
            'Preserve each id. No tools, external retrieval or additional problems.\n\n' +
            json.dumps([{'id': r['id'], 'question': G.format_prompt(r), 'gold_answer': r['answer'],
                         **({'reference_explanation': r['explanation']} if r.get('explanation') else {})}
                        for r in items], ensure_ascii=False))


def parse(text):
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end < start:
        raise ValueError('No JSON array')
    result = json.loads(text[start:end + 1])
    if not isinstance(result, list):
        raise ValueError('Response is not a list')
    return result


def main():
    from transformers import AutoTokenizer
    from common import model_path
    data=HERE/'teacher_data';data.mkdir(exist_ok=True)
    lock=(data/'generator.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    api=client()
    tok=AutoTokenizer.from_pretrained(model_path('Qwen3-1.7B'),local_files_only=True)
    stats=read_json(data/'usage.json',{'calls':0,'prompt_tokens':0,'completion_tokens':0,'errors':0,
        'charged_tokens':0,'unknown_usage_calls':0,'reused_examples':0})
    ledger=read_json(data/'attempts.json',{})
    all_items={task:[json.loads(x) for x in (HERE/'data'/f'{task}.jsonl').read_text().splitlines()] for task in TASKS}
    fingerprints={task:sha(HERE/'data'/f'{task}.jsonl') for task in TASKS}
    calls_dir=data/'calls';calls_dir.mkdir(exist_ok=True)
    accepted={task:set() for task in TASKS}
    active_reserve=0
    cooldown_until=max(read_json(data/'heartbeat.json',{}).get('retry_after',0),read_json(data/'INCOMPLETE.json',{}).get('cooldown_until',0))
    active_requests=0
    stats.setdefault('concurrency_limit',2)
    stats.setdefault("generation_calls",stats["calls"]-stats.get("errors",0))
    stats.setdefault("transport_failures",stats.get("errors",0))
    atomic_json(data/'limits.json',{'teacher':TEACHER,'effort':'high','workers':WORKERS,
        'adaptive_concurrency_initial':2,'adaptive_concurrency_min':1,'max_generation_calls':MAX_CALLS,'max_transport_failures':MAX_TRANSPORT_FAILURES,'max_charged_tokens':MAX_TOTAL_TOKENS,'max_attempts_per_example':MAX_ATTEMPTS,
        'new_examples':sum(len(x)-100 for x in all_items.values()),'reused_examples':400})

    def checked(item,solution):
        return (isinstance(solution,str) and len(solution.split())>=20 and 'final answer:' in solution.lower()
            and G.grade(item,solution)['correct']
            and item['prompt_tokens']+len(tok.encode(solution,add_special_tokens=False))+16<=8192)

    def finalize(task):
        if len(accepted[task])!=len(all_items[task]): return
        if (data/f'{task}.ready.json').exists(): return
        rows=[read_json(data/task/f"{r['id']}.json") for r in all_items[task]]
        path=data/f'{task}.jsonl';tmp=path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));os.replace(tmp,path)
        atomic_json(data/f'{task}.ready.json',{'n_examples':len(rows),'sha256':sha(path),'teacher':TEACHER,
            'effort':'high','source':f'outputs/engineer/grpo_main/data/{task}.jsonl',
            'source_sha256':fingerprints[task],'paired_baseline':'grpo_k','original_visible_reused':100,
            'generated_questions':(1900 if task in ('t2','t5') else 0),'time':time.time()})

    for task,items in all_items.items():
        (data/task).mkdir(exist_ok=True)
        prior=ROOT/'outputs/engineer/distill_main/data'/f'{task}.jsonl'
        visible={r['id']:r for r in (json.loads(x) for x in prior.read_text().splitlines())}
        for it in items:
            path=data/task/f"{it['id']}.json"
            rec=read_json(path)
            if rec and rec.get('input_sha256')==fingerprints[task] and checked(it,rec.get('solution')):
                accepted[task].add(it['id']);continue
            old=visible.get(it['id'])
            if old and old['question']==it['question'] and str(old['answer'])==str(it['answer']) and checked(it,old['solution']):
                atomic_json(path,{**it,'solution':old['solution'],'teacher':TEACHER,'effort':'high',
                    'input_sha256':fingerprints[task],'reused_from':str(prior.relative_to(ROOT)),
                    'reused_source_sha256':sha(prior),'source':'visible_dev_reused','kept':True,
                    'grade':G.grade(it,old['solution'])})
                accepted[task].add(it['id'])
        finalize(task)
    stats['reused_examples']=400
    atomic_json(data/'usage.json',stats)
    progress(data/'heartbeat.json',phase='generating',accepted={t:len(x) for t,x in accepted.items()})

    def work(task,items):
        nonlocal active_reserve, cooldown_until, active_requests
        if STOP.is_set() or (HERE/'STOP').exists() or (HERE/'PAUSE').exists(): return
        request=prompt(task,items)
        prompt_estimate=len(tok.encode(request,add_special_tokens=False))+64
        reservation=prompt_estimate+16000
        while True:
            if STOP.is_set() or (HERE/'STOP').exists() or (HERE/'PAUSE').exists(): return
            with LOCK:
                if stats['generation_calls']>=MAX_CALLS or stats['transport_failures']>=MAX_TRANSPORT_FAILURES or stats['charged_tokens']+active_reserve+reservation>MAX_TOTAL_TOKENS:
                    STOP.set();return
                if time.time()>=cooldown_until and active_requests<stats['concurrency_limit']:
                    active_requests+=1
                    stats['calls']+=1;stats['generation_calls']+=1;call_id=stats['calls'];active_reserve+=reservation
                    for it in items:
                        key=task+':'+it['id'];ledger[key]=ledger.get(key,0)+1
                    attempt=max(ledger[task+':'+it['id']] for it in items)
                    atomic_json(data/'usage.json',stats);atomic_json(data/'attempts.json',ledger)
                    break
                progress(data/'heartbeat.json',phase='api_cooldown' if time.time()<cooldown_until else 'generating',
                    retry_after=cooldown_until,active_requests=active_requests,concurrency_limit=stats['concurrency_limit'],
                    accepted={t:len(x) for t,x in accepted.items()})
            time.sleep(5)
        started=time.time()
        record={'task':task,'ids':[r['id'] for r in items],'attempt':attempt,'teacher':TEACHER,
            'started':started,'request':request,'reserved_tokens':reservation}
        # A durable request record makes interrupted API calls visible without silently duplicating them.
        atomic_json(calls_dir/f'{call_id:04d}.json',record)
        charged=reservation
        try:
            response=api.chat.completions.create(model=TEACHER,reasoning_effort='high',max_tokens=16000,
                messages=[{'role':'user','content':request}])
            body=response.choices[0].message.content or ''
            u=response.usage
            record.update(response=body,finish_reason=response.choices[0].finish_reason,
                returned_model=response.model,usage=u.model_dump() if u else {})
            if u:
                charged=(u.prompt_tokens or 0)+(u.completion_tokens or 0)
                with LOCK:
                    stats['prompt_tokens']+=u.prompt_tokens or 0;stats['completion_tokens']+=u.completion_tokens or 0
            else:
                with LOCK: stats['unknown_usage_calls']+=1
            got={str(r['id']):r['solution'] for r in parse(body)
                if isinstance(r,dict) and 'id' in r and isinstance(r.get('solution'),str)}
            kept=0
            for it in items:
                solution=got.get(str(it['id']),'')
                if not checked(it,solution): continue
                atomic_json(data/task/f"{it['id']}.json",{**it,'solution':solution,'teacher':TEACHER,'effort':'high',
                    'input_sha256':fingerprints[task],'call_id':call_id,'source':'frozen_K_training',
                    'kept':True,'grade':G.grade(it,solution)})
                with LOCK: accepted[task].add(it['id'])
                kept+=1
            record['kept']=kept
        except Exception as e:
            record['error_type']=type(e).__name__;record['status_code']=getattr(e,'status_code',None)
            # An explicit proxy auth-unavailable rejection never reached a model. Preserve the call,
            # but it is a transport failure rather than a generated demonstration attempt or token charge.
            body=str(getattr(e,'body','')).lower()
            unavailable=record['status_code'] in (429,503) and ('auth_unavailable' in body or 'no auth' in body)
            record['proxy_auth_unavailable']=unavailable
            with LOCK:
                stats['errors']+=1
                if unavailable:
                    charged=0;stats['transport_failures']+=1;stats['generation_calls']-=1
                    stats['concurrency_limit']=max(1,stats['concurrency_limit']//2)
                    for it in items:
                        key=task+':'+it['id'];ledger[key]=max(0,ledger.get(key,0)-1)
                    atomic_json(data/'attempts.json',ledger)
                cooldown_until=max(cooldown_until,time.time()+(300 if unavailable else 60))
            if record['status_code'] in (401,403): STOP.set()
            # Do not exhaust all rows while the sole proxy account is cooling down.
            if record['status_code'] in (401,403):
                atomic_json(data/'AUTH_ERROR.json',{'time':time.time(),'status':record['status_code']})

        finally:
            record['seconds']=round(time.time()-started,2);record['charged_tokens']=charged;record['finished']=time.time()
            atomic_json(calls_dir/f'{call_id:04d}.json',record)
            with LOCK:
                active_requests-=1;active_reserve-=reservation;stats['charged_tokens']+=charged
                atomic_json(data/'usage.json',stats)
                progress(data/'heartbeat.json',phase='generating',calls=stats['calls'],last_call=call_id,
                    accepted={t:len(x) for t,x in accepted.items()})
                finalize(task)
            print(json.dumps({k:record[k] for k in ['task','attempt','kept','error_type','status_code','seconds'] if k in record}),flush=True)

    # Charge interrupted requests conservatively. Their persisted attempts already count against the per-row limit.
    unknown=[p for p in calls_dir.glob('*.json') if not read_json(p,{}).get('finished')]
    for path in unknown:
        rec=read_json(path);charge=rec.get('reserved_tokens',16000)
        stats['charged_tokens']+=charge;stats['unknown_usage_calls']+=1
        rec.update(finished=time.time(),charged_tokens=charge,error_type='InterruptedUnknownUsage')
        atomic_json(path,rec)
    atomic_json(data/'usage.json',stats)
    for round_no in range(MAX_ATTEMPTS):
        jobs=[]
        for task in TASKS:
            missing=[it for it in all_items[task] if it['id'] not in accepted[task]
                and ledger.get(task+':'+it['id'],0)<MAX_ATTEMPTS]
            fresh=[it for it in missing if ledger.get(task+':'+it['id'],0)==0]
            retry=[it for it in missing if ledger.get(task+':'+it['id'],0)>0]
            jobs.extend((task,fresh[i:i+5]) for i in range(0,len(fresh),5))
            jobs.extend((task,retry[i:i+2]) for i in range(0,len(retry),2))
        if not jobs or STOP.is_set() or (HERE/'STOP').exists() or (HERE/'PAUSE').exists(): break
        # Finish each task's data in turn so GPU distillation can begin while other solutions are generated.
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda j:work(*j),jobs))
        if all(len(accepted[t])==len(all_items[t]) for t in TASKS): break
        time.sleep(15)
    done=all(len(accepted[t])==len(all_items[t]) for t in TASKS)
    for task in TASKS: finalize(task)
    if done: (data/'INCOMPLETE.json').unlink(missing_ok=True)
    progress(data/'heartbeat.json',done=done,phase='complete' if done else 'incomplete',calls=stats['calls'],
        accepted={t:len(x) for t,x in accepted.items()})
    if not done:
        atomic_json(data/'INCOMPLETE.json',{'time':time.time(),'calls':stats['calls'],
            'accepted':{t:len(x) for t,x in accepted.items()},'charged_tokens':stats['charged_tokens'],
            'auth_or_budget_stop':stats['generation_calls']>=MAX_CALLS or stats['charged_tokens']>=MAX_TOTAL_TOKENS or stats['transport_failures']>=MAX_TRANSPORT_FAILURES,
            'cooldown_until':cooldown_until,'message':'No incomplete teacher dataset is published; independent GRPO jobs may continue.'})
        raise SystemExit(2)


if __name__=='__main__': main()
