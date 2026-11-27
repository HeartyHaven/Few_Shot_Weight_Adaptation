"""Persistent, GPU-restricted GRPO scheduler with recovery and bounded retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
from common import HERE, ROOT, PYTHON, TASKS, MODELS, GPUS, SEEDS, OWNER, BOARD_ID
from common import atomic_json, environment, event, read_json, run_name, sha, BASELINES, BASELINE_INFO, dataset_dir
import math
from collect import collect, valid
from resource_policy import (POLICY_VERSION, ENV_PROFILE, ENV_RESERVE, ENV_CAP,
                             profiles, recovered_profile, choose_profile, committed_gib)

STATE=HERE/'watchdog_state.json'
CHILDREN={}


def alive(rec):
    try:
        p=psutil.Process(rec['pid'])
        return abs(p.create_time()-rec['created'])<.1 and p.status()!=psutil.STATUS_ZOMBIE
    except (KeyError,psutil.Error): return False


def owned_stop(rec):
    if not alive(rec): return
    p=psutil.Process(rec['pid'])
    if not any(str(HERE) in x for x in p.cmdline()):
        raise RuntimeError('Refusing to stop a process not owned by this task')
    children=p.children(recursive=True)
    for child in children[::-1]:
        try: child.terminate()
        except psutil.Error: pass
    p.terminate()
    _, live=psutil.wait_procs(children+[p],timeout=5)
    for child in live:
        try: child.kill()
        except psutil.Error: pass


def gpu_snapshot(state):
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,power.draw',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    result={}; uuid_to_index={}
    for line in p.stdout.splitlines():
        index,uuid,total,used,util,power=[x.strip() for x in line.split(',')]
        index=int(index)
        if index in GPUS:
            result[index]={'total_gib':int(total)/1024,'used_gib':int(used)/1024,'util':int(util),
                           'power_w':float(power),'job_used_gib':{}}
            uuid_to_index[uuid]=index
    if set(result)!=set(GPUS): raise RuntimeError('An authorized GPU is missing')
    owners={}
    for j in state['jobs'].values():
        if j['status']!='running': continue
        owners[j['pid']]=j['id']
        try:
            for child in psutil.Process(j['pid']).children(recursive=True): owners[child.pid]=j['id']
        except psutil.Error: pass
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                         '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    for line in apps.stdout.splitlines():
        uuid,pid,memory=[x.strip() for x in line.split(',')]
        if uuid not in uuid_to_index or int(pid) not in owners: continue
        try: used=float(memory)/1024
        except ValueError: continue  # Unknown usage stays in the foreign/unattributed allowance.
        snapshot=result[uuid_to_index[uuid]]; job=owners[int(pid)]
        snapshot['job_used_gib'][job]=snapshot['job_used_gib'].get(job,0)+used
    for snapshot in result.values():
        snapshot['foreign_gib']=max(0,snapshot['used_gib']-sum(snapshot['job_used_gib'].values()))
    return result


def record_utilization(state,gpus):
    now=time.time()
    rows=state.setdefault('utilization_window',[])
    rows.append({'time':now,'gpus':{str(k):v['util'] for k,v in gpus.items()}})
    rows[:]=[r for r in rows if now-r['time']<=300]
    state['utilization_5min']={str(g):round(sum(r['gpus'][str(g)] for r in rows)/len(rows),1) for g in GPUS}
    state['resource_policy_version']=POLICY_VERSION
    with (HERE/'gpu_utilization.jsonl').open('a') as f:
        f.write(json.dumps({'time':now,'gpus':gpus,'running':[
            {'id':j['id'],'gpu':j['gpu'],'pid':j['pid'],'reserved_gib':j.get('reserved_gib'),
             'resource_profile':j.get('resource_profile','legacy')} for j in state['jobs'].values()
            if j['status']=='running']})+'\n')


def jobs():
    result={}
    for baseline in BASELINES:
        for seed in SEEDS:
            for task in TASKS:
                for model in MODELS:
                    name=run_name(task,model,seed,baseline)
                    for kind in ('train','evaluate'):
                        key=f'{kind}:{name}'
                        result[key]={'id':key,'kind':kind,'task':task,'model':model,'seed':seed,'base':False,
                            'baseline':baseline,'name':name,'status':'pending','attempts':0,
                            'microbatch':({'Qwen3-1.7B':8,'Qwen3-4B':4,'Qwen3-14B':2} if baseline=='distill_k'
                                          else {'Qwen3-1.7B':8,'Qwen3-4B':4,'Qwen3-14B':4})[model]}
    for task in TASKS:
        for model in MODELS:
            name=f'GRPO_BASE__{task}__{model}';key=f'evaluate:{name}'
            result[key]={'id':key,'kind':'evaluate','task':task,'model':model,'seed':0,'base':True,
                'baseline':'grpo_k','name':name,'status':'pending','attempts':0}
    return result


def output(j):
    return HERE/'runs'/j['name']/'train_done.json' if j['kind']=='train' else HERE/'evaluations'/(j['name']+'.json')


def completed(j):
    r=read_json(output(j))
    if not r: return False
    if j['kind']=='evaluate':
        return valid(r,j['task'],j['model'],baseline='Base-matched' if j['base'] else j['baseline'])
    try:
        ready=read_json(dataset_dir(j['baseline'])/(j['task']+'.ready.json'))
        if not ready: return False
        n=ready['n_examples']
        expected=5*math.ceil(n/8) if j['baseline']=='distill_k' else n//(4 if j['baseline']=='grpo_100' else 8)
        if not (r['data_sha256']==ready['sha256'] and r['steps']==r['expected_steps']==expected and
            r['seed']==j['seed'] and r['task']==j['task'] and r['student']=='Qwen/'+j['model'] and
            r['n_examples']==n and r['baseline']==j['baseline'] and
            r['adapter_sha256']==sha(HERE/'runs'/j['name']/'adapter/adapter_model.safetensors')): return False
        if j['baseline']=='distill_k': return r['epochs']==5 and r['effective_batch']==8
        return r.get('base_unchanged') is True and r.get('unique_prompts_seen')==n
    except (KeyError,FileNotFoundError): return False


def heartbeat(j):
    return (HERE/'runs'/j['name']/'train_progress.json' if j['kind']=='train'
            else HERE/'evaluation_progress'/(j['name']+'.json'))


def ready(j,state):
    if j['status'] not in ('pending','retry') or time.time()<j.get('retry_after',0): return False
    if j['attempts']>=(8 if j['kind']=='evaluate' else 4): return False
    if j['kind']=='train':
        if not (dataset_dir(j['baseline'])/(j['task']+'.ready.json')).exists(): return False
        if not read_json(HERE/'INITIAL_VALIDATION.json',{}).get('passed'): return False
    elif not j['base']:
        if not completed(state['jobs']['train:'+j['name']]): return False
    if j['kind']=='evaluate':
        evaluated=any(x['kind']=='evaluate' and x['status']=='done' for x in state['jobs'].values())
        active_eval=sum(x['kind']=='evaluate' and x['status']=='running' for x in state['jobs'].values())
        if not evaluated and active_eval>=2: return False
    return True


def resources(j):
    profile=profiles(j)[0]
    return profile['reserved_gib'],profile.get('eval_mem')


def priority(j,state):
    if j['kind']=='evaluate': tier=0
    else:
        active=[x for x in state['jobs'].values() if x['status']=='running' and x['kind']=='train']
        counts={b:sum(x['baseline']==b for x in active) for b in BASELINES}
        if j['baseline']=='grpo_100': tier=1 if counts['grpo_100']<3 else 3
        elif j['baseline']=='grpo_k': tier=1.5 if counts['grpo_k']<2 else 4
        else: tier=2
    return (tier,j['seed'],TASKS.index(j['task']),MODELS.index(j['model']))


def launch(j,gpu,profile):
    reserve,mem=profile['reserved_gib'],profile.get('eval_mem')
    if j['kind']=='train': j['microbatch']=profile['microbatch']
    cmd=[str(PYTHON),str(HERE/'worker.py'),j['kind'],'--task',j['task'],'--model',j['model'],'--seed',str(j['seed']),'--baseline',j['baseline']]
    if j['kind']=='train': cmd+=['--microbatch',str(j['microbatch'])]
    else:
        cmd+=['--mem',str(mem)]
        if j['base']: cmd+=['--base']
    logs=HERE/'logs'; logs.mkdir(exist_ok=True)
    code=HERE/'code_history'; code.mkdir(exist_ok=True)
    digest=sha(HERE/'worker.py')
    snapshot=code/f'worker_{digest}.py'
    if not snapshot.exists(): snapshot.write_bytes((HERE/'worker.py').read_bytes())
    for source in ['grpo_training.py','distill_training.py','common.py','collect.py','resource_policy.py','watchdog.py']:
        source_path=HERE/source; version=code/f'{source_path.stem}_{sha(source_path)}.py'
        if not version.exists(): version.write_bytes(source_path.read_bytes())
    path=logs/f'{j["name"]}.{j["kind"]}.attempt{j["attempts"]+1}.log'
    env=environment(gpu)
    env.update({ENV_PROFILE:profile['profile'],ENV_RESERVE:str(reserve)})
    if profile['torch_cap_gib'] is not None: env[ENV_CAP]=str(profile['torch_cap_gib'])
    else: env.pop(ENV_CAP,None)
    with path.open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN[p.pid]=p
    j.update(status='running',pid=p.pid,created=psutil.Process(p.pid).create_time(),started=time.time(),
             gpu=gpu,reserved_gib=reserve,resource_profile=profile['profile'],torch_cap_gib=profile['torch_cap_gib'],
             log=str(path),attempts=j['attempts']+1,worker_source_sha256=digest)
    event('launch',job=j['id'],gpu=gpu,pid=p.pid,attempt=j['attempts'],resources=profile)


def reap(state):
    now=time.time()
    for j in state['jobs'].values():
        if j['status']=='running':
            p=CHILDREN.get(j.get('pid'))
            if p is not None: p.poll()
            if alive(j):
                b=read_json(heartbeat(j),{})
                last=max(j['started'],b.get('time',0))
                if now-last>2700 or now-j['started']>24*3600:
                    owned_stop(j); event('stalled_job',job=j['id'],last_progress_seconds=now-last)
                else: continue
            j['exit_code']=p.poll() if p is not None else None
            j['last_wall_seconds']=now-j['started']
            if completed(j):
                j['status']='done'; j['finished']=now; event('complete',job=j['id'])
                continue
            log=Path(j.get('log',''))
            tail=log.read_text(errors='replace')[-5000:] if log.is_file() else ''
            if j['kind']=='train' and ('out of memory' in tail.lower() or j['exit_code']==42):
                j['microbatch']=max(1,j['microbatch']//2)
                j['oom']=True
            elif j['kind']=='evaluate' and ('out of memory' in tail.lower() or j['exit_code']==42):
                j['eval_mem']=max(.36 if j['model']=='Qwen3-14B' else .24,resources(j)[1]-.04)
                j['oom']=True
            j['status']='retry' if j['attempts']<(8 if j['kind']=='evaluate' else 4) else 'failed'
            j['retry_after']=now+min(300,30*j['attempts'])
            j['failure_tail']=tail[-1800:]
            event('failure',job=j['id'],attempt=j['attempts'],exit_code=j['exit_code'],status=j['status'])
        elif completed(j):
            j['status']='done'
        elif j['status']=='done':
            j['status']='retry'; j['retry_after']=now
            event('artifact_protocol_changed',job=j['id'])


def adopt_generator(state):
    for p in psutil.process_iter(['pid','cmdline']):
        if str(HERE/'generate.py') in (p.info['cmdline'] or []):
            state['generator']={'pid':p.pid,'created':p.create_time(),'started':p.create_time()}
            event('generator_adopted',pid=p.pid)
            break


def monitor_generator(state,paused):
    needed=any(not (HERE/'teacher_data'/f'{t}.ready.json').exists() for t in TASKS)
    rec=state.get('generator',{})
    if rec and alive(rec):
        beat=read_json(HERE/'teacher_data/heartbeat.json',{})
        last=max(rec['started'],beat.get('time',0))
        if time.time()-last>1200:
            owned_stop(rec);event('generator_stalled',seconds=time.time()-last)
        else: return
    if rec:
        child=CHILDREN.get(rec['pid'])
        if child: child.poll()
        state['last_generator']=rec;state.pop('generator',None)
        event('generator_exit',complete=not needed)
    if not needed or paused: return
    incomplete=read_json(HERE/'teacher_data/INCOMPLETE.json',{})
    if incomplete.get('auth_or_budget_stop') or (HERE/'teacher_data/AUTH_ERROR.json').exists():
        state['generator_attention']='Teacher authentication or bounded token/call budget stopped generation';return
    if state.get('generator_restarts',0)>=4:
        state['generator_attention']='Teacher generation exhausted bounded restarts; inspect INCOMPLETE.json';return
    if time.time()-state.get('last_generator_launch',0)<300: return
    log=HERE/'generator.log'
    with log.open('a') as f:
        child=subprocess.Popen([str(PYTHON),str(HERE/'generate.py')],cwd=ROOT,env=environment(),
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN[child.pid]=child
    state['generator']={'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'started':time.time()}
    state['last_generator_launch']=time.time();state['generator_restarts']=state.get('generator_restarts',0)+1
    event('generator_launch',pid=child.pid,attempt=state['generator_restarts'])


def tick(state):
    reap(state)
    gpus=gpu_snapshot(state)
    record_utilization(state,gpus)
    paused=(HERE/'PAUSE').exists()
    if not paused:
        # Start on empty GPUs before packing. Full budgets account for workers still loading.
        running=[x for x in state['jobs'].values() if x['status']=='running']
        # Retain two lanes for the long 14B queue; perpetual small-job packing must
        # not fragment every GPU and starve the dominant part of the makespan.
        lanes=state.setdefault('large_model_lanes',list(dict.fromkeys(
            j['gpu'] for j in running if j['kind']=='train' and j['model']=='Qwen3-14B'))[:2])
        for gpu in sorted(gpus,key=lambda g:(sum(j['gpu']==g for j in running),g)):
            if len(lanes)>=2: break
            if gpu not in lanes: lanes.append(gpu)
        large_waiting=any(j['kind']=='train' and j['model']=='Qwen3-14B' and ready(j,state)
                          for j in state['jobs'].values())
        order=sorted(gpus,key=lambda gpu:(sum(j['gpu']==gpu for j in running),gpus[gpu]['util'],gpu))
        for gpu in order:
            snapshot=gpus[gpu]
            active=[x for x in state['jobs'].values() if x['status']=='running' and x['gpu']==gpu]
            candidates=[j for j in state['jobs'].values() if ready(j,state)]
            if large_waiting and gpu in lanes:
                candidates=[j for j in candidates if j['kind']=='evaluate' or j['model']=='Qwen3-14B']
                candidates.sort(key=lambda j:(not (j['kind']=='train' and j['model']=='Qwen3-14B'),priority(j,state)))
            else:
                candidates.sort(key=lambda j:priority(j,state))
            for j in candidates:
                profile=choose_profile(j,active,snapshot)
                if profile is None: continue
                launch(j,gpu,profile);atomic_json(STATE,state);break
    for gpu,snapshot in gpus.items():
        active=[j for j in state['jobs'].values() if j['status']=='running' and j['gpu']==gpu]
        snapshot['committed_gib']=committed_gib(active,snapshot)
    monitor_generator(state,paused)
    state['updated']=time.time(); state['gpus']=gpus
    state['paused']=paused
    state['counts']={s:sum(j['status']==s for j in state['jobs'].values()) for s in ('pending','running','retry','done','failed')}
    failed=[{'job':j['id'],'log':j.get('log'),'reason':j.get('failure_tail','')[-800:]}
            for j in state['jobs'].values() if j['status']=='failed']
    if failed or state.get('generator_attention'):
        atomic_json(HERE/'ATTENTION.json',{'time':time.time(),'failed':failed,'teacher':state.get('generator_attention'),
                    'message':'These jobs exhausted bounded retries; unaffected jobs continue. No failed result is published.'})
    summary=collect()
    state['verified_seed_evaluations']=summary['completed_seed_evaluations']
    for baseline,group in summary['baselines'].items():
        matching=[j for j in state['jobs'].values() if not j['base'] and j['baseline']==baseline]
        if group['completed_seed_evaluations']==36 and all(j['status']=='done' for j in matching) and baseline not in state.get('board_done',[]):
            subprocess.run([str(PYTHON),'-m','tdwa.board','done',BASELINE_INFO[baseline]['board'],
                '--owner',OWNER,'--no-push','--result',baseline+'：3 学生×4 任务×3 种子及通用能力评测齐全，主表已自动回填。',
                '--output','outputs/engineer/grpo_main/RESULTS.md'],cwd=ROOT,check=True)
            state.setdefault('board_done',[]).append(baseline)

    state['phase']='complete' if summary['completed_seed_evaluations']==108 and summary['matched_base_count']==12 and all(j['status']=='done' for j in state['jobs'].values()) else 'running'
    unfinished=[j for j in state['jobs'].values() if j['status']!='done']
    if unfinished and not state['counts']['running'] and all(
        j['status']=='failed' or (j['kind']=='evaluate' and not j['base'] and
        state['jobs']['train:'+j['name']]['status']=='failed') for j in unfinished): state['phase']='blocked'
    atomic_json(STATE,state)
    if state['phase']=='complete':
        manifest=read_json(HERE/'manifest.json',{})
        manifest.update(status='complete',finished=time.time())
        atomic_json(HERE/'manifest.json',manifest)
        atomic_json(HERE/'COMPLETE.json',{'finished':time.time(),'verified_seed_evaluations':108,'state':str(STATE)})
    return state['phase']


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--once',action='store_true'); a=ap.parse_args()
    lock=(HERE/'watchdog.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read_json(STATE,{'started':time.time(),'jobs':jobs(),'generator_restarts':0})
    # Recover a worker launched just before a watchdog crash, even if its PID was not yet persisted.
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        cmd=p.info['cmdline'] or []
        if str(HERE/'worker.py') not in cmd or '--smoke' in cmd: continue
        try:
            at=cmd.index(str(HERE/'worker.py')); kind=cmd[at+1]
            task=cmd[cmd.index('--task')+1]; model=cmd[cmd.index('--model')+1]
            seed=int(cmd[cmd.index('--seed')+1]); base='--base' in cmd
            baseline=cmd[cmd.index('--baseline')+1] if '--baseline' in cmd else 'grpo_k'
            name=f'GRPO_BASE__{task}__{model}' if base else run_name(task,model,seed,baseline)
            j=state['jobs'][kind+':'+name]
            if j['status']=='running' and alive(j): continue
            worker_env=p.environ()
            gpu=int(worker_env['CUDA_VISIBLE_DEVICES'])
            if gpu not in GPUS: raise RuntimeError('Recovered worker is on an unauthorized GPU')
            j.update(status='running',pid=p.pid,created=p.create_time(),started=p.create_time(),gpu=gpu,
                     attempts=max(1,j['attempts']),**recovered_profile(j,worker_env))
            if kind=='train': j['microbatch']=int(cmd[cmd.index('--microbatch')+1])
            event('worker_adopted',job=j['id'],pid=p.pid,gpu=gpu)
        except (psutil.Error,ValueError,KeyError): continue
    adopt_generator(state)
    state['watchdog_pid']=os.getpid(); state['watchdog_created']=psutil.Process().create_time()
    atomic_json(HERE/'watchdog_process.json',{'pid':os.getpid(),'created':psutil.Process().create_time(),'started':time.time()})
    event('watchdog_start',pid=os.getpid())
    failures=0
    while not (HERE/'STOP').exists():
        try:
            phase=tick(state); failures=0
            print(time.strftime('%F %T'),state['counts'],'verified',state['verified_seed_evaluations'],flush=True)
            if phase=='complete' or a.once: return
        except Exception as e:
            import traceback
            traceback.print_exc(); failures+=1
            event('watchdog_exception',type=type(e).__name__,message=str(e)[:1000],consecutive=failures)
            atomic_json(STATE,state)
            if a.once: raise
        time.sleep(5 if failures<3 else 60)
    # STOP prevents new launches; existing workers are left intact and can be adopted on restart.
    state['phase']='stopped'; atomic_json(STATE,state)


if __name__=='__main__':
    main()
