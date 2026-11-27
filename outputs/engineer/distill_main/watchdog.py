"""Persistent, GPU-restricted distillation scheduler with recovery and bounded retries."""
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
from common import atomic_json, environment, event, read_json, run_name, sha
from collect import collect, valid

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


def gpu_snapshot():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    result={}
    for line in p.stdout.splitlines():
        index,total,used,util=[int(x.strip()) for x in line.split(',')]
        if index in GPUS: result[index]={'total_gib':total/1024,'used_gib':used/1024,'util':util}
    if set(result)!=set(GPUS): raise RuntimeError('An authorized GPU is missing')
    return result


def jobs():
    result={}
    for seed in SEEDS:
        for task in TASKS:
            for model in MODELS:
                name=run_name(task,model,seed)
                for kind in ('train','evaluate'):
                    key=f'{kind}:{name}'
                    result[key]={'id':key,'kind':kind,'task':task,'model':model,'seed':seed,'base':False,
                                 'name':name,'status':'pending','attempts':0,
                                 'microbatch':{'Qwen3-1.7B':4,'Qwen3-4B':2,'Qwen3-14B':1}[model]}
    for task in TASKS:
        for model in MODELS:
            name=f'B3_BASE__{task}__{model}'; key=f'evaluate:{name}'
            result[key]={'id':key,'kind':'evaluate','task':task,'model':model,'seed':0,'base':True,
                         'name':name,'status':'pending','attempts':0}
    return result


def output(j):
    return HERE/'runs'/j['name']/'train_done.json' if j['kind']=='train' else HERE/'evaluations'/(j['name']+'.json')


def completed(j):
    r=read_json(output(j))
    if not r: return False
    if j['kind']=='evaluate': return valid(r,j['task'],j['model'])
    try:
        ready=read_json(HERE/'data'/(j['task']+'.ready.json'))
        return (ready and r['data_sha256']==ready['sha256'] and r['steps']==r['expected_steps']==65 and
                r['seed']==j['seed'] and r['task']==j['task'] and r['student']=='Qwen/'+j['model'] and r['n_examples']==100 and
                r['adapter_sha256']==sha(HERE/'runs'/j['name']/'adapter/adapter_model.safetensors'))
    except (KeyError,FileNotFoundError): return False


def heartbeat(j):
    return (HERE/'runs'/j['name']/'train_progress.json' if j['kind']=='train'
            else HERE/'evaluation_progress'/(j['name']+'.json'))


def ready(j,state):
    if j['status'] not in ('pending','retry') or time.time()<j.get('retry_after',0): return False
    if j['attempts']>=(8 if j['kind']=='evaluate' else 4): return False
    if j['kind']=='train':
        if not (HERE/'data'/(j['task']+'.ready.json')).exists(): return False
        # Validate the first real training run before multiplying its implementation across the grid.
        trained=any(x['kind']=='train' and x['status']=='done' for x in state['jobs'].values())
        active_train=any(x['kind']=='train' and x['status']=='running' for x in state['jobs'].values())
        if not trained and active_train: return False
    elif not j['base']:
        if not completed(state['jobs']['train:'+j['name']]): return False
    if j['kind']=='evaluate':
        evaluated=any(x['kind']=='evaluate' and x['status']=='done' for x in state['jobs'].values())
        active_eval=sum(x['kind']=='evaluate' and x['status']=='running' for x in state['jobs'].values())
        if not evaluated and active_eval>=2: return False
    return True


def resources(j):
    if j['kind']=='train':
        return {'Qwen3-1.7B':34,'Qwen3-4B':48,'Qwen3-14B':76}[j['model']],None
    mem=j.get('eval_mem',.50 if j['model']=='Qwen3-14B' else .28)
    return (72 if j['model']=='Qwen3-14B' else 44),mem


def priority(j):
    # Finish the first pass through all cells, then repetitions. Baselines fill otherwise idle capacity.
    phase=-1 if j['base'] else (0 if j['kind']=='evaluate' else 1)
    return (0 if j['base'] else j['seed'],TASKS.index(j['task']),MODELS.index(j['model']),phase)


def launch(j,gpu):
    reserve,mem=resources(j)
    cmd=[str(PYTHON),str(HERE/'worker.py'),j['kind'],'--task',j['task'],'--model',j['model'],'--seed',str(j['seed'])]
    if j['kind']=='train': cmd+=['--microbatch',str(j['microbatch'])]
    else:
        cmd+=['--mem',str(mem)]
        if j['base']: cmd+=['--base']
    logs=HERE/'logs'; logs.mkdir(exist_ok=True)
    code=HERE/'code_history'; code.mkdir(exist_ok=True)
    digest=sha(HERE/'worker.py')
    snapshot=code/f'worker_{digest}.py'
    if not snapshot.exists(): snapshot.write_bytes((HERE/'worker.py').read_bytes())
    path=logs/f'{j["name"]}.{j["kind"]}.attempt{j["attempts"]+1}.log'
    with path.open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN[p.pid]=p
    j.update(status='running',pid=p.pid,created=psutil.Process(p.pid).create_time(),started=time.time(),
             gpu=gpu,reserved_gib=reserve,log=str(path),attempts=j['attempts']+1,worker_source_sha256=digest)
    event('launch',job=j['id'],gpu=gpu,pid=p.pid,attempt=j['attempts'])


def generator(state):
    if all((HERE/'data'/(t+'.ready.json')).exists() for t in TASKS): return
    rec=read_json(HERE/'generator_process.json',{})
    if alive(rec):
        hb=read_json(HERE/'data/heartbeat.json',{})
        if time.time()-max(hb.get('time',rec.get('started',0)),rec.get('started',0))>1200:
            owned_stop(rec); event('teacher_stall',pid=rec['pid'])
        else: return
    count=state.get('generator_restarts',0)
    usage=read_json(HERE/'data/usage.json',{})
    if count>=3 or usage.get('calls',0)>=240 or time.time()<state.get('generator_retry_after',0): return
    with (HERE/'generate.log').open('a') as f:
        p=subprocess.Popen([str(PYTHON),str(HERE/'generate.py')],cwd=ROOT,env=environment(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(HERE/'generator_process.json',{'pid':p.pid,'created':psutil.Process(p.pid).create_time(),'started':time.time()})
    state['generator_restarts']=count+1; state['generator_retry_after']=time.time()+600
    CHILDREN[p.pid]=p
    event('teacher_restart',pid=p.pid,number=count+1)


def reap(state):
    now=time.time()
    for j in state['jobs'].values():
        if j['status']=='running':
            p=CHILDREN.get(j.get('pid'))
            if p is not None: p.poll()
            if alive(j):
                b=read_json(heartbeat(j),{})
                last=max(j['started'],b.get('time',0))
                if now-last>2700 or now-j['started']>4*3600:
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


def tick(state):
    reap(state); generator(state)
    gpus=gpu_snapshot()
    paused=(HERE/'PAUSE').exists()
    if not paused:
        candidates=sorted((j for j in state['jobs'].values() if ready(j,state)),key=priority)
        for j in candidates:
            # Recheck after every launch, especially the first-training gate.
            if not ready(j,state): continue
            running=[x for x in state['jobs'].values() if x['status']=='running']
            reserve,_=resources(j)
            available=[]
            for gpu,s in gpus.items():
                active=[x for x in running if x['gpu']==gpu]
                booked=sum(x['reserved_gib'] for x in active)
                # At most 3 workers/card, with 12+ GiB left for context/runtime variation.
                if len(active)>=3 or booked+reserve>128 or s['used_gib']+reserve>s['total_gib']-8: continue
                available.append((len(active),booked,s['util'],gpu))
            if available:
                gpu=min(available)[-1]; launch(j,gpu)
                atomic_json(STATE,state)
                # A just-launched process has not appeared in nvidia-smi yet. Reservations prevent overbooking.
    state['updated']=time.time(); state['gpus']=gpus
    state['paused']=paused
    state['counts']={s:sum(j['status']==s for j in state['jobs'].values()) for s in ('pending','running','retry','done','failed')}
    failed=[{'job':j['id'],'log':j.get('log'),'reason':j.get('failure_tail','')[-800:]}
            for j in state['jobs'].values() if j['status']=='failed']
    if failed:
        atomic_json(HERE/'ATTENTION.json',{'time':time.time(),'failed':failed,
                    'message':'These jobs exhausted bounded retries; unaffected jobs continue. No failed result is published.'})
    summary=collect()
    state['verified_seed_evaluations']=summary['completed_seed_evaluations']
    state['phase']='complete' if summary['completed_seed_evaluations']==36 else 'running'
    unfinished=[j for j in state['jobs'].values() if j['status']!='done']
    if unfinished and not state['counts']['running'] and all(
        j['status']=='failed' or (j['kind']=='evaluate' and not j['base'] and
        state['jobs']['train:'+j['name']]['status']=='failed') for j in unfinished): state['phase']='blocked'
    atomic_json(STATE,state)
    if state['phase']=='complete':
        manifest=read_json(HERE/'manifest.json',{})
        manifest.update(status='complete',finished=time.time())
        atomic_json(HERE/'manifest.json',manifest)
        atomic_json(HERE/'COMPLETE.json',{'finished':time.time(),'verified_seed_evaluations':36,'state':str(STATE)})
        subprocess.run([str(PYTHON),'-m','tdwa.board','done',BOARD_ID,'--owner',OWNER,'--no-push',
                        '--result','主表蒸馏完成：3 学生×4 任务×3 种子，同协议 Base、隐藏测试、四项通用能力均齐；主表已更新。',
                        '--output','outputs/engineer/distill_main/RESULTS.md'],cwd=ROOT,check=True)
    return state['phase']


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--once',action='store_true'); a=ap.parse_args()
    lock=(HERE/'watchdog.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read_json(STATE,{'started':time.time(),'jobs':jobs(),'generator_restarts':0})
    # Recover a worker launched just before a watchdog crash, even if its PID was not yet persisted.
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        cmd=p.info['cmdline'] or []
        if str(HERE/'worker.py') not in cmd: continue
        try:
            at=cmd.index(str(HERE/'worker.py')); kind=cmd[at+1]
            task=cmd[cmd.index('--task')+1]; model=cmd[cmd.index('--model')+1]
            seed=int(cmd[cmd.index('--seed')+1]); base='--base' in cmd
            name=f'B3_BASE__{task}__{model}' if base else run_name(task,model,seed)
            j=state['jobs'][kind+':'+name]
            if j['status']=='running' and alive(j): continue
            gpu=int(p.environ()['CUDA_VISIBLE_DEVICES'])
            if gpu not in GPUS: raise RuntimeError('Recovered worker is on an unauthorized GPU')
            j.update(status='running',pid=p.pid,created=p.create_time(),started=p.create_time(),gpu=gpu,
                     reserved_gib=resources(j)[0],attempts=max(1,j['attempts']))
            event('worker_adopted',job=j['id'],pid=p.pid,gpu=gpu)
        except (psutil.Error,ValueError,KeyError): continue
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
        time.sleep(15 if failures<3 else 60)
    # STOP prevents new launches; existing workers are left intact and can be adopted on restart.
    state['phase']='stopped'; atomic_json(STATE,state)


if __name__=='__main__':
    main()
