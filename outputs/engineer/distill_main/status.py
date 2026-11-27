"""Read-only status: .venv-vllm/bin/python outputs/engineer/distill_main/status.py"""
import time
from common import HERE, read_json

s=read_json(HERE/'watchdog_state.json',{})
print('Phase:',s.get('phase','not started'),'| status age:',round(time.time()-s.get('updated',time.time())),'seconds')
print('Jobs:',s.get('counts',{}),'| verified distilled evaluations:',s.get('verified_seed_evaluations',0),'/36')
for gpu,r in s.get('gpus',{}).items():
    active=[j for j in s.get('jobs',{}).values() if j.get('status')=='running' and str(j.get('gpu'))==str(gpu)]
    print(f'GPU {gpu}: {r["util"]}% | {r["used_gib"]:.1f} GiB | '+', '.join(j['id'] for j in active))
for j in s.get('jobs',{}).values():
    if j['status'] in ('failed','retry'):
        print(j['status'].upper(),j['id'],'attempt',j['attempts'],'log:',j.get('log'))
print('Teacher:',read_json(HERE/'data/heartbeat.json',{}))
print('Results:',HERE/'RESULTS.md')
