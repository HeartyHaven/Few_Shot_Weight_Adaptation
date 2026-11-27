"""Short, bounded read-only GPU/worker throughput observation."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from common import GPUS,HERE,read_json,atomic_json


def workers():
    result={}
    for j in read_json(HERE/'watchdog_state.json',{}).get('jobs',{}).values():
        if j['status']!='running':continue
        path=HERE/'runs'/j['name']/'train_progress.json' if j['kind']=='train' else HERE/'evaluation_progress'/f"{j['name']}.json"
        log=HERE/'runs'/j['name']/'train_log.jsonl'
        last=None
        if log.exists():
            for line in reversed(log.read_text().splitlines()):
                row=json.loads(line)
                if 'num_tokens' in row:last=row;break
        result[j['id']]={'pid':j['pid'],'gpu':j['gpu'],'kind':j['kind'],'resource_profile':j.get('resource_profile','legacy'),
            'reserved_gib':j['reserved_gib'],'progress':read_json(path,{}),'train_log':last}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,default=60);ap.add_argument('--output',default='utilization_after_packing.json')
    a=ap.parse_args();assert 5<=a.seconds<=600 and Path(a.output).name==a.output
    start=time.time();samples=[];before=workers()
    while time.time()-start<a.seconds:
        res=subprocess.run(['nvidia-smi','-i',','.join(map(str,GPUS)),
            '--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=15)
        now=time.time()
        for line in res.stdout.splitlines():
            gpu,util,mem,power=map(float,line.split(','))
            samples.append({'time':now,'gpu':int(gpu),'util':util,'memory_mib':mem,'power_w':power})
        time.sleep(1)
    end=time.time();after=workers()
    averages={str(g):round(statistics.mean(r['util'] for r in samples if r['gpu']==g),2) for g in GPUS}
    tail={str(g):round(statistics.mean(r['util'] for r in samples if r['gpu']==g and r['time']>=end-60),2) for g in GPUS}
    throughput={}
    for key,old in before.items():
        new=after.get(key)
        x=old['train_log'];y=new['train_log'] if new else None
        if x and y and old['pid']==new['pid'] and y['time']>x['time']:
            elapsed=y['time']-x['time']
            throughput[key]={'gpu':old['gpu'],'steps':y['step']-x['step'],'seconds':elapsed,
                'seconds_per_step':elapsed/(y['step']-x['step']),
                'tokens_per_second':(y['num_tokens']-x['num_tokens'])/elapsed}
    record={'start':start,'end':end,'samples':samples,'util_mean':averages,'last60s_util_mean':tail,
        'workers_before':before,'workers_after':after,'throughput':throughput}
    atomic_json(HERE/a.output,record)
    print(json.dumps({'util_mean':averages,'last60s_util_mean':tail,'throughput':throughput,'workers':len(after)},ensure_ascii=False))


if __name__=='__main__':main()
