"""Read-only operational status; no dependencies beyond the standard library."""
import time
from common import HERE, BASELINES, read_json
s=read_json(HERE/'watchdog_state.json',{})
print(time.strftime('%F %T %Z'),'phase=',s.get('phase','not launched'),'counts=',s.get('counts',{}))
print('GPU 5-minute mean utilization:',s.get('utilization_5min',{}))
print('14B priority lanes:',s.get('large_model_lanes',[]))
for gpu,v in s.get('gpus',{}).items():
 print('GPU',gpu,'used/committed GiB',round(v['used_gib'],1),round(v.get('committed_gib',v['used_gib']),1))
for baseline in BASELINES:
 jobs=[j for j in s.get('jobs',{}).values() if not j['base'] and j['baseline']==baseline]
 print(baseline,{k:sum(j['kind']==k and j['status']=='done' for j in jobs) for k in ('train','evaluate')},'/36 each')
for j in s.get('jobs',{}).values():
 if j['status']=='running':
  path=HERE/'runs'/j['name']/'train_progress.json' if j['kind']=='train' else HERE/'evaluation_progress'/f"{j['name']}.json"
  progress=read_json(path,{})
  print('GPU',j['gpu'],j['id'],'pid',j['pid'],'profile',j.get('resource_profile','legacy'),'progress',
    {k:progress[k] for k in ('phase','step','expected_steps','of') if k in progress},
    'age_s',round(time.time()-progress.get('time',time.time())))
print('Teacher:',read_json(HERE/'teacher_data/heartbeat.json',{}))
print('Teacher usage:',read_json(HERE/'teacher_data/usage.json',{}))
if s.get('generator_attention'): print('ATTENTION:',s['generator_attention'])
for j in s.get('jobs',{}).values():
 if j['status'] in ('failed','retry'): print(j['status'],j['id'],'attempt',j['attempts'],'log',j.get('log'))
