"""Admission regression checks: no CUDA allocation or job launches."""
import copy
import hashlib
import json
from pathlib import Path
import time
from resource_policy import (profiles, choose_profile, recovered_profile, legacy_reserve,
                             ENV_RESERVE, ENV_CAP, ENV_PROFILE, fits)
from common import HERE, atomic_json, GPUS


def validate():
    small={'id':'small','kind':'train','baseline':'grpo_k','model':'Qwen3-1.7B','microbatch':8}
    medium={**small,'id':'medium','model':'Qwen3-4B','microbatch':4}
    big={**small,'id':'big','model':'Qwen3-14B','microbatch':4}
    ev={**small,'id':'eval','kind':'evaluate'}
    snap={'total_gib':143771/1024,'used_gib':0,'foreign_gib':.1,'job_used_gib':{},'util':0}
    checks={}
    def check(label,value):
        assert value,label
        checks[label]=True
    old={**small,'reserved_gib':75}
    compact=choose_profile(small,[old],snap)
    check('legacy75_plus_compact56',compact['profile']=='grpo-compact' and compact['microbatch']==4)
    check('legacy90_does_not_shrink',choose_profile(small,[{**medium,'reserved_gib':90}],snap) is None)
    check('normal_small_pair',choose_profile(medium,[{**small,'reserved_gib':64}],snap)['reserved_gib']==66)
    check('14B_training_exclusive',choose_profile(ev,[{**big,'reserved_gib':128}],snap) is None)
    check('third_training_blocked',choose_profile(small,[{**small,'reserved_gib':10},{**medium,'reserved_gib':10}],snap) is None)
    check('three_small_evals_fit',choose_profile(ev,[{**ev,'reserved_gib':44},{**ev,'id':'eval2','reserved_gib':44}],snap) is not None)
    check('fourth_worker_blocked',choose_profile(ev,[{**ev,'reserved_gib':5}]*3,snap) is None)
    check('foreign_memory_protected',choose_profile(medium,[{**small,'reserved_gib':64}],{**snap,'foreign_gib':20}) is None)
    check('observed_growth_protected',choose_profile(medium,[{**small,'reserved_gib':64}],{**snap,'job_used_gib':{'small':90}}) is None)
    check('startup_budget_reserved',choose_profile(small,[{**small,'reserved_gib':64}],snap)['reserved_gib']==64)
    check('old_orphan_recovers75',recovered_profile(small,{})['reserved_gib']==75)
    env={ENV_PROFILE:'grpo-compact',ENV_RESERVE:'56',ENV_CAP:'50'}
    rec=recovered_profile(small,env)
    check('capped_orphan_recovers56',rec['reserved_gib']==56 and rec['torch_cap_gib']==50)
    check('oom_retries_alone',choose_profile({**small,'oom':True},[old],snap) is None)
    check('oom_recovery_budget',choose_profile({**small,'oom':True,'microbatch':2},[],snap)['reserved_gib']==75)
    for model in (small,medium,big):
        for p in profiles(model):
            for batch in (32,64):
                assert batch % p['microbatch']==0 and p['microbatch']*(batch//p['microbatch'])==batch
            assert p['torch_cap_gib']<=p['reserved_gib']-4
    check('effective_batches_preserved',True)
    import grpo_training
    frozen=json.loads((HERE/'TRAIN_PROTOCOL.json').read_text())
    check('GRPO_protocol_unchanged',grpo_training.PROTOCOL==frozen['grpo_k'])
    check('authorized_GPUs_unchanged',list(GPUS)==[2,4,5,6,7])
    # Exercise the actual scheduler with fake launches: busy old workers survive,
    # one compact backfill starts; a vacant large-model lane takes a 14B train.
    import watchdog as w
    current=copy.deepcopy(json.loads((HERE/'watchdog_state.json').read_text()))
    w.gpu_snapshot(current)  # Exercise nvidia-smi process attribution read-only.
    original={name:getattr(w,name) for name in ['reap','gpu_snapshot','record_utilization','ready','launch',
              'atomic_json','monitor_generator','collect']}
    try:
        live={**small,'id':'live','name':'live','seed':0,'task':'t1','status':'running','gpu':7,'pid':1,
              'attempts':1,'reserved_gib':75}
        pending={**small,'id':'next','name':'next','seed':0,'task':'t2','status':'pending','attempts':0}
        heavy={**big,'id':'heavy','name':'heavy','seed':0,'task':'t2','status':'pending','attempts':0}
        train4={**medium,'id':'blocked','name':'blocked','seed':0,'task':'t1','status':'running',
                'gpu':4,'pid':2,'attempts':1,'reserved_gib':90}
        simulated={'jobs':{j['id']:j for j in [live,pending,heavy,train4]},'large_model_lanes':[2,6]}
        launched=[]
        w.reap=lambda s:None
        w.gpu_snapshot=lambda s:{g:copy.deepcopy(snap) for g in GPUS}
        w.record_utilization=lambda *args:None
        w.ready=lambda j,s:j['status']=='pending'
        def fake_launch(j,g,p):
            launched.append((j['id'],g,p['profile']))
            j.update(status='running',gpu=g,reserved_gib=p['reserved_gib'],pid=999)
        w.launch=fake_launch
        w.atomic_json=lambda *args:None
        w.monitor_generator=lambda *args:None
        w.collect=lambda:{'completed_seed_evaluations':0,'matched_base_count':12,'baselines':{}}
        w.tick(simulated)
        check('large_job_not_starved',any(j=='heavy' and g in [2,6] for j,g,p in launched))
        check('original_worker_not_relaunched',live['pid']==1 and train4['pid']==2)
        check('launches_only_authorized_GPUs',all(g in GPUS for j,g,p in launched))
    finally:
        for name,value in original.items():setattr(w,name,value)
    atomic_json(HERE/'resource_validation.json',{'time':time.time(),'passed':True,'checks':checks,
        'code_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in
                       ['resource_policy.py','watchdog.py','grpo_training.py','validate_resources.py']}})
    print(json.dumps({'passed':True,'checks':len(checks)}))


if __name__=='__main__':validate()
