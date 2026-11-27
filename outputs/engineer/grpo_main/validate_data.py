"""Checks exclusion, exact pairing, nested sizes and reward self-consistency without model access to held-out rows."""
import json
import time
from common import *
from prepare import question_hash, excluded
from grpo_training import reward_values
from tdwa import grade as G


def validate():
    result={'time':time.time(),'passed':False,'tasks':{}}
    for task in TASKS:
        large=[json.loads(x) for x in (HERE/'data'/f'{task}.jsonl').read_text().splitlines()]
        small=[json.loads(x) for x in (HERE/'data_100'/f'{task}.jsonl').read_text().splitlines()]
        original=[json.loads(x) for x in (ROOT/'data'/task/'dev.jsonl').read_text().splitlines()]
        distilled=[json.loads(x) for x in (ROOT/'outputs/engineer/distill_main/data'/f'{task}.jsonl').read_text().splitlines()]
        by_id={r['id']:r for r in large};sd={r['id']:r for r in small};od={r['id']:r for r in original};dd={r['id']:r for r in distilled}
        assert len(large)==(1488 if task=='t4' else 2000) and len(large)==len(by_id)
        assert len(small)==100 and set(sd)==set(od)==set(dd)
        for key,row in sd.items():
            for field in ('question','answer','task'):
                assert row[field]==od[key][field]==dd[key][field]==by_id[key][field]
        hashes={question_hash(r) for r in large};hidden={question_hash(r) for r in excluded(task)}
        assert len(hashes)==len(large) and not hashes&hidden
        assert {question_hash(r) for r in small}.issubset(hashes)
        for folder,rows in [('data',large),('data_100',small)]:
            path=HERE/folder/f'{task}.jsonl';ready=read_json(HERE/folder/f'{task}.ready.json')
            assert ready['n_examples']==len(rows) and ready['sha256']==sha(path)
        payload=[json.dumps({k:r[k] for k in ['id','task','answer','lower','upper'] if k in r}) for r in large]
        assert reward_values(['Final answer: '+str(r['answer']) for r in large],payload)==[1.0]*len(large)
        assert reward_values(['Final answer: INVALID_NO_ANSWER']*len(large),payload)==[0.0]*len(large)
        if task=='t5':
            generated=[r for r in large if r['id'] not in sd]
            assert len(generated)==1900
            assert len({r['structure_sha256'] for r in generated})==len(generated)
        result['tasks'][task]={'n_K':len(large),'n_100':100,'exact_original_distill_100_pairing':True,
            'nested':True,'hidden_overlap':0,'unique_ids_and_questions':True,
            'all_gold_reward_one':True,'all_invalid_reward_zero':True,
            'large_sha256':sha(HERE/'data'/f'{task}.jsonl'),'small_sha256':sha(HERE/'data_100'/f'{task}.jsonl')}
    result['passed']=True
    atomic_json(HERE/'data_validation.json',result)
    return result

if __name__=='__main__': print(json.dumps(validate(),ensure_ascii=False,indent=2))
