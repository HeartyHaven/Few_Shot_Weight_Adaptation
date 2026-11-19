"""One line per run: state, tool calls, edits, rollbacks, hidden-validation scores, restarts, whether evaluated."""
import json, glob, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for R in sorted(glob.glob(ROOT + '/runs/*')):
    n = os.path.basename(R); c = json.load(open(R + '/config.json')); fin = 'finished' in c
    ed = [json.load(open(m)) for m in sorted(glob.glob(R + '/edits/*/meta.json'))]
    ex = [json.load(open(x)) for x in sorted(glob.glob(R + '/exams/*.json'))]
    led = [json.loads(l) for l in open(R + '/ledger.jsonl')] if os.path.exists(R + '/ledger.jsonl') else []
    rb = [(x['to'], x['removed']) for x in led if x['event'] == 'rollback']; kr = sum(1 for x in led if x['event'] in ('kernel_restart',))
    calls = sum(1 for l in open(R + '/transcript.jsonl') if '"event": "tool"' in l) if os.path.exists(R + '/transcript.jsonl') else 0
    age = (time.time() - os.path.getmtime(R + '/transcript.jsonl')) / 60 if calls else -1
    print(f"{n} | {'FIN' if fin else 'run'} calls={calls} idle={age:.0f}m edits={[(e['edit_id'], e['accepted'], e['op_class'], e['n_values_changed']) for e in ed]} rb={rb} "
          f"exams={[(x['after_edit'], x['accuracy']) for x in ex]} restarts={kr} eval={'yes' if os.path.exists(ROOT + '/eval/' + n + '.json') else 'no'}")
