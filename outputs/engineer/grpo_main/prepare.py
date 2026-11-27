"""Freeze thousand-example public/generated training sets; hidden rows only exclude overlaps."""
import ast
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from common import HERE, ROOT, TASKS, atomic_json, model_path, sha
sys.path.insert(0, str(ROOT))
from tdwa import data as D, grade as G

DATA_SEED = 261126


def question_hash(row):
    return hashlib.sha256(' '.join(row['question'].split()).encode()).hexdigest()


def excluded(task):
    rows = []
    for path in sorted((ROOT/'_secret'/task).glob('*.jsonl')):
        if path.name.startswith(('val', 'test')):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def program_structure(code):
    class Constants(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)
    return hashlib.sha256(ast.dump(Constants().visit(ast.parse(code)), include_attributes=False).encode()).hexdigest()


def code_item(rng, index):
    kind = index % 4
    lines = ['def f(x):']
    if kind == 0:
        arg = rng.randint(-20, 30)
        lines += ['    y = x']
        for _ in range(rng.randint(3, 6)):
            a, b = rng.randint(1, 9), rng.randint(2, 6)
            lines += rng.choice([
                [f'    y += {a}'], [f'    y = y * {b} - {a}'],
                [f'    y = y // {b} + {a}'], [f'    y = y % {a+b}'],
                [f'    y = y - {a} if y % 2 == 0 else y + {b}'],
                [f'    for i in range({b}):', f'        y += i if y % 2 else -{a}'],
                [f'    y = abs(y - {a})'], [f'    y = (y ^ {a}) + {b}'],
            ])
        lines += ['    return y']
        alternate = [arg+1, arg+7, -arg-3]
    elif kind == 1:
        arg = [rng.randint(-5, 12) for _ in range(rng.randint(3, 8))]
        lines += ['    y = list(x)']
        for _ in range(rng.randint(3, 6)):
            a, b = rng.randint(1, 5), rng.randint(2, 5)
            lines += rng.choice([
                [f'    y = [v + {a} for v in y]'], [f'    y = [v * {b} - {a} for v in y]'],
                [f'    y = [v % {b+3} for v in y]'], ['    y = y[::-1]'],
                ['    y = sorted(y)'], [f'    y = y[{a%3}:]'],
                [f'    y = [v for v in y if v % {b} != 0]'],
                [f'    y = y + [{a}]'], ['    y = sorted(set(y))'],
                ['    z = []', '    total = 0', '    for v in y:', '        total += v', '        z.append(total)', '    y = z'],
                [f'    y = [v - i for i, v in enumerate(y)]'],
            ])
        lines += [rng.choice(['    return y', '    return sum(y)', '    return (len(y), sum(y))'])]
        alternate = [arg+[2], list(reversed(arg)), [v+3 for v in arg]]
    elif kind == 2:
        arg = ''.join(rng.choice('abcde xyz') for _ in range(rng.randint(6, 15)))
        lines += ['    y = x']
        for _ in range(rng.randint(3, 6)):
            a = rng.choice('abcde')
            b = rng.choice('xyz')
            lines += rng.choice([
                ['    y = y[::-1]'], ['    y = y.upper()'], ['    y = y.lower()'],
                [f'    y = y.replace({a!r}, {b!r})'], ['    y = y.strip()'],
                ['    y = "".join(sorted(y))'], ['    y = "-".join(y.split())'],
                [f'    y = y[::{rng.choice([2, 3])}]'], [f'    y = {a!r} + y + {b!r}'],
                [f'    y = y.replace({a!r}, "", 1)'], ['    y = (y + y[:2])[:30]'],
            ])
        lines += [rng.choice(['    return y', '    return len(y)', '    return list(y[:6])'])]
        alternate = [arg+'ab', arg[::-1], 'cab xyzda']
    else:
        arg = {k: rng.randint(-3, 9) for k in 'abcd'}
        lines += ['    y = dict(x)']
        for _ in range(rng.randint(3, 6)):
            a, b = rng.randint(1, 5), rng.randint(2, 4)
            lines += rng.choice([
                [f'    y = {{k: v + {a} for k, v in y.items()}}'],
                [f'    y = {{k: v * {b} for k, v in y.items()}}'],
                [f'    y = {{k: v % {b+3} for k, v in y.items()}}'],
                [f'    y = {{k: v for k, v in y.items() if v >= {a}}}'],
                [f'    y[{rng.choice("abcd")!r}] = sum(y.values()) + {a}'],
                ['    y = {k: abs(v) for k, v in y.items()}'],
                ['    y = {k: v - i for i, (k, v) in enumerate(sorted(y.items()))}'],
                [f'    y.pop({rng.choice("abcd")!r}, None)'],
            ])
        lines += [rng.choice(['    return y', '    return sorted(y.items())', '    return sum(y.values())'])]
        alternate = [{**arg, 'e': 5}, {k:v+4 for k,v in arg.items()}, {'a': 19, 'b': -7}]
    code = '\n'.join(lines)
    safe = {name: getattr(__import__('builtins'), name) for name in
            ['range','list','dict','set','tuple','len','sum','abs','sorted','enumerate']}
    scope = {'__builtins__': safe}
    exec(compile(code, '<generated-bounded-program>', 'exec'), scope)
    result = scope['f'](arg)
    answer = repr(result)
    if len(answer) > 1500 or all(scope['f'](x) == result for x in alternate):
        return None
    assert ast.literal_eval(answer) == result
    return {'id': f'grpo-code-{index}', 'task':'t5', 'group':['integer','list','string','dictionary'][kind],
            'question': 'Here is a Python function and an input. Predict the exact value that the function returns for this input.\n\n'
                        f'```python\n{code}\n```\n\nInput: f({arg!r})',
            'answer':answer, 'code':code, 'input':repr(arg), 'structure_sha256':program_structure(code)}


def build_pool(task):
    rng = random.Random(DATA_SEED)
    hidden = excluded(task)
    hashes = {question_hash(r) for r in hidden}
    if task == 't1':
        from datasets import load_dataset
        blocked = {r['id'].split('-',1)[1] for r in hidden if r['id'].startswith('val-')}
        rows = []
        for r in load_dataset('nsk7153/MedCalc-Bench-Verified')['train']:
            if r['Category'] not in D.RULE_CATS:
                continue
            if str(r['Row Number']) in blocked:
                continue
            rows.append({'id':f"train-{r['Row Number']}",'task':'t1','group':r['Calculator Name'],
                         'question':f"Patient note:\n{r['Patient Note']}\n\nQuestion: {r['Question']}",
                         'answer':r['Ground Truth Answer'],'lower':float(r['Lower Limit']),
                         'upper':float(r['Upper Limit']),'calc_id':r['Calculator ID']})
        source = 'nsk7153/MedCalc-Bench-Verified train; same public source as B4; rule-based calculators matching the main task; excludes hidden-validation rows'
    elif task == 't4':
        from tdwa.data_mathcode import math_items
        blocked = {r['id'] for r in hidden}
        rows = [r for r in math_items('train',level5=False) if r['id'] not in blocked]
        source = 'EleutherAI/hendrycks_math train; number_theory and counting_and_probability; all levels; excludes 150 hidden-validation rows'
    elif task == 't2':
        rows = [D.make_item(rng,n,None,f'grpo-tracking-{n}-{i}')
                for n,count in [(3,600),(5,1200),(7,1200)] for i in range(count)]
        source = 'tdwa.data.make_item; independent seed 261126; 3/5/7 objects in 20/40/40 proportions'
    else:
        rows=[]; structures=set(); i=0
        while len(rows)<2400:
            row=code_item(rng,i); i+=1
            if row is None or row['structure_sha256'] in structures: continue
            structures.add(row['structure_sha256']); rows.append(row)
        source = 'independently generated bounded Python programs; executed ground truth; unique constant-normalized AST per training item; not official CRUXEval rows'
    rng.shuffle(rows)
    return rows, hashes, source, len(hidden)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path('Qwen3-1.7B'),local_files_only=True)
    manifest={'created':time.time(),'seed':DATA_SEED,'max_prompt_tokens':4096,'tasks':{}}
    target=HERE/'data'; target.mkdir(exist_ok=True)
    for task in TASKS:
        print('Preparing',task,flush=True)
        pool, hidden_hashes, source, hidden_n=build_pool(task)
        visible=[json.loads(x) for x in (ROOT/'data'/task/'dev.jsonl').read_text().splitlines()]
        assert len(visible)==100
        pool=visible+pool
        rows=[]; seen=set(); rejected=collections.Counter()
        for row in pool:
            h=question_hash(row)
            if h in hidden_hashes or h in seen:
                rejected['overlap_or_duplicate']+=1; continue
            if not G.grade(row,'Final answer: '+str(row['answer']))['correct']:
                rejected['reference_not_graded_correct']+=1; continue
            prompt=tok.apply_chat_template([{'role':'user','content':G.format_prompt(row)}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tok.encode(prompt,add_special_tokens=False))
            if n>4096:
                rejected['prompt_too_long']+=1; continue
            row['prompt_tokens']=n
            rows.append(row);seen.add(h)
            if task!='t4' and len(rows)==2000: break
        remainder=len(rows)%8
        if remainder:
            rejected['fixed_group_batch_alignment']=remainder
            rows=rows[:-remainder]
        assert len(rows)>=1000,(task,len(rows))
        assert {question_hash(r) for r in visible}.issubset({question_hash(r) for r in rows})
        assert all(question_hash(r) not in hidden_hashes for r in rows)
        path=target/f'{task}.jsonl'
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        if path.exists() and path.read_text()!=payload:
            raise RuntimeError(f'Refusing to overwrite frozen dataset: {path}')
        path.write_text(payload)
        info={'task':task,'n_examples':len(rows),'sha256':sha(path),'source':source+'; includes the exact 100 original visible questions','pool_n':len(pool),
              'hidden_rows_checked':hidden_n,'hidden_question_overlap':0,'rejections':dict(rejected),
              'groups':dict(collections.Counter(r['group'] for r in rows)),
              'max_prompt_tokens':max(r['prompt_tokens'] for r in rows),'data_seed':DATA_SEED,
              'original_visible_examples':100}
        atomic_json(target/f'{task}.ready.json',info);manifest['tasks'][task]=info
        small=HERE/'data_100';small.mkdir(exist_ok=True)
        small_path=small/f'{task}.jsonl'
        small_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in visible))
        atomic_json(small/f'{task}.ready.json',{'task':task,'n_examples':100,'sha256':sha(small_path),
                    'source':f'data/{task}/dev.jsonl; exact same questions as completed Distill-100',
                    'hidden_question_overlap':0,'nested_in_k':True})
        print(task,len(rows),'max_prompt',info['max_prompt_tokens'],'excluded',dict(rejected),flush=True)
    atomic_json(HERE/'data_manifest.json',manifest)


if __name__=='__main__': main()
