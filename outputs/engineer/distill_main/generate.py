"""Teacher demonstrations from the 100 public dev examples ONLY; resumable."""
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
MAX_CALLS = 240
MAX_TOTAL_TOKENS = 8_000_000


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
    data = HERE / 'data'; data.mkdir(exist_ok=True)
    lock = (data / 'generator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    api = client()
    stats = read_json(data / 'usage.json', {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'errors': 0})
    all_items = {task: [json.loads(l) for l in (ROOT / 'data' / task / 'dev.jsonl').read_text().splitlines() if l.strip()] for task in TASKS}
    assert all(len(v) == 100 for v in all_items.values())
    for task in TASKS:
        (data / task).mkdir(exist_ok=True)
    calls_dir = data / 'calls'; calls_dir.mkdir(exist_ok=True)

    def accepted(task, it):
        r = read_json(data / task / (it['id'] + '.json'))
        return r and r.get('input_sha256') == sha(ROOT / 'data' / task / 'dev.jsonl') and G.grade(it, r.get('solution', ''))['correct']

    def finalize(task):
        items = all_items[task]
        if not all(accepted(task, it) for it in items):
            return
        rows = [read_json(data / task / (it['id'] + '.json')) for it in items]
        path = data / f'{task}.jsonl'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        os.replace(tmp, path)
        atomic_json(data / f'{task}.ready.json', {'n': len(rows), 'sha256': sha(path), 'teacher': TEACHER,
                    'effort': 'high', 'source': f'data/{task}/dev.jsonl', 'source_sha256': sha(ROOT / 'data' / task / 'dev.jsonl'),
                    'synthetic_examples': 0, 'time': time.time()})

    def work(task, items, attempt):
        if STOP.is_set(): return
        with LOCK:
            if stats['calls'] >= MAX_CALLS or stats['prompt_tokens'] + stats['completion_tokens'] >= MAX_TOTAL_TOKENS:
                STOP.set(); return
            stats['calls'] += 1; call_id = stats['calls']
            atomic_json(data / 'usage.json', stats)
        started = time.time()
        request = prompt(task, items)
        record = {'task': task, 'ids': [r['id'] for r in items], 'attempt': attempt, 'teacher': TEACHER, 'started': started,
                  'request': request}
        try:
            response = api.chat.completions.create(model=TEACHER, reasoning_effort='high', max_tokens=16000,
                       messages=[{'role': 'user', 'content': request}])
            text = response.choices[0].message.content or ''
            record.update(response=text, finish_reason=response.choices[0].finish_reason,
                          returned_model=response.model, usage=response.usage.model_dump() if response.usage else {})
            u = response.usage
            with LOCK:
                stats['prompt_tokens'] += (u.prompt_tokens or 0) if u else 0
                stats['completion_tokens'] += (u.completion_tokens or 0) if u else 0
                atomic_json(data / 'usage.json', stats)
            got = {str(r['id']): r['solution'] for r in parse(text) if isinstance(r, dict) and 'id' in r and isinstance(r.get('solution'), str)}
            kept = 0
            for it in items:
                sol = got.get(str(it['id']), '')
                check = G.grade(it, sol)
                if not check['correct'] or 'final answer:' not in sol.lower() or len(sol.split()) < 20:
                    continue
                atomic_json(data / task / (it['id'] + '.json'), {**it, 'solution': sol, 'teacher': TEACHER,
                            'input_sha256': sha(ROOT / 'data' / task / 'dev.jsonl'), 'call_id': call_id,
                            'source': 'dev', 'kept': True, 'grade': check})
                kept += 1
            record['kept'] = kept
        except Exception as e:
            # Do not print raw SDK errors: HTTP messages may contain configuration.
            record['error_type'] = type(e).__name__
            record['status_code'] = getattr(e, 'status_code', None)
            with LOCK:
                stats['errors'] += 1
            if record['status_code'] in (401, 403):
                STOP.set()
        finally:
            record['seconds'] = round(time.time() - started, 2)
            atomic_json(calls_dir / f'{call_id:04d}.json', record)
            with LOCK:
                atomic_json(data / 'usage.json', stats)
                progress(data / 'heartbeat.json', calls=stats['calls'], last_call=call_id,
                         accepted={t: sum(bool(accepted(t, it)) for it in all_items[t]) for t in TASKS})
            print(json.dumps({k: record[k] for k in ('task','attempt','kept','error_type','status_code','seconds') if k in record}), flush=True)
            with LOCK:
                finalize(task)

    for attempt in range(1, 5):
        jobs = []
        for task in TASKS:
            finalize(task)
            missing = [it for it in all_items[task] if not accepted(task, it)]
            size = 5 if attempt == 1 else 2
            jobs.extend((task, missing[i:i+size], attempt) for i in range(0, len(missing), size))
        if not jobs or STOP.is_set(): break
        # Round-robin tasks so first datasets complete without starving the others.
        jobs.sort(key=lambda j: (all_items[j[0]].index(j[1][0]) // 5, TASKS.index(j[0])))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda j: work(*j), jobs))
        if attempt < 4 and any(not (data / f'{t}.ready.json').exists() for t in TASKS):
            time.sleep(15 * attempt)
    done = all((data / f'{t}.ready.json').exists() for t in TASKS)
    progress(data / 'heartbeat.json', done=done, calls=stats['calls'],
             accepted={t: sum(bool(accepted(t, it)) for it in all_items[t]) for t in TASKS})
    if not done:
        raise SystemExit('Teacher data incomplete; see data/heartbeat.json and redacted per-call statuses')


if __name__ == '__main__':
    main()
