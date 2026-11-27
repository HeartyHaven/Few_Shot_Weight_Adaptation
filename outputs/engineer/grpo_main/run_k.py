"""Plain runner for the thousand-example Distill / GRPO baselines.

One cell = (method, model, benchmark, seed): train one LoRA adapter with worker.py, then evaluate it.
Each listed GPU runs one cell at a time, in order. Verified results are skipped, and both trainers
resume from their own checkpoints, so repeating the same command continues after an interruption.
No packing, no priority lanes, no report editing. Steps and settings: BASELINES_K.md in the repo root.

    P=.venv-vllm/bin/python; R=outputs/engineer/grpo_main/run_k.py
    $P $R --check                          # prerequisites, CPU only
    $P $R --dry-run                        # list the 24 cells and their state
    $P $R --gpus 0,1                       # run everything that is runnable
    $P $R --gpus 0 --methods grpo_k --models Qwen3-14B
"""
import argparse
import os
import signal
import subprocess
import threading
import time

from common import HERE, ROOT, PYTHON, TASKS, MODELS, dataset_dir, environment, model_path, read_json, run_name, sha
from watchdog import completed
import collect_k

METHODS = ('distill_k', 'grpo_k')
# Starting microbatch (gradient accumulation keeps the effective batch fixed) and the vLLM memory
# fraction for evaluation: the values used on the 141 GiB H200 cards.
MICROBATCH = {'distill_k': {'Qwen3-1.7B': 8, 'Qwen3-4B': 4, 'Qwen3-14B': 2},
              'grpo_k': {'Qwen3-1.7B': 8, 'Qwen3-4B': 4, 'Qwen3-14B': 4}}
EVAL_MEM = {'Qwen3-1.7B': .28, 'Qwen3-4B': .28, 'Qwen3-14B': .50}
LOCK = threading.Lock()
CHILDREN = {}
STOPPING = threading.Event()


def say(*parts):
    print(time.strftime('%F %T'), *parts, flush=True)


def job(kind, cell):
    name = (f'GRPO_BASE__{cell["task"]}__{cell["model"]}' if cell['base']
            else run_name(cell['task'], cell['model'], cell['seed'], cell['method']))
    return {'id': f'{kind}:{name}', 'kind': kind, 'name': name, 'task': cell['task'], 'model': cell['model'],
            'seed': cell['seed'], 'base': cell['base'], 'baseline': cell['method']}


def state(cell):
    """done | train | evaluate | blocked (teacher demonstrations for this benchmark are not frozen yet)."""
    evaluation = job('evaluate', cell)
    if cell['base']:
        # --with-base re-measures the Base on this machine; a record copied from the earlier batch does not count.
        record = read_json(HERE / 'evaluations' / f'{evaluation["name"]}.json', {})
        return 'done' if completed(evaluation) and 'reused_from' not in record else 'evaluate'
    if completed(evaluation):
        return 'done'
    if completed(job('train', cell)):
        return 'evaluate'
    if not (dataset_dir(cell['method']) / f'{cell["task"]}.ready.json').exists():
        return 'blocked'
    return 'train'


def cells(a):
    result = []
    if a.with_base:
        result += [{'method': 'grpo_k', 'model': m, 'task': t, 'seed': 0, 'base': True}
                   for m in a.models for t in a.tasks]
    result += [{'method': method, 'model': m, 'task': t, 'seed': seed, 'base': False}
               for seed in a.seeds for m in a.models for t in a.tasks for method in a.methods]
    return result


def step(kind, cell, gpu, a):
    """Run one train or evaluate step; True once its output passes the completeness check."""
    j = job(kind, cell)
    microbatch = MICROBATCH[cell['method']][cell['model']]
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    for attempt in range(1, a.max_attempts + 1):
        if STOPPING.is_set():
            return False
        cmd = [str(PYTHON), str(HERE / 'worker.py'), kind, '--task', cell['task'], '--model', cell['model'],
               '--seed', str(cell['seed']), '--baseline', cell['method']]
        if kind == 'train':
            cmd += ['--microbatch', str(microbatch)]
        else:
            cmd += ['--mem', str(a.eval_mem or EVAL_MEM[cell['model']])] + (['--base'] if cell['base'] else [])
        index = 1
        while (logs / f'{j["name"]}.{kind}.run{index}.log').exists():
            index += 1
        log = logs / f'{j["name"]}.{kind}.run{index}.log'
        env = environment()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        say(f'GPU {gpu} start {j["id"]}', f'attempt {attempt}', f'log {log.relative_to(ROOT)}')
        started = time.time()
        with log.open('w') as f:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        with LOCK:
            CHILDREN[child.pid] = child
        code = child.wait()
        with LOCK:
            CHILDREN.pop(child.pid, None)
        if completed(j):
            say(f'GPU {gpu} done  {j["id"]}', f'{(time.time() - started) / 3600:.2f} h')
            return True
        tail = log.read_text(errors='replace')[-5000:].lower()
        oom = code == 42 or 'out of memory' in tail
        say(f'GPU {gpu} FAILED {j["id"]}', f'exit {code}', 'out of memory' if oom else '', f'see {log.relative_to(ROOT)}')
        if oom and kind == 'train' and microbatch > 1:
            microbatch //= 2
            say(f'GPU {gpu} retry with microbatch {microbatch} (effective batch unchanged)')
    return False


def gpu_loop(gpu, queue, a):
    while not STOPPING.is_set() and not (HERE / 'STOP').exists():
        with LOCK:
            cell = next((c for c in queue if c['status'] == 'todo'), None)
            if cell is None:
                return
            cell['status'] = 'running'
        todo = state(cell)
        ok = todo == 'done'
        if todo == 'train':
            ok = step('train', cell, gpu, a) and step('evaluate', cell, gpu, a)
        elif todo == 'evaluate':
            ok = step('evaluate', cell, gpu, a)
        with LOCK:
            cell['status'] = 'done' if ok else 'failed'
            collect_k.write()


def check():
    """CPU-only prerequisites. Prints one line per item and returns True when all pass."""
    results = []

    def item(label, ok, detail=''):
        results.append(ok)
        print('OK  ' if ok else 'FAIL', label, detail, flush=True)

    item('python', PYTHON.exists(), str(PYTHON))
    probe = ('import torch, transformers, vllm, trl, peft, accelerate;'
             'print(torch.__version__, transformers.__version__, vllm.__version__, trl.__version__, '
             'peft.__version__, accelerate.__version__)')
    p = subprocess.run([str(PYTHON), '-c', probe], env=environment(), capture_output=True, text=True)
    versions = p.stdout.split()
    item('packages torch/transformers/vllm/trl/peft/accelerate',
         p.returncode == 0 and versions[3:] == ['0.25.1', '0.17.1', '1.11.0'],
         ' '.join(versions) if p.returncode == 0 else p.stderr.strip().splitlines()[-1])
    for model in MODELS:
        try:
            item(f'model {model}', True, str(model_path(model).relative_to(ROOT)))
        except RuntimeError as exc:
            item(f'model {model}', False, str(exc))
    for task in TASKS:
        ready = read_json(HERE / 'data' / f'{task}.ready.json', {})
        path = HERE / 'data' / f'{task}.jsonl'
        item(f'training questions {task}', path.exists() and sha(path) == ready.get('sha256'),
             f'{ready.get("n_examples")} questions')
        tests = sorted((ROOT / '_secret' / task).glob('test*.jsonl'))
        base = read_json(HERE / 'evaluations' / f'GRPO_BASE__{task}__{MODELS[0]}.json', {})
        item(f'hidden test {task} matches the Base records',
             bool(tests) and {str(x.relative_to(ROOT)): sha(x) for x in tests} == base.get('test_fingerprints'),
             f'{len(tests)} file(s) in _secret/{task}')
    for model in MODELS:
        for task in TASKS:
            item(f'matched Base {task} {model}',
                 completed(job('evaluate', {'method': 'grpo_k', 'model': model, 'task': task, 'seed': 0, 'base': True})))
    for task, (have, total, frozen) in collect_k.teacher_progress().items():
        print('INFO', f'teacher demonstrations {task}: {have}/{total}',
              'frozen, Distill-K can run' if frozen else 'not complete, Distill-K for this benchmark is blocked')
    print('ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED')
    return all(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpus', help='comma-separated GPU indices; one cell at a time on each')
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    ap.add_argument('--seeds', nargs='+', type=int, choices=[0, 1, 2], default=[0])
    ap.add_argument('--with-base', action='store_true', help='also re-measure the 12 matched Base scores here')
    ap.add_argument('--eval-mem', type=float, help='vLLM gpu_memory_utilization for evaluation (default .28, 14B .50)')
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true', help='list cells and their state, start nothing')
    ap.add_argument('--check', action='store_true', help='verify prerequisites (CPU only), start nothing')
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if check() else 1)
    queue = cells(a)
    for cell in queue:
        cell['state'] = state(cell)
        cell['status'] = {'done': 'done', 'blocked': 'blocked'}.get(cell['state'], 'todo')
    for cell in queue:
        print(f'{cell["state"]:9}', 'Base' if cell['base'] else cell['method'], cell['model'], cell['task'],
              '' if cell['base'] else f's{cell["seed"]}')
    print({s: sum(c['state'] == s for c in queue) for s in ('done', 'train', 'evaluate', 'blocked')})
    if a.dry_run:
        return
    if not a.gpus:
        ap.error('--gpus is required to run')

    def stop(*_):
        STOPPING.set()
        with LOCK:
            for child in CHILDREN.values():
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    threads = [threading.Thread(target=gpu_loop, args=(int(g), queue, a)) for g in a.gpus.split(',')]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(1)
    collect_k.write()
    say('finished', {s: sum(c['status'] == s for c in queue) for s in ('done', 'failed', 'blocked', 'todo')},
        'table:', str((HERE / 'RESULTS_K.md').relative_to(ROOT)))
    if any(c['status'] == 'failed' for c in queue):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
