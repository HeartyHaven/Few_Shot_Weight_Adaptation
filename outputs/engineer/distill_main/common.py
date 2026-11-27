"""Shared configuration for the user-authorized main-table distillation run."""
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / '.venv-vllm/bin/python'
TASKS = ('t1', 't2', 't4', 't5')
MODELS = ('Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-14B')
GPUS = (2, 4, 5, 6, 7)
SEEDS = (0, 1, 2)
TEACHER = 'gpt-6.2-astra'
OWNER = 'codex-distill@masdar'
BOARD_ID = 'BL-DISTILL-MAIN-20261126'
EVAL_PROTOCOL = 2


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(kind, **fields):
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'event': kind, **fields}, ensure_ascii=False) + '\n')


def model_path(model):
    paths = list((ROOT / 'cache/hf' / ('models--Qwen--' + model) / 'snapshots').glob('*/config.json'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one cached snapshot for {model}, found {len(paths)}')
    return paths[0].parent


def environment(gpu=None):
    env = dict(os.environ)
    ipc_tmp = Path('/tmp') / f'fswa_distill_{os.getuid()}'
    ipc_tmp.mkdir(mode=0o700, exist_ok=True)
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PATH=os.pathsep.join([str(HERE/'deps/bin'),str(ROOT/'.venv-vllm/bin'),env.get('PATH','')]),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               HF_HOME=str(ROOT / 'cache/hf'), HF_HUB_CACHE=str(ROOT / 'cache/hf'),
               HF_DATASETS_CACHE=str(ROOT / 'cache/hf/datasets'), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TDWA_HF=str(ROOT / 'cache/hf'),
               PYTHONPATH=os.pathsep.join([str(HERE / 'deps'), str(ROOT), str(HERE)]),
               TMPDIR=str(ipc_tmp), VLLM_WORKER_MULTIPROC_METHOD='spawn',
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache/torchinductor'),
               TRITON_CACHE_DIR=str(ROOT / 'cache/triton'),
               VLLM_CACHE_ROOT=str(ROOT / 'cache/vllm'),
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    (HERE / 'tmp').mkdir(exist_ok=True)
    if gpu is not None:
        if int(gpu) not in GPUS:
            raise ValueError(f'GPU {gpu} is outside user authorization')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    for key in list(env):
        if any(s in key for s in ('CLIPROXY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY')):
            env.pop(key, None)
    return env


def run_name(task, model, seed):
    return f'B3_100__{task}__{model}__s{seed}'


def progress(path, **fields):
    atomic_json(path, {'time': time.time(), 'pid': os.getpid(), **fields})
