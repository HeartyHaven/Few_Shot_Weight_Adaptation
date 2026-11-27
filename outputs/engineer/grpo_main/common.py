"""Shared configuration for the user-authorized main-table GRPO run."""
import hashlib
import json
import os
from pathlib import Path
import time
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / '.venv-vllm/bin/python'
TASKS = ('t1', 't2', 't4', 't5')
MODELS = ('Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-14B')
GPUS = (2, 4, 5, 6, 7)
SEEDS = (0, 1, 2)
TEACHER = None
OWNER = 'codex-grpo@masdar'
BOARD_ID = 'BL-GRPO-MAIN-20261126'
EVAL_PROTOCOL = 2
BASELINES = ('grpo_100', 'grpo_k', 'distill_k')
BASELINE_INFO = {
    'grpo_100': {'prefix':'GRPO_100','label':'GRPO（100 条）','board':'BL-GRPO-100-20261126'},
    'grpo_k': {'prefix':'GRPO_K','label':'GRPO（千级数据）','board':'BL-GRPO-MAIN-20261126'},
    'distill_k': {'prefix':'DISTILL_K','label':'Distill（Astra，千级）','board':'BL-DISTILL-K-20261126'},
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


_SHA_CACHE = {}

def sha(path):
    path=Path(path)
    st=path.stat()
    stamp=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    key=str(path)
    cached=_SHA_CACHE.get(key)
    if cached and cached[0]==stamp: return cached[1]
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    value=digest.hexdigest();_SHA_CACHE[key]=(stamp,value)
    return value



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
    ipc_tmp = Path('/tmp') / f'fswa_grpo_{os.getuid()}'
    ipc_tmp.mkdir(mode=0o700, exist_ok=True)
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PATH=os.pathsep.join([str(HERE/'deps/bin'),str(ROOT/'.venv-vllm/bin'),env.get('PATH','')]),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               HF_HOME=str(ROOT / 'cache/hf'), HF_HUB_CACHE=str(ROOT / 'cache/hf'),
               HF_DATASETS_CACHE=str(ROOT / 'cache/hf/datasets'), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TDWA_HF=str(ROOT / 'cache/hf'),
               PYTHONPATH=os.pathsep.join([str(HERE / 'deps'), str(ROOT / 'outputs/engineer/distill_main/deps'), str(ROOT), str(HERE)]),
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


def run_name(task, model, seed, baseline='grpo_k'):
    return f'{BASELINE_INFO[baseline]["prefix"]}__{task}__{model}__s{seed}'


def dataset_dir(baseline):
    return HERE / {'grpo_100':'data_100','grpo_k':'data','distill_k':'teacher_data'}[baseline]


def progress(path, **fields):
    atomic_json(path, {'time': time.time(), 'pid': os.getpid(), **fields})
