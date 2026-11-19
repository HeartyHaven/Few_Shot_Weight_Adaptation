"""Read-only WP0 environment evidence. Emits JSON to stdout; no token contents."""
import datetime
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

HF_HOME = Path('/mnt/data2/users/lang.gao/huggingface_cache')


def command(args):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'command': args, 'exit_code': proc.returncode,
                'stdout': proc.stdout, 'stderr': proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': args, 'error_type': type(exc).__name__, 'error': str(exc)}


def snapshot(repo):
    folders = [base / ('models--' + repo.replace('/', '--')) for base in [HF_HOME, HF_HOME / 'hub']]
    entries = []
    for path in sorted(p for folder in folders for p in folder.glob('snapshots/*')):
        files = [{'name': item.name, 'bytes': item.stat().st_size,
                  'is_symlink': item.is_symlink()}
                 for item in sorted(path.iterdir()) if item.is_file()]
        entry = {'path': str(path), 'revision': path.name, 'files': files}
        index = path / 'model.safetensors.index.json'
        if index.is_file():
            shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
            entry['indexed_weight_shards'] = shards
            entry['all_indexed_shards_exist'] = all((path / name).is_file() for name in shards)
        cfg = path / 'config.json'
        if cfg.is_file():
            entry['config'] = json.loads(cfg.read_text())
            entry['config_sha256'] = hashlib.sha256(cfg.read_bytes()).hexdigest()
        tok = path / 'tokenizer_config.json'
        if tok.is_file():
            template = json.loads(tok.read_text()).get('chat_template', '')
            entry['chat_template_contains_enable_thinking'] = 'enable_thinking' in str(template)
        entries.append(entry)
    return {'checked_cache_paths': [str(folder) for folder in folders],
            'cache_directories_found': [str(folder) for folder in folders if folder.is_dir()],
            'snapshots': entries,
            'model_loaded': False}


def main():
    dns = {}
    for host in ['huggingface.co', 'pypi.org']:
        try:
            addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            dns[host] = {'resolved': True, 'addresses': sorted({r[4][0] for r in addresses})}
        except OSError as exc:
            dns[host] = {'resolved': False, 'error_type': type(exc).__name__,
                         'errno': exc.errno, 'error': str(exc)}
    disk = {path: dict(zip(['total_bytes', 'used_bytes', 'free_bytes'], shutil.disk_usage(path)))
            for path in ['/', '/mnt/data2']}
    datasets = {}
    for repo in ['nsk7153/MedCalc-Bench-Verified', 'lukaemon/bbh']:
        name = 'datasets--' + repo.replace('/', '--')
        paths = [HF_HOME / name, HF_HOME / 'hub' / name]
        datasets[repo] = {'checked_cache_paths': [str(p) for p in paths],
                          'cache_directories_found': [str(p) for p in paths if p.is_dir()],
                          'scope': 'HF_HOME root and hub directories only; not an exhaustive disk search'}
    source_hashes = {}
    for name in ['HANDOFF.md', 'DESIGN.md', 'task.md', 'AGENTS.md', 'CLAUDE.md']:
        p = Path(name)
        source_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    output = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cwd': str(Path.cwd()), 'python': sys.executable, 'python_version': sys.version,
        'torch_installed_in_this_interpreter': importlib.util.find_spec('torch') is not None,
        'configured_hf_home': str(HF_HOME),
        'configured_pip_cache_dir': '/mnt/data2/users/lang.gao/pip_cache',
        'hf_token_file_exists': (HF_HOME / 'token').is_file(),
        'hf_token_contents_read': False, 'gemma_access_permission_verified': False,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_device_nodes': glob.glob('/dev/nvidia*'),
        'nvidia_smi': command(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv']),
        'disk': disk, 'df': command(['df', '-h', '/', '/mnt/data2']),
        'uv': command(['uv', '--version']), 'dns': dns,
        'models': {repo: snapshot(repo) for repo in ['google/gemma-3-1b-it', 'Qwen/Qwen3-1.7B']},
        'datasets': datasets, 'project_input_sha256': source_hashes,
        'interpretation': 'Device visibility and DNS results apply to this execution session; host-wide failure is not established.'
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
