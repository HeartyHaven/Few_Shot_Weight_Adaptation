#!/usr/bin/env bash
set -euo pipefail
wp0_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd -- "$wp0_project"
export HF_HOME=/mnt/data2/users/lang.gao/huggingface_cache
export PIP_CACHE_DIR=/mnt/data2/users/lang.gao/pip_cache
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$wp0_project/outputs/engineer/wp0/tmp"
mkdir -p "$TMPDIR"
df -h / /mnt/data2
uv --no-cache --offline --no-config venv --no-project \
  --no-python-downloads --no-managed-python \
  --python /home/lang.gao/miniconda3/bin/python outputs/engineer/.venv
# Dependencies are deliberately not installed while GPU and network are unavailable.
# No seed packages, network requests, shared-cache writes, or weight copies.
