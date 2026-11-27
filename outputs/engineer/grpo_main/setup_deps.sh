#!/usr/bin/env bash
# Install the three training libraries next to the scripts, without touching the vLLM environment.
# common.environment() puts this deps/ directory on PYTHONPATH for every worker.
# Versions are the ones the Distill-100 / GRPO-100 results were produced with.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
uv pip install --no-deps --python "$root/.venv-vllm/bin/python" --target "$here/deps" \
  trl==0.25.1 peft==0.17.1 accelerate==1.11.0
PYTHONPATH="$here/deps" "$root/.venv-vllm/bin/python" -c \
  'import trl, peft, accelerate; print("trl", trl.__version__, "peft", peft.__version__, "accelerate", accelerate.__version__)'
