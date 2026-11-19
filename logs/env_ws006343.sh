# per-host environment for ws006343 (2x 49 GB, driver CUDA 12.2 -> use the vLLM venv's torch cu128).
# All large files and caches live on the data disk under the project (cache/ is git-ignored); never write to the system disk.
P=/mnt/data/users/Lang_Gao/proj/My_Proj/Few_Shot_Weight_Adaptation; C=$P/cache
export HF_HOME=$C/hf TDWA_HF=$C/hf/hub HF_DATASETS_CACHE=$C/hf/datasets HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton PIP_CACHE_DIR=$C/pip XDG_CACHE_HOME=$C/xdg TMPDIR=/mnt/data/users/Lang_Gao/.tmp_tdwa
export PY=$P/.venv-vllm/bin/python VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
