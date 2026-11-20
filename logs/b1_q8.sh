#!/bin/bash
# Same-protocol rule-in-prompt control for Qwen3-8B MedCalc (vLLM bf16): base, then base + B1 procedure text.
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
HF=/mnt/data2/users/lang.gao/huggingface_cache; SNAP=$(ls -d $HF/models--Qwen--Qwen3-8B/snapshots/* | head -1)
export CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN TDWA_VLLM_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=$HF HF_HUB_CACHE=$HF HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_CACHE_ROOT=$PWD/cache/vllm TOKENIZERS_PARALLELISM=false
V=.venv-vllm/bin/python
$V -m tdwa.evaluate_vllm --task t1 --student Qwen/Qwen3-8B --model-path $SNAP --dtype bfloat16 --mem 0.85 --name base__t1__Qwen3-8B__vllm16 > logs/b1_q8_base.log 2>&1
$V -m tdwa.evaluate_vllm --task t1 --student Qwen/Qwen3-8B --model-path $SNAP --dtype bfloat16 --mem 0.85 --b1 --name base__t1__Qwen3-8B__vllm16__b1 > logs/b1_q8_b1.log 2>&1
