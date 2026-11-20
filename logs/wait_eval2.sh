#!/bin/bash
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
while :; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1); gi=${g%%,*}; gf=${g##*, }; [ "$gf" -ge 24000 ] && break; sleep 120; done
CUDA_VISIBLE_DEVICES=$gi TDWA_MEM_GB=22 TDWA_BS=4 /mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python -m tdwa.evaluate --run runs/t1__Qwen3-4B__claude-fable-5-2__s0 > logs/eval_t1_qwen4b_fable_s0.log 2>&1
