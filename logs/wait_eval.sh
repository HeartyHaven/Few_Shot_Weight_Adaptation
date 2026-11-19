#!/bin/bash
# Start the full-precision Qwen3-4B MedCalc baseline when some GPU has >= 24 GB free.
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
while :; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1); gi=${g%%,*}; gf=${g##*, }
  [ "$gf" -ge 24000 ] && break; sleep 120
done
CUDA_VISIBLE_DEVICES=$gi TDWA_MEM_GB=22 TDWA_BS=8 /mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python -m tdwa.evaluate --base --task t1 --student Qwen/Qwen3-4B > logs/eval_base_t1_Qwen3-4B_fp32.log 2>&1
