#!/bin/bash
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
while pgrep -f "wait_evalfast.sh|eval_fast runs" >/dev/null; do sleep 60; done
while :; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1); gi=${g%%,*}; gf=${g##*, }; [ "$gf" -ge 30000 ] && break; sleep 120; done
SP=/tmp/claude-1023/-mnt-data2-users-lang-gao-Few-Shot-Weight-Adaptation/b518130e-29c1-44de-8bf0-025e2942dae3/scratchpad/st
timeout 2400 /mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python -m tdwa.teach --task t1 --student Qwen/Qwen3-8B --teacher selftest --seed 98 --gpu $gi --dtype bfloat16 --exec-timeout 2400 --scripted $SP/bf16.py > logs/selftest_bf16_q8.log 2>&1
