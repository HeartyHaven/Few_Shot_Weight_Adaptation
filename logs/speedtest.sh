#!/bin/bash
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
PY=/mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python
for mode in default expandable; do
  [ $mode = expandable ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  timeout 3000 $PY -m tdwa.teach --task t1 --student Qwen/Qwen3-8B --teacher selftest --seed 9$([ $mode = default ] && echo 0 || echo 1) --gpu 0 --dtype bfloat16 --exec-timeout 2800 --scripted /tmp/claude-1023/-mnt-data2-users-lang-gao-Few-Shot-Weight-Adaptation/b518130e-29c1-44de-8bf0-025e2942dae3/scratchpad/st/speed.py > logs/speedtest_$mode.log 2>&1
done
