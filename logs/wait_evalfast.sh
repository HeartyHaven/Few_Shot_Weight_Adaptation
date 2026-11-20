#!/bin/bash
# usage: logs/wait_evalfast.sh runs/<name> <gpu_mb_needed> <mem_fraction>
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
while :; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1); gi=${g%%,*}; gf=${g##*, }; [ "$gf" -ge "$2" ] && break; sleep 120; done
/mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python -m tdwa.eval_fast "$1" --gpu $gi --mem "$3" $4 > logs/evalfast_$(basename $1)$4.log 2>&1
