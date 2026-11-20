#!/bin/bash
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
PY=/mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python
$PY -m tdwa.eval_fast runs/t1__Qwen3-4B__gpt-6.2-astra__s0 --gpu 1 --mem 0.5 --b1 > logs/evalfast_t1__Qwen3-4B__gpt-6.2-astra__s0--b1.log 2>&1
$PY -m tdwa.eval_fast runs/t1__Qwen3-4B__gpt-6-astra__s0 --gpu 1 --mem 0.5 > logs/evalfast_t1__Qwen3-4B__gpt-6-astra__s0.log 2>&1
$PY -m tdwa.eval_fast --base --task t1 --student Qwen/Qwen3-4B --gpu 1 --mem 0.5 --b1 > logs/evalfast_base_t1_qwen4b_b1.log 2>&1
