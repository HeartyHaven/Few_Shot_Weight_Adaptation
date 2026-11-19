#!/bin/bash
# Baselines of larger students (no teaching): 3 candidate new tasks + hidden tests of T1 and T2. Runs one model at a time on a GPU
# with enough free memory; teaching runs keep priority (this script only waits).
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
PY=/mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python
export HF_HOME=/mnt/data2/users/lang.gao/huggingface_cache
# a GPU qualifies if it has enough free memory and no large teaching sandbox (those grow to ~26 GB)
bigon() { for p in $(pgrep -f "tdwa/kernel.py"); do c=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'); m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$p '$1==p{print $2}'); [ "${m:-0}" -gt 15000 ] && echo $c; done; }
pick() { local busy=" $(bigon | tr '\n' ' ') "; nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v need=$1 -v busy="$busy" '$2>=need && index(busy, " "$1" ")==0 {print $1; exit}'; }
while read -r repo need bs; do
  [ -z "$repo" ] && continue; tag=$(basename $repo)
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', cache_dir='$HF_HOME', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])" >> logs/bigpilot.log 2>&1 || { echo "$(date +%T) DOWNLOAD FAILED $repo" >> logs/bigpilot.log; continue; }
  while :; do g=$(pick $need); avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ -n "$g" ] && [ "$avail" -ge 20 ] && break; sleep 120; done
  echo "$(date +%T) start $repo on gpu$g" >> logs/bigpilot.log
  cap=$((need/1000))        # hard cap on this job's GPU memory (GB) so it cannot crowd out teaching runs
  [ -f pilot/out2/summary__$tag.json ] || CUDA_VISIBLE_DEVICES=$g PILOT_BS=$bs TDWA_MEM_GB=$cap $PY pilot/scripts/baseline2.py $repo > pilot/logs/baseline2_$tag.log 2>&1
  for t in t1 t2; do [ -f eval/base__${t}__${tag}__bf16.json ] || CUDA_VISIBLE_DEVICES=$g TDWA_DTYPE=bfloat16 TDWA_BS=$bs TDWA_MEM_GB=$cap $PY -m tdwa.evaluate --base --task $t --student $repo > logs/eval_base_${t}_$tag.log 2>&1; done
  echo "$(date +%T) done $repo" >> logs/bigpilot.log
done < logs/bigpilot.txt
