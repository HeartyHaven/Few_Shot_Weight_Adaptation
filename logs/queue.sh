#!/bin/bash
# Launch queued teaching runs one at a time when the shared machine has room.
# queue.txt lines: mod task student teacher seed [gpu_mb_needed]   (mod = api | mcp)
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation
PY=/mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python
set -a; . /mnt/data2/users/lang.gao/Weight_Programming_Eval/.secrets/cliproxy.env; set +a
while read -r mod task student teacher seed need; do
  [ -z "$mod" ] && continue; need=${need:-12000}
  name="${task}__$(basename $student)__${teacher}__s${seed}"; [ -d runs/$name ] && continue
  while :; do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); n=$(pgrep -fc "tdwa/kernel.py")
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1); gi=${g%%,*}; gf=${g##*, }
    # a sandbox can grow to its cap later, so a large student also needs no other large-student sandbox on that GPU
    big=0; for p in $(pgrep -f "tdwa/kernel.py"); do c=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'); m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$p '$1==p{print $2}'); [ "$c" = "$gi" ] && [ "${m:-0}" -gt 15000 ] && big=1; done
    [ "$need" -le 20000 ] && big=0
    [ "$avail" -ge 16 ] && [ "$n" -lt 6 ] && [ "$gf" -ge "$need" ] && [ "$big" = 0 ] && break; sleep 60
  done
  to=1200; [ "$need" -gt 20000 ] && to=2400
  echo "$(date +%F_%T) launch $mod $task $student $teacher s$seed gpu$gi avail=${avail}G gpufree=${gf}" >> logs/queue.log
  if [ "$mod" = api ]; then nohup setsid $PY -m tdwa.teach --task $task --student $student --teacher $teacher --seed $seed --gpu $gi --effort high --exec-timeout $to > logs/${task}_$(basename $student)_${teacher}_s$seed.log 2>&1 &
  else nohup setsid $PY -m tdwa.mcp_teach run --task $task --student $student --teacher $teacher --seed $seed --gpu $gi --exec-timeout $to > logs/${task}_$(basename $student)_${teacher}_s$seed.log 2>&1 & fi
  sleep 240
done < logs/queue.txt
echo "$(date +%F_%T) queue empty" >> logs/queue.log
