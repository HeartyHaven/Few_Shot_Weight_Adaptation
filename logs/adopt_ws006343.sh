#!/bin/bash
# adopt jobs started by the first scheduler (no .rc files): write rc when the process ends
cd /mnt/data/users/Lang_Gao/proj/My_Proj/Few_Shot_Weight_Adaptation
id=$1; pid=$2
while kill -0 $pid 2>/dev/null; do sleep 30; done
[ -f baselines/$id/config.json ] && echo 0 > logs/job_train_$id.rc || echo 1 > logs/job_train_$id.rc
