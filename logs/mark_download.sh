#!/bin/bash
# touch <model cache dir>/.complete once the download log shows the snapshot path (scheduler wait_file)
log=$1; repo=$2; dir=$3
cd /mnt/data/users/Lang_Gao/proj/My_Proj/Few_Shot_Weight_Adaptation
while ! grep -q "$repo /mnt" "$log"; do sleep 30; done; touch "$dir/.complete"
