#!/bin/bash
# mark the 27B download complete (scheduler wait_file) once snapshot_download returns
cd /mnt/data/users/Lang_Gao/proj/My_Proj/Few_Shot_Weight_Adaptation
until grep -q "^google/gemma-3-27b-it /" logs/download_g27.log; do sleep 60; pgrep -f "snapshot_download('google/gemma-3-27b-it'" >/dev/null || break; done
grep -q "^google/gemma-3-27b-it /" logs/download_g27.log && touch cache/hf/hub/models--google--gemma-3-27b-it/.complete
