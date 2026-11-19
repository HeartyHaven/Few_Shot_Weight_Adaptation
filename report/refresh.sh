#!/bin/bash
# Rebuild the report data every 10 minutes so the local site (http://127.0.0.1:8765/index.html) stays current.
cd /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation/report
while :; do /mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv/bin/python build_atlas.py > .atlas.log 2>&1; python3 build_roles.py >> .atlas.log 2>&1; python3 build_site.py > .refresh.log 2>&1; sleep 600; done
