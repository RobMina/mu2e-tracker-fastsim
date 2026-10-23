#!/bin/bash
set -u
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
V=${M2FS_VENV:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf}/bin/python
cd $M2FS_CODE && export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker
for FL in cov info; do F1_FILTER=$FL $V tracker/scripts/f1_parts.py --tracks 8192 --hits 32 --xings 64 | sed "s/^/$FL /"; done
echo "parts done"
