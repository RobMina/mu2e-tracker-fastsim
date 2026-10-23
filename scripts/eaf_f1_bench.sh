#!/bin/bash
# F1 throughput on an EAF GPU slice: toy tracks; inverse kind, track size (hits/crossings padding) and batch size.
set -u
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
V=${M2FS_VENV:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf}/bin/python
cd $M2FS_CODE && export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker
nvidia-smi -L
B=tracker/scripts/f1_bench.py
for INV in lu chol; do F1_INV=$INV $V $B --tracks 16384 --repeat 2 | sed "s/^/inv $INV /"; done
for NM in "32 48" "16 32"; do set -- $NM
  for T in 16384 32768; do $V $B --tracks $T --hits $1 --xings $2 --repeat 2; done
done
echo "bench done"
