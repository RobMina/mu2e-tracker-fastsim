#!/bin/bash
# Audit crossing observability on one CPU; caller owns the idle-host lock.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=/exp/mu2e/data/users/ram2aq
OUT=$1
if test -e "$OUT"; then echo "refusing existing crossing observability report" >&2; exit 2; fi
export PYTHONPATH="$HERE:$DATA/trk_f1native_r2_20261027/deps-core-233771b"
export MU2E_SEARCH_PATH=/exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
cd "$HERE"
taskset -c "$CORE" "$DATA/crv_fastsim/venv_eaf/bin/python3" scripts/f1_crossing_observability.py \
  --transition-report "$DATA/trk_f1native_r2_20261027/transitions_v5.json" \
  --native-source /exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026 --out "$OUT"
