#!/bin/bash
# Reuse immutable single-target traces on one CPU; caller owns host lock.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=/exp/mu2e/data/users/ram2aq
OUT=$1; shift
if test -e "$OUT"; then echo "refusing existing measurement audit" >&2; exit 2; fi
export PYTHONPATH="$HERE:/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/core"
export MU2E_SEARCH_PATH=/exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
cd "$HERE"
taskset -c "$CORE" "$DATA/crv_fastsim/venv_eaf/bin/python3" scripts/f1_measurement_audit.py \
  --check "$DATA/trk_fastreco/p109_iteration_3k_20261027/check.json" \
  --native-source /exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026 --out "$OUT" "$@"
