#!/bin/bash
# Reuse immutable single-target traces on one CPU; run on an otherwise idle host.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout (core/ is put on PYTHONPATH)}"
: "${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}"
F1_CAPTURE_AREA=${TRKFS_MUSE_AREAS:?set TRKFS_MUSE_AREAS to the directory holding the muse work areas}/fastreco_capture_p109_20261026
DATA=${M2FS_DATA:?set M2FS_DATA to the data area holding the saved campaign inputs}
OUT=$1; shift
if test -e "$OUT"; then echo "refusing existing measurement audit" >&2; exit 2; fi
export PYTHONPATH="$HERE:$M2FS_CODE/core"
export MU2E_SEARCH_PATH=$F1_CAPTURE_AREA
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
cd "$HERE"
taskset -c "$CORE" "$PY_JAX" scripts/f1_measurement_audit.py \
  --check "$DATA/trk_fastreco/p109_iteration_3k_20261027/check.json" \
  --native-source "$F1_CAPTURE_AREA" --out "$OUT" "$@"
