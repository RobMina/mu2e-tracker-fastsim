#!/bin/bash
# Run preserved initial operands on one CPU; run on an otherwise idle host.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}"
F1_CAPTURE_AREA=${TRKFS_MUSE_AREAS:?set TRKFS_MUSE_AREAS to the directory holding the muse work areas}/fastreco_capture_p109_20261026
DATA=${M2FS_DATA:?set M2FS_DATA to the data area holding the saved campaign inputs}
OUT=$1
if test -e "$OUT"; then echo "refusing existing residual audit" >&2; exit 2; fi
export PYTHONPATH="$HERE:$DATA/trk_f1native_r2_20261027/deps-core-233771b"
export MU2E_SEARCH_PATH=$F1_CAPTURE_AREA
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
cd "$HERE"
taskset -c "$CORE" "$PY_JAX" scripts/f1_initial_residual_audit.py \
  --ca-report "$DATA/trk_f1native_r2_20261027/captured_ca_population_v2.json" \
  --native-source "$F1_CAPTURE_AREA" --out "$OUT"
