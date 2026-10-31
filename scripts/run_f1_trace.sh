#!/bin/bash
# Run one explicit diagnostic target on one CPU; output directory must not exist.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout (core/ is put on PYTHONPATH)}"
: "${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}"
F1_CAPTURE_AREA=${TRKFS_MUSE_AREAS:?set TRKFS_MUSE_AREAS to the directory holding the muse work areas}/fastreco_capture_p109_20261026
DATA=${TRK_FASTRECO_DATA:-${M2FS_DATA:?set M2FS_DATA (or TRK_FASTRECO_DATA) to the data area}/trk_fastreco}
OUT=$1; KEY=$2; shift 2
if test -e "$OUT" || test -e "$OUT.log"; then echo "refusing existing trace output" >&2; exit 2; fi
export PYTHONPATH="$HERE:$M2FS_CODE/core"
export MU2E_SEARCH_PATH=$F1_CAPTURE_AREA
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
cd "$HERE"
taskset -c "$CORE" "$PY_JAX" scripts/f1_trace_capture.py \
  --check "$DATA/p109_iteration_3k_20261027/check.json" \
  --conditions "$DATA/p109_prior_all_20261025/conditions.json" \
  --straw-axis "$DATA/align/TrkAlignStraw.csv" --key "$KEY" --out "$OUT" "$@"
