#!/bin/bash
# Single-core detector-only F1 run; inputs and output directory are explicit arguments.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$HERE:${M2FS_CORE:-${M2FS_CODE:?set M2FS_CORE, or M2FS_CODE to the mu2e-fastsim checkout}/core}"
export MU2E_SEARCH_PATH=${MU2E_SEARCH_PATH:-${TRKFS_MUSE_AREAS:?set MU2E_SEARCH_PATH, or TRKFS_MUSE_AREAS to the directory holding the muse work areas}/fastreco_capture_p109_20261026}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
exec taskset -c "$CORE" "$PY" -m trkfastsim.reco.autonomous "$@"
