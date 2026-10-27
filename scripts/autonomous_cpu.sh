#!/bin/bash
# Single-core detector-only F1 run; inputs and output directory are explicit arguments.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/exp/mu2e/app/users/ram2aq/analysis
export PYTHONPATH="$HERE:$ROOT/mu2e-fastsim/core"
export MU2E_SEARCH_PATH=${MU2E_SEARCH_PATH:-/exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=${PY_JAX:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3}
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
exec taskset -c "$CORE" "$PY" -m trkfastsim.reco.autonomous "$@"
