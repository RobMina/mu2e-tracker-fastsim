#!/bin/bash
# Compare saved good-only EventNtuples without fitting or changing their event populations.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CORE=${M2FS_CORE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/core}
export PYTHONPATH="$HERE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY_JAX:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3}
exec "$PY" -m trkfastsim.reco.autonomous_compare "$@"
