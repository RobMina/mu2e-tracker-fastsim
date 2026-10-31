#!/bin/bash
# Compare saved good-only EventNtuples without fitting or changing their event populations.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CORE=${M2FS_CORE:-${M2FS_CODE:?set M2FS_CORE, or M2FS_CODE to the mu2e-fastsim checkout}/core}
export PYTHONPATH="$HERE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}
exec "$PY" -m trkfastsim.reco.autonomous_compare "$@"
