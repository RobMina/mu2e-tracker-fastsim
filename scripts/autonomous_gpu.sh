#!/bin/bash
# Detector-only F1 on an already allocated GPU; no server start, installation, or build.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/exp/mu2e/app/users/ram2aq/analysis
export PYTHONPATH="$HERE:$ROOT/mu2e-fastsim/core"
export MU2E_SEARCH_PATH=${MU2E_SEARCH_PATH:-/exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=${PY_JAX:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3}
nvidia-smi -L
"$PY" -c "import jax, sys; print(jax.__version__, jax.devices()); sys.exit(jax.default_backend() != 'gpu')"
exec "$PY" -m trkfastsim.reco.autonomous "$@"
