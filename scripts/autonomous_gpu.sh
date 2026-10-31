#!/bin/bash
# Detector-only F1 on an already allocated GPU; no server start, installation, or build.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$HERE:${M2FS_CORE:-${M2FS_CODE:?set M2FS_CORE, or M2FS_CODE to the mu2e-fastsim checkout}/core}"
export MU2E_SEARCH_PATH=${MU2E_SEARCH_PATH:-${TRKFS_MUSE_AREAS:?set MU2E_SEARCH_PATH, or TRKFS_MUSE_AREAS to the directory holding the muse work areas}/fastreco_capture_p109_20261026}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}
nvidia-smi -L
"$PY" -c "import jax, sys; print(jax.__version__, jax.devices()); sys.exit(jax.default_backend() != 'gpu')"
exec "$PY" -m trkfastsim.reco.autonomous "$@"
