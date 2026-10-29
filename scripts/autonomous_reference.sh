#!/bin/bash
# Extract matching conditions and write good-only stock fits without selecting input events.
set -e -o pipefail
DUMP=${1:?dump ROOT}; OUT=${2:?new output directory}; SETTINGS=${3:?mc or data}
case "$SETTINGS" in mc|data) ;; *) echo 'settings must be mc or data' >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")/.." && pwd)
CORE=${M2FS_CORE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/core}
export PYTHONPATH="$HERE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=${PY_JAX:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3}
mkdir "$OUT"
"$PY" "$HERE/scripts/autonomous_inventory.py" "$DUMP" > "$OUT/input_inventory.json"
"$PY" -m trkfastsim reco-conditions "$DUMP" --out "$OUT/conditions.json"
"$PY" -m trkfastsim reco-ntuple "$DUMP" --fit stock --settings "$SETTINGS" \
  --conditions "$OUT/conditions.json" --out "$OUT/stock.root"
