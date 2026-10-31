#!/bin/bash
# Fit the injected stock dump's exact detector digis, then evaluate against its saved stock fits.
set -euo pipefail
ROOT=${ROOT:?task root}
DUMP=${DUMP:?injected stock dump}
CANONICAL=${CANONICAL:?the exact canonical fast digi input}
OUT=${OUT:?new comparison directory}
ANN=${ANN:?matching Offline ANN inputs}
SIF=${SIF:?project image}
mkdir "$OUT"
module load apptainer
apptainer exec --cleanenv \
  --env "JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONPATH=$ROOT/code/tracker:$ROOT/code/core,MU2E_SEARCH_PATH=$ANN" \
  -B "$ROOT" -B "$ANN:$ANN:ro" "$SIF" bash -c '
    set -euo pipefail
    python3 -m trkfastsim.digitable compare "$3" "$1" --report "$2/injection_identity.json"
    python3 -m trkfastsim reco-conditions "$1" --out "$2/conditions.json"
    python3 -m trkfastsim reco-ntuple "$1" --fit stock --settings mc --conditions "$2/conditions.json" --out "$2/stock.root"
    python3 -m trkfastsim.reco.autonomous "$1" --conditions "$2/conditions.json" --drift-source "$1" --settings mc --chunk 32 --outdir "$2/autonomous"
    python3 -m trkfastsim.reco.autonomous_compare "$2/autonomous/EventNtuple.root" "$2/stock.root" --same-digis --out "$2/compare.json"
  ' _ "$DUMP" "$OUT" "$CANONICAL"
