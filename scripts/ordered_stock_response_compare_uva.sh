#!/usr/bin/env bash
# Compare existing stock consumer outputs with unchanged NumPy response on the same selected ordered steps.
set -euo pipefail
R=${1:?output root}
STOCK=${2:?completed bounded stock-chain directory}
OUT=${3:?new output directory}
[[ -f "$STOCK/COMPLETE" && -s "$STOCK/coverage.json" ]] || { echo 'stock coverage not complete'; exit 2; }
: "${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH to the project work area on the Slurm cluster (holds inputs/ and mu2efastsim.sif)}"
SIF=${SIF:-$FASTSIMKIT_SCRATCH/mu2efastsim.sif}
module load apptainer
mkdir "$OUT"
C=$FASTSIMKIT_SCRATCH/inputs/tracker_digi_reference_v2_20261026
T=${4:-$R/code/tracker_continuation_v34}
TABLE=${5:-$R/evidence/full_midpoint1000_v26/ordered_steps.root}
sha256sum "$TABLE" "$STOCK/strawhits.root" "$STOCK/ntuple.root" "$STOCK/coverage.json" "$C/v2/digicond.npz" "$C/v2/cmp/hitcond.json" "$C/ref/strawhits_ref.v18L_1001.root" > "$OUT/input.sha256"
E="JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$T:$R/code/core:$R/code/crv,MU2E_SEARCH_PATH=$R/code/offline_search"
cd "$T"
run() { apptainer exec --cleanenv --env "$E" -B "$R" "$SIF" python3 "$@"; }
run -m trkfastsim digi "$TABLE" --cond "$C/v2/digicond.npz" --events "$STOCK/strawhits.root" --seed 1947 --chunk 64 --outdir "$OUT/digi"
run -m trkfastsim digicmp "$OUT/digi/fastdigi.ordered_steps.root" --ref "$STOCK/strawhits.root" --hit-cond "$C/v2/cmp/hitcond.json" --drift-source "$C/ref/strawhits_ref.v18L_1001.root" --f2 --f2-chunk 64 --f2-fixed-shapes --require-same-events --outdir "$OUT/compare"
sha256sum -c "$OUT/input.sha256"
printf '%s\n' 'Bounded same-selected-event consumer diagnostic only; no stock-vs-stock floor or precision equivalence claim.' > "$OUT/scope.txt"
touch "$OUT/COMPLETE"
