#!/bin/bash
# Current-code 50k generator point pair; NumPy transport and F2 identical, digitiser selected explicitly.
set -euo pipefail
CODE=${CODE:?core and tracker snapshots}
OUT=${OUT:?new point-pair output directory}
INPUTS=${INPUTS:?shared inputs}
GEOMETRY=${GEOMETRY:?matching extracted v04 GDML}
GEN=${GEN:?complete generator table}
SIF=${SIF:?project image}
mkdir "$OUT"
module load apptainer
for backend in numpy jax; do
  apptainer exec --nv --cleanenv --env "JAX_PLATFORMS=cuda,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$CODE/tracker:$CODE/core,XLA_PYTHON_CLIENT_PREALLOCATE=false" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" -B "$CODE" -B "$INPUTS:$INPUTS:ro" \
    -B "$(dirname "$GEN"):$(dirname "$GEN"):ro" -B "$(dirname "$GEOMETRY"):$(dirname "$GEOMETRY"):ro" -B "$OUT" --pwd "$CODE/tracker" "$SIF" \
    python3 -m trkfastsim.kitpoint --gen "$GEN" --gdml "$GEOMETRY" \
    --digi-cond "$INPUTS/tracker_digi_reference_v2_20261026/v2/digicond.npz" \
    --reco-cond "$INPUTS/tracker_digi_reference_v2_20261026/v2/cmp/hitcond.json" \
    --drift-source "$INPUTS/tracker_digi_reference_v2_20261026/ref/strawhits_ref.v18L_1001.root" \
    --seed 1 --backend "$backend" --retain-ntuple --outdir "$OUT/$backend" > "$OUT/$backend.log" 2>&1
done
apptainer exec --cleanenv -B "$CODE" -B "$OUT" "$SIF" \
  python3 "$CODE/tracker/scripts/kit_point_compare.py" "$OUT/jax" "$OUT/numpy" > "$OUT/compare.json"
