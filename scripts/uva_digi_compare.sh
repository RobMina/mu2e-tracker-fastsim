#!/bin/bash
# Paired closure from persisted fast digi arms, with strict full-population admission.
set -euo pipefail
CODE=${CODE:?core and tracker snapshot directory}
OUT=${OUT:?new output directory}
RUN=${RUN:?directory with ten_jax_cpu, ten_numpy and ten_floor}
INPUTS=${INPUTS:?shared input root}
SIF=${SIF:?project image}
F2=${F2:-0}
FAST=${FAST:-$RUN/ten_jax_cpu}
REF=${REF:-$RUN/ten_numpy}
FLOOR=${FLOOR:-$RUN/ten_floor}
mkdir "$OUT"
module load apptainer
extra=()
if [[ $F2 == 1 ]]; then extra=(--f2 --f2-chunk 50 --f2-fixed-shapes --f2-progress); fi
apptainer exec --cleanenv --env "JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONUNBUFFERED=1,PYTHONPATH=$CODE/tracker:$CODE/core" \
  -B "$CODE" -B "$INPUTS:$INPUTS:ro" -B "$RUN" --pwd "$CODE/tracker" "$SIF" \
  python3 -m trkfastsim digicmp "$FAST/"*.root --ref "$REF/"*.root --floor "$FLOOR/"*.root \
  --hit-cond "$INPUTS/tracker_digi_reference_v2_20261026/v2/cmp/hitcond.json" \
  --drift-source "$INPUTS/tracker_digi_reference_v2_20261026/ref/strawhits_ref.v18L_1001.root" \
  --require-same-events "${extra[@]}" --outdir "$OUT"
