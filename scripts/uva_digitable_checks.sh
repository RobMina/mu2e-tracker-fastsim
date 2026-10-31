#!/bin/bash
# Validate only this snapshot's change-selected tests in the existing project image.
set -euo pipefail
ROOT=${ROOT:?snapshot/output root}
SIF=${SIF:?project image}
INPUT=${INPUT:?shared read-only tracker digi reference root}
TESTS=${TESTS:?laptop testselect output from this exact source snapshot}
CHECKS=${CHECKS:?new output directory under ROOT}
case "$CHECKS" in "$ROOT"/*) ;; *) echo 'CHECKS must be under ROOT' >&2; exit 2 ;; esac
mkdir "$CHECKS"
module load apptainer
cd "$ROOT/code/tracker"
cp "$TESTS" "$CHECKS/selection.txt"
test -s "$CHECKS/selection.txt"
while IFS= read -r testfile; do
  apptainer exec --cleanenv \
    --env "JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$ROOT/code/tracker:$ROOT/code/core" \
    -B "$ROOT" "$SIF" python3 -m pytest -q -p no:cacheprovider "$testfile"
done < "$CHECKS/selection.txt"
apptainer exec --cleanenv \
  --env "PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$ROOT/code/tracker:$ROOT/code/core" \
  -B "$ROOT" -B "$INPUT:$INPUT:ro" "$SIF" python3 -m trkfastsim.digitable \
  convert "$INPUT/ref/strawhits_ref.v18L_1001.root" "$CHECKS/legacy_mc.root" \
  --assume-zero-flags --report "$CHECKS/legacy_mc_conversion.json"
