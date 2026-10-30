#!/bin/bash
# Run the exact change-selected test list in isolated image processes.
set -euo pipefail
CODE=${CODE:?core and final_tracker snapshot directory}
OUT=${OUT:?new output directory}
SIF=${SIF:?project image}
TESTS=${TESTS:?testselect output from the same source snapshot}
mkdir "$OUT"
module load apptainer
cd "$CODE/final_tracker"
cp "$TESTS" "$OUT/selection.txt"
rc=0
while IFS= read -r testfile; do
  printf '\n== %s\n' "$testfile"
  apptainer exec --cleanenv --env "JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$CODE/final_tracker:$CODE/core" \
    -B "$CODE" -B "$OUT" --pwd "$CODE/final_tracker" "$SIF" python3 -m pytest -q -p no:cacheprovider --runslow "$testfile" || rc=1
done < "$OUT/selection.txt"
exit "$rc"
