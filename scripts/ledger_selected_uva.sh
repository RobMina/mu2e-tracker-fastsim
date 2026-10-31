#!/usr/bin/env bash
set -euo pipefail
R=${1:?output root}
SIF=${2:?project image}
SELECTION=${3:?testselect output from the coordinating checkout}
TRACKER=${4:-$R/code/tracker}
PARQUET_ENV=${5:-}
EXTRA_PATH=
if [[ -n "$PARQUET_ENV" ]]; then
  source "$PARQUET_ENV/env.sh"
  EXTRA_PATH="$FASTSIM_PARQUET_SITE_PACKAGES:"
fi
cd "$TRACKER"
cat "$SELECTION"
status=0
while IFS= read -r file; do
  [[ "$file" == tests/*.py ]] || continue
  echo "==== $file"
  apptainer exec --cleanenv \
    --env "JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$EXTRA_PATH$TRACKER:$R/code/core:$R/code/crv,MU2E_SEARCH_PATH=$R/code/offline_search" \
    -B "$R" "$SIF" python3 -m pytest "$file" -q -p no:cacheprovider || status=1
done < "$SELECTION"
exit "$status"
