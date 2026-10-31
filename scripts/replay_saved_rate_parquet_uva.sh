#!/usr/bin/env bash
set -euo pipefail
R=${1:?owned output root}
INPUTS=${2:?shared inputs root}
SIF=${3:?existing project image}
ENVIRONMENT=${4:?shared Parquet environment}
TOOLS=${5:?versioned checker and tests directory}
TRACKER=${6:-$R/code/tracker_continuation_v20}
OUT="$R/evidence/parquet_replay_${SLURM_JOB_ID:?run inside Slurm}"
mkdir "$OUT"
module load apptainer
source "$ENVIRONMENT/env.sh"
(
  cd "$ENVIRONMENT"
  sha256sum -c MANIFEST.sha256 > "$OUT/environment_hashes.txt"
)
(
  cd "$INPUTS/tracker_rate_ledger"
  sha256sum -c MANIFEST.sha256 > "$OUT/input_hashes.before.txt"
)
ENV_ARGS="JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$FASTSIM_PARQUET_SITE_PACKAGES:$TRACKER:$R/code/core:$R/code/crv,MU2E_SEARCH_PATH=$R/code/offline_search"
apptainer exec --cleanenv --env "$ENV_ARGS" -B "$R" "$SIF" \
  python3 "$TOOLS/scripts/parquet_environment_check.py" --expected-root "$ENVIRONMENT/site-packages" \
  --out "$OUT/environment_check.json"
(
  cd "$TRACKER"
  apptainer exec --cleanenv --env "$ENV_ARGS" -B "$R" "$SIF" \
    python3 -m pytest tests/test_ratearms_provenance.py -q -p no:cacheprovider
)
apptainer exec --cleanenv --env "$ENV_ARGS" -B "$R" "$SIF" \
  python3 -m pytest "$TOOLS/tests/test_parquet_environment_check.py" -q -p no:cacheprovider
apptainer exec --cleanenv --env "$ENV_ARGS" -B "$R" "$SIF" \
  python3 -m trkfastsim.analyses.ratearms compare \
  --g4 "$INPUTS/tracker_rate_ledger/g4.parquet" \
  --arm "legacy=$INPUTS/tracker_rate_ledger/legacy" --base legacy \
  --out "$OUT/rate_current_analysis.json"
(
  cd "$TRACKER"
  apptainer exec --cleanenv --env "$ENV_ARGS" -B "$R" "$SIF" \
    python3 scripts/rate_residual_summary.py "$OUT/rate_current_analysis.json" \
    --out "$OUT/rate_diagnosis.json"
)
(
  cd "$INPUTS/tracker_rate_ledger"
  sha256sum -c MANIFEST.sha256 > "$OUT/input_hashes.after.txt"
)
diff -u "$OUT/input_hashes.before.txt" "$OUT/input_hashes.after.txt"
sha256sum "$TRACKER/trkfastsim/analyses/ratearms.py" "$TRACKER/scripts/rate_residual_summary.py" \
  "$TOOLS/scripts/parquet_environment_check.py" "$0" > "$OUT/replay_sources.sha256"
printf 'Saved-product replay complete: %s\n' "$OUT"
