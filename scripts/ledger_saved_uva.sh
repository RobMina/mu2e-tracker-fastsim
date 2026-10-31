#!/usr/bin/env bash
set -euo pipefail
R=${1:?output root}
I=${2:?shared input root}
cd "$R/code/tracker"
python3 -m trkfastsim.analyses.ratearms compare \
  --g4 "$I/tracker_rate_ledger/g4.parquet" \
  --arm "legacy=$I/tracker_rate_ledger/legacy" --base legacy --out "$R/evidence/rate_current_analysis.json"
python3 scripts/transport_capture_audit.py "$I/tracker_capture_50k/capture_50k_AMT_v1" --out "$R/evidence/capture_AMT_audit.json"
python3 scripts/transport_capture_audit.py "$I/tracker_capture_50k/capture_50k_legacy_v1" --out "$R/evidence/capture_legacy_audit.json"
python3 scripts/secondary_occupancy_jackknife.py \
  "$I/tracker_tv3_default_l/migration_moments_v1/migration.json" \
  "$I/tracker_tv3_default_l/migration_moments_v1/occupancy_groups.npz" \
  --out "$R/evidence/migration_current_analysis.json"
