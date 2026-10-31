#!/usr/bin/env bash
set -euo pipefail
R=${1:?output root}
I=${2:?input root}
G=${3:?extracted GDML}
cd "$R/code/tracker"
python3 scripts/transport_capture_audit.py "$I/tracker_capture_50k/capture_50k_AMT_v1" --out "$R/evidence/capture_AMT_audit.json"
python3 scripts/transport_capture_audit.py "$I/tracker_capture_50k/capture_50k_legacy_v1" --out "$R/evidence/capture_legacy_audit.json"
python3 scripts/secondary_occupancy_jackknife.py \
  "$I/tracker_tv3_default_l/migration_moments_v1/migration.json" \
  "$I/tracker_tv3_default_l/migration_moments_v1/occupancy_groups.npz" \
  --out "$R/evidence/migration_current_analysis.json"
python3 scripts/orderedpath_probe.py "$I/tracker_capture_50k/capture_50k_AMT_v1" \
  --gdml "$G" --outdir "$R/evidence/ordered_probe_v1" \
  --tracker-ref 372b6ae3111ed832298b3231eb9d3f4472c9d1b3 \
  --core-ref 5dc004ffb148c88cb0f78c04d620a9a0be469646
