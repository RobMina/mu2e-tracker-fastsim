#!/bin/bash
# Replay a contiguous tail of an existing fast-step campaign, keeping original file indices.
set -euo pipefail
if [ "$#" -ne 7 ]; then
  echo "usage: $0 BASE_DIR ORIGINAL_LIB ANNOTATED_LIB GDML OUTDIR SEED START_INDEX" >&2
  exit 2
fi
BASE=$1; LIB=$2; ANNOTATED=$3; GDML=$4; OUT=$5; SEED=$6; START=$7
mapfile -t FILES < <(printf '%s\n' "$BASE"/fasttrk*.root | LC_ALL=C sort)
if [ "$START" -lt 0 ] || [ "$START" -ge "${#FILES[@]}" ]; then
  echo "invalid campaign start index" >&2
  exit 2
fi
python3 "$(dirname "$0")/secondary_replay.py" "${FILES[@]:$START}" --start-index "$START" --seed "$SEED" \
  --library "$LIB" --annotated "$ANNOTATED" --gdml "$GDML" --outdir "$OUT"
