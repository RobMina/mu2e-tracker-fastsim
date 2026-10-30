#!/bin/bash
# Compare real-table backends on one allocation, including cold and two warm passes.
set -euo pipefail
CODE=${CODE:?core and tracker snapshot directory}
OUT=${OUT:?new output directory}
INPUTS=${INPUTS:?shared input root}
SIF=${SIF:?project image}
DEVICE=${DEVICE:-cpu}
mkdir "$OUT"
export LIMIT=0 REPEATS=3
export SHARDS=${SHARDS:-21}
if [[ $DEVICE == gpu ]]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/device.txt"
fi
for backend in jax numpy; do
  CODE="$CODE" INPUTS="$INPUTS" SIF="$SIF" DEVICE="$DEVICE" BACKEND="$backend" OUT="$OUT/$backend" \
    bash "$CODE/tracker/scripts/uva_digi_cert.sh"
done
