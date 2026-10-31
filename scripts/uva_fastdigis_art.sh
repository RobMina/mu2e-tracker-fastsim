#!/bin/bash
# Run stock reconstruction on admitted fast digis with the original event context.
set -euo pipefail
ROOT=${ROOT:?task root}
INPUT_ROOT=${INPUT_ROOT:?read-only original art input root}
CVMFSEXEC_DIST=${CVMFSEXEC_DIST:?saved EL8 CVMFS client}
export CVMFSEXEC_DIST
OUT="$ROOT/${TAG:?new output tag}"
mkdir "$OUT"
cd "$OUT"
module load apptainer
export OFFLINE_FCL_PATH="$ROOT/runtime/fcl"
export OFFLINE_BINDS="$ROOT:$ROOT"
export OFFLINE_LOG_PREFIX=fast
printf '#include "TrkFastSim/fcl/reco_digis_table_mc.fcl"\nphysics.producers.tableSD.tableFile : "%s/fast_input/digis.root"\nphysics.analyzers.trkdigi : { module_type : StrawDigiTableDumper StrawDigiTag : "tableSD" }\nphysics.DumpPath : [ trkdump, trkdigi ]\n' "$ROOT" > fast.fcl
bash "$ROOT/runtime/offline_fast.sh" "$ROOT/built_area" "$OUT/fast.fcl" \
  "$INPUT_ROOT/trkfs_digi_20261026/ref/dig.v18L_1001.art" > runtime.log 2>&1
