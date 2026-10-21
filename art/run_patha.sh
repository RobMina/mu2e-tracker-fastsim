#!/bin/bash
# Path A on one shard: gen-stream art file + fast trk (and CRV) tables -> stock extracted digitisation (needs a BUILT area; no build here).
# usage: run_patha.sh <gen.art> <fasttrk.root> [fastcrv.root|""] [out.art] [base fcl, default digi_from_fasttable.fcl]
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${TRKFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$TRKFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
GEN=${1:?gen.art}; TRK=${2:?fasttrk.root}; CRV=${3:-}; OUT=${4:-dig.fast.art}; BASE=${5:-digi_from_fasttable.fcl}
for f in "$GEN" "$TRK" ${CRV:+"$CRV"}; do [ -s "$f" ] || { echo "missing $f"; exit 2; }; done
W=$(mktemp -d)
printf '#include "TrkFastSim/fcl/%s"\nphysics.producers.compressDetStepMCs.trkTableFile : "%s"\nphysics.producers.compressDetStepMCs.crvTableFile : "%s"\n' \
  "$BASE" "$TRK" "$CRV" > "$W/patha.fcl"
FHICL_FILE_PATH="$W:$FHICL_FILE_PATH" mu2e -c "$W/patha.fcl" -s "$GEN" -o "$OUT"
echo "rc=$? fcl=$W/patha.fcl"
