#!/bin/bash
# Dump the grid gen-only art files of one campaign to gen tables (gen.001400_<subrun>.root), serially on one GPVM
# (~30 s per 500 k-event file). Reads the art files from their dCache scratch paths, as dump_ref.sh does.
# usage: dump_gen_only.sh TAG FIRST_SUBRUN LAST_SUBRUN    (needs site.env: TRKFS_CODE, TRKFS_PNFS, TRKFS_OWNER, M2FS_DATA)
OUT_TAG=${1:?tag}; S0=${2:?first subrun}; S1=${3:?last subrun}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${TRKFS_CODE:?source site.env}/art" && { setup muse; muse setup; } >/dev/null 2>&1 || true
set -u    # only after the mu2e/muse setup, which reads unset variables
[ "${MUSE_WORK_DIR:-}" = "$TRKFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=${MUSE_WORK_DIR:-unset}"; exit 2; }
IN=$TRKFS_PNFS/$OUT_TAG
O=$M2FS_DATA/trk_fastsim/$OUT_TAG
mkdir -p "$O/logs"
for ((s = S0; s <= S1; s++)); do
  SEQ=$(printf "001400_%08d" "$s"); T="$O/gen.$SEQ.root"
  [ -s "$T" ] && { echo "$SEQ exists"; continue; }
  mu2e -c TrkFastSim/fcl/dump_sgsteps_gen.fcl -s "$IN/gen.${TRKFS_OWNER}.CosmicCRYExtractedGen.v0.$SEQ.art" -T "$T" \
    > "$O/logs/dump.$SEQ.log" 2>&1
  echo "$SEQ rc=$? $(date +%T)"
done
echo "done $(date -Is)"
