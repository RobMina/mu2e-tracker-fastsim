#!/usr/bin/env bash
# Full-census G4 transport followed by the shared stock downstream stages.
set -e
area=$1
out=$2
source_art=$3
code=$(cd "$(dirname "$0")/.." && pwd)
cd "$area"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup muse
set +e
muse setup
setup_rc=$?
set -e
test "$setup_rc" -eq 0
set -u
test "$MUSE_WORK_DIR" = "$area"
export FHICL_FILE_PATH="$(dirname "$out")/prepared:$FHICL_FILE_PATH"
cd "$out"
test ! -e transport.root
mu2e --debug-config expanded-transport.fcl -c TrkFastSim/fcl/tracker_entry_g4_full.fcl -s "$source_art" > config.log 2>&1
/usr/bin/time -v -o transport.time mu2e -c TrkFastSim/fcl/tracker_entry_g4_full.fcl -s "$source_art" > transport.log 2>&1
bash "$code/scripts/entry_chain.sh" "$area" "$out" "$source_art"
