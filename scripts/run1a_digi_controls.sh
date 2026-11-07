#!/usr/bin/env bash
# Read a disjoint saved-digi block and run an independent stock seed on saved injected steps.
set -e
area=$1
base=$2
out=$3
code=$(cd "$(dirname "$0")/.." && pwd)
cd "$area"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup muse
set +e
muse setup
rc=$?
set -e
test "$rc" -eq 0
test "$MUSE_WORK_DIR" = "$area"
set -u
mkdir "$out"
export FHICL_FILE_PATH="$code/art/TrkFastSim/fcl:$FHICL_FILE_PATH"
cd "$out"
mkdir heldout reseed
printf '#include "run1a_digi_reference.fcl"\nsource.skipEvents: 1000\n' > heldout/job.fcl
cd heldout
/usr/bin/time -v -o export.time mu2e -c job.fcl -s "$base/g4-full/dig.art" -n 10000 > export.log 2>&1
cd ../reseed
/usr/bin/time -v -o response.time mu2e -c run1a_digi_reseed.fcl -s "$base/g4-full/inj.art" -n 1000 > response.log 2>&1
cd ..
printf 'finished\n' > controls.done
