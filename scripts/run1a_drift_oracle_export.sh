#!/usr/bin/env bash
# Query stock drift response after the user builds FieldOnDriftOracle in the supplied area.
set -e
area=$1
input=$2
out=$3
count=${4:-10}
code=$(cd "$(dirname "$0")/.." && pwd)
cd "$area"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup muse
set +e
muse setup
setup_rc=$?
set -e
test "$setup_rc" -eq 0
test "$MUSE_WORK_DIR" = "$area"
set -u
export FHICL_FILE_PATH="$code/art/TrkFastSim/fcl:$FHICL_FILE_PATH"
mkdir "$out"
cd "$out"
printf '#include "run1a_drift_oracle.fcl"\n' > job.fcl
fhicl-dump -c job.fcl > expanded.fcl
backing=$(readlink -f "$area/backing")
sha256sum expanded.fcl "$code/art/TrkFastSim/src/FieldOnDriftOracle_module.cc" \
  "$backing/Offline/TrackerConditions/src/StrawDrift.cc" \
  "$backing/Offline/TrackerConditions/src/StrawDriftMaker.cc" \
  "$backing/Offline/TrackerConditions/src/StrawResponse.cc" > source-sha256.txt
printf '%s\n' "$backing" > backing.txt
/usr/bin/time -v -o export.time mu2e -c job.fcl -s "$input" -n "$count" > export.log 2>&1
printf 'finished\n' > export.done
