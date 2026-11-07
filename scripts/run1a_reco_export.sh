#!/usr/bin/env bash
# Export stock hits from saved digis using the existing built dumper.
set -e
area=$1
input=$2
out=$3
count=${4:-1000}
skip=${5:-0}
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
printf '#include "run1a_reco_reference.fcl"\nsource.skipEvents: %s\n' "$skip" > job.fcl
fhicl-dump -c job.fcl > expanded.fcl
/usr/bin/time -v -o export.time mu2e -c job.fcl -s "$input" -n "$count" > export.log 2>&1
printf 'finished\n' > export.done
