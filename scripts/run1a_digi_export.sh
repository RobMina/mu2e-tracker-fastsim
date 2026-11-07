#!/usr/bin/env bash
# Export existing stock digis and their response inputs using an already-built area.
set -e
area=$1
input=$2
config=$3
out=$4
count=${5:-1000}
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
mkdir "$out"
cd "$out"
fhicl-dump -c "$config" > expanded.fcl
for table in TrkPreampStraw TrkAlignStrawSim TrkDelayPanel TrkDelayRStraw; do
  dbTool print-run --purpose Sim_best --version v1_1 --run 1430:0 --table "$table" --content > "db_$table.csv"
done
cp "$MUSE_WORK_DIR/backing/Offline/GlobalConstantsService/data/globalConstants_01.txt" globalConstants.txt
/usr/bin/time -v -o export.time mu2e -c "$code/art/TrkFastSim/fcl/run1a_digi_reference.fcl" -s "$input" -n "$count" > export.log 2>&1
printf 'finished\n' > export.done
