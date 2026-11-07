#!/usr/bin/env bash
# Run one already-prepared arm through the identical stock downstream chain.
set -e
area=$1
out=$2
source_art=$3
cd "$area"
if test -z "${MUSE_WORK_DIR:-}"; then
  source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
  setup muse
  set +e
  muse setup
  setup_rc=$?
  set -e
  test "$setup_rc" -eq 0
fi
set -u
test "$MUSE_WORK_DIR" = "$area"
cd "$out"
test -s transport.root
for product in inj.art dig.art mcs-CH.art nts.root; do
  if test -e "$product"; then echo "Refusing to overwrite $out/$product" >&2; exit 2; fi
done
/usr/bin/time -v -o inject.time mu2e -c TrkFastSim/fcl/tracker_entry_inject.fcl -s "$source_art" > inject.log 2>&1
/usr/bin/time -v -o digi.time mu2e -c TrkFastSim/fcl/tracker_entry_digi.fcl -s inj.art > digi.log 2>&1
/usr/bin/time -v -o reco.time mu2e -c TrkFastSim/fcl/tracker_entry_reco.fcl -s dig.art > reco.log 2>&1
/usr/bin/time -v -o ntuple.time mu2e -c TrkFastSim/fcl/tracker_entry_ntuple.fcl -s mcs-CH.art > ntuple.log 2>&1
printf 'finished\n' > chain.done
