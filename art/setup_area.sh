#!/bin/bash
# Make art/ a muse work area for the TrkFastSim package, backed by the CVMFS Musing (no muse commands: the user builds).
# TrkFastSim must be a real directory here: muse's SConscript walk skips symlinked repos.
# EventNtuple v06_15_00 (2026-10-01) is the release that matches the backing Offline v13_39_00 (2026-09-30), for path A ntuples.
set -e
AREA="${TRKFS_CODE:?set TRKFS_CODE to this repository}/art"
EVENTNTUPLE_TAG=${EVENTNTUPLE_TAG:-v06_15_00}
ln -sfn /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay "$AREA/backing"
if [ ! -d "$AREA/EventNtuple" ]; then
  git clone -q https://github.com/Mu2e/EventNtuple.git "$AREA/EventNtuple"
  git -C "$AREA/EventNtuple" checkout -q "$EVENTNTUPLE_TAG"
fi
echo "EventNtuple at $(git -C "$AREA/EventNtuple" describe --tags --always)"
ls -la "$AREA"
echo "next (user, on mu2ebuild02):"
echo "  cd $AREA && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32"
