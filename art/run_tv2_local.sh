#!/bin/bash
# T-V2/T-V3 chain on one shard interactively (smoke test of grid/grid_tv2.sh): digi -> stock reco -> EventNtuple.
# usage: run_tv2_local.sh fast <gen.art> <fasttrk.root> <outdir> [nevents]  |  run_tv2_local.sh g4 <dts.art> - <outdir> [nevents]
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${TRKFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$TRKFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
ARM=${1:?fast|g4}; IN=${2:?input art}; TAB=${3:?table or -}; O=${4:?outdir}; N=${5:--1}
mkdir -p "$O/fcl" && cd "$O" || exit 2
export FHICL_FILE_PATH="$O/fcl:${CRVFS_FCL_PATH%%:*}:$TRKFS_CODE/art/TrkFastSim/fcl:$FHICL_FILE_PATH"
if [ "$ARM" = fast ]; then
  printf '#include "TrkFastSim/fcl/digi_from_fasttable_alltrig.fcl"\nphysics.producers.compressDetStepMCs.trkTableFile : "%s"\n' "$TAB" > fcl/digi.fcl
else
  printf '#include "TrkFastSim/fcl/digi_g4_alltrig.fcl"\n' > fcl/digi.fcl
fi
T0=$(date +%s)
mu2e -c fcl/digi.fcl -s "$IN" -o dig.$ARM.art -n "$N" > digi.log 2>&1; echo "digi rc=$? $(( $(date +%s) - T0 )) s $(grep -m1 'Events total' digi.log)"
T1=$(date +%s)
mu2e -c reco_kkts_extrap_nofilter_v04.fcl -s dig.$ARM.art --output mcs.$ARM.art > reco.log 2>&1; echo "reco rc=$? $(( $(date +%s) - T1 )) s $(grep -m1 'Events total' reco.log)"
T2=$(date +%s)
mu2e -c ntuple_kkts_extrap_v04.fcl -s mcs.$ARM.art -T nts.$ARM.root > ntuple.log 2>&1; echo "ntuple rc=$? $(( $(date +%s) - T2 )) s $(grep -m1 'Events total' ntuple.log)"
