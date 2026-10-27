#!/bin/bash
# Stock digitisation reference for the fast digitiser: path A (gen art + fast step table) -> dig art -> StrawHitTableDumper.
# Prints the StrawPhysics/StrawElectronics conditions once (makeSD printLevel 1, strawElectronics verbose 1) and the
# TimeTracker per-module summary, so makeSD's time per event can be read from the digi log. Needs a BUILT TrkFastSim area.
# usage: run_digi_ref.sh <gen.art> <fasttrk.root> <outdir> <tag> [nevents, -1 = all]
# env:   TRKFS_ART  built area (default /exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/tracker/art)
#        SEED       SeedService.baseSeed (default: the job's, 8); another seed gives the stock-vs-stock floor
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
AREA=${TRKFS_ART:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/tracker/art}
cd "$AREA" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
GEN=${1:?gen.art}; TRK=${2:?fasttrk.root}; O=${3:?outdir}; TAG=${4:?tag}; N=${5:--1}
mkdir -p "$O/fcl" "$O/logs" || exit 2
printf '%s\n' '#include "TrkFastSim/fcl/digi_from_fasttable_alltrig.fcl"' \
  "physics.producers.compressDetStepMCs.trkTableFile : \"$TRK\"" \
  'physics.producers.makeSD.printLevel : 1' \
  'services.ProditionsService.strawElectronics.verbose : 1' \
  'services.TimeTracker.printSummary : true' ${SEED:+"services.SeedService.baseSeed : $SEED"} > "$O/fcl/digi_$TAG.fcl"
DIG=$O/dig.$TAG.art
T0=$(date +%s)
FHICL_FILE_PATH="$O/fcl:$FHICL_FILE_PATH" mu2e -c "$O/fcl/digi_$TAG.fcl" -s "$GEN" -o "$DIG" -n "$N" > "$O/logs/digi_$TAG.log" 2>&1
echo "digi rc=$? $(( $(date +%s) - T0 )) s $(grep -m1 'Events total' "$O/logs/digi_$TAG.log")"
T1=$(date +%s)
mu2e -c TrkFastSim/fcl/dump_strawhits_mc.fcl -s "$DIG" -T "$O/strawhits_ref.$TAG.root" > "$O/logs/dump_$TAG.log" 2>&1
echo "dump rc=$? $(( $(date +%s) - T1 )) s $(grep -m1 'Events total' "$O/logs/dump_$TAG.log")"
