#!/usr/bin/env bash
# Grid worker: one shard of CRY primaries without Geant4 transport (gen_only_extracted.fcl), Musing SimJob from CVMFS.
# Shard k = PROCESS: subrun FIRST_SUBRUN+k, baseSeed SEED_OFFSET+subrun (so a subrun always has the same seed, whatever
# subrange a submission covers), NEVTS events. Writes gen.<owner>.<desc>.v0.<seq>.art.
set -o pipefail
OUTDIR="${1:?need OUTDIR (pnfs scratch)}"
NEVTS="${NEVTS:?}"; SEED_OFFSET="${SEED_OFFSET:?}"; FIRST_SUBRUN="${FIRST_SUBRUN:?}"; FCL_PNFS="${FCL_PNFS:?}"
MUSING="${MUSING:-MDC2025ay}"; OWNER="${OWNER:-mu2e}"; DESC="${DESC:-CosmicCRYExtractedGen}"
K="${PROCESS:?}"; SUBRUN=$((FIRST_SUBRUN + K)); SEED=$((SEED_OFFSET + SUBRUN))
SEQ=$(printf "001400_%08d" "$SUBRUN")
GEN="gen.${OWNER}.${DESC}.v0.${SEQ}.art"
echo "=== grid_gen_only: shard $K subrun $SUBRUN seed $SEED nevts $NEVTS musing $MUSING host $(hostname) $(date -u)"

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup muse || true
muse setup SimJob "$MUSING" || true
setup ifdhc 2>/dev/null || true
which mu2e ifdh || { echo "ERROR: mu2e/ifdh missing"; exit 2; }

if ifdh ls "$OUTDIR/$GEN" >/dev/null 2>&1; then echo "SKIP: $OUTDIR/$GEN exists"; exit 0; fi

mkdir -p fcl && ifdh cp "$FCL_PNFS/gen_only_extracted.fcl" fcl/gen_only_extracted.fcl || { echo "ERROR: fcl"; exit 2; }
cat > fcl/shard.fcl <<FCL
#include "gen_only_extracted.fcl"
source.firstSubRun : $SUBRUN
services.SeedService.baseSeed : $SEED
outputs.PrimaryOutput.fileName : "$GEN"
FCL
export FHICL_FILE_PATH="$PWD/fcl:$FHICL_FILE_PATH"

/usr/bin/time -v mu2e -c fcl/shard.fcl -n "$NEVTS" > mu2e.log 2>&1
rc=$?
grep -E "TrigReport Events total|VmHWM|Maximum resident|Elapsed|User time|TimeReport CPU" mu2e.log
[ $rc -eq 0 ] || { tail -40 mu2e.log; echo "MU2E FAILED rc=$rc"; exit 3; }

ifdh mkdir_p "$OUTDIR/logs"
ifdh cp "$GEN" "$OUTDIR/$GEN" || { echo "COPY gen FAILED"; exit 5; }
gzip -c mu2e.log > "mu2e.${SEQ}.log.gz" && ifdh cp "mu2e.${SEQ}.log.gz" "$OUTDIR/logs/mu2e.${SEQ}.log.gz"
echo "=== grid_gen_only: done shard $K"
