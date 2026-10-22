#!/usr/bin/env bash
# Grid worker, T-V2/T-V3: one shard -> digi -> stock extracted reco -> EventNtuple, the same code for both arms.
#   ARM=fast: gen-stream art + fast tracker table -> StrawGasStepsFromTable path A (digi_from_fasttable_alltrig.fcl)
#   ARM=g4:   G4 dts (MDC2025as) -> stock extracted digitisation with the same output selection (digi_g4_alltrig.fcl)
# Shard k = PROCESS: line k+1 of LIST_PNFS ("<input art> [<fast table>]" per line). Code from CODE_TARBALL (muse tarball of
# tracker/art: TrkFastSim + EventNtuple on MDC2025ay); reco and ntuple fcls from FCL_PNFS. Skips if the ntuple exists.
set -o pipefail
OUTDIR="${1:?need OUTDIR (pnfs scratch)}"
ARM="${ARM:?fast|g4}"; LIST_PNFS="${LIST_PNFS:?}"; FCL_PNFS="${FCL_PNFS:?}"; CODE_TARBALL="${CODE_TARBALL:?}"
NEVTS="${NEVTS:--1}"; K="${PROCESS:?}"
echo "=== grid_tv2: arm $ARM shard $K host $(hostname) $(date -u)"

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup muse || true
setup ifdhc 2>/dev/null || true
which ifdh >/dev/null || { echo "ERROR: no ifdh"; exit 2; }

ifdh cp "$LIST_PNFS" ./list.txt || { echo "ERROR: list"; exit 2; }
read -r INPUT TABLE <<< "$(sed -n "$((K + 1))p" ./list.txt)"
[ -n "$INPUT" ] || { echo "ERROR: no line $((K + 1))"; exit 2; }
SEQ=$(basename "$INPUT" .art | awk -F. '{print $NF}')
NTS="nts.ram2aq.tv2${ARM}.v0.${SEQ}.root"
if ifdh ls "$OUTDIR/$NTS" >/dev/null 2>&1; then echo "SKIP: $OUTDIR/$NTS exists"; exit 0; fi

ifdh cp "$CODE_TARBALL" ./Code.tar.bz2 && tar xjf ./Code.tar.bz2 && rm -f ./Code.tar.bz2 || { echo "ERROR: code"; exit 2; }
source ./Code/setup.sh || { echo "ERROR: Code/setup.sh"; exit 2; }
which mu2e >/dev/null || { echo "ERROR: mu2e missing"; exit 2; }
mkdir -p fcl
for f in reco_kkts_extrap_nofilter_v04.fcl reco_smoke_kl_ts.fcl reco_evd_v03.fcl ntuple_kkts_extrap_v04.fcl digi_g4_alltrig.fcl; do
  ifdh cp "$FCL_PNFS/$f" "fcl/$f" || { echo "ERROR: fcl $f"; exit 2; }
done
export FHICL_FILE_PATH="$PWD/fcl:$FHICL_FILE_PATH"

ifdh cp "$INPUT" ./input.art || { echo "ERROR: stage input"; exit 2; }
DIG=dig.art; MCS=mcs.art
T0=$(date +%s)
if [ "$ARM" = fast ]; then
  ifdh cp "$TABLE" ./fasttrk.root || { echo "ERROR: stage table"; exit 2; }
  printf '#include "TrkFastSim/fcl/digi_from_fasttable_alltrig.fcl"\nphysics.producers.compressDetStepMCs.trkTableFile : "%s"\n' \
    "$PWD/fasttrk.root" > fcl/digi.fcl
else
  printf '#include "digi_g4_alltrig.fcl"\n' > fcl/digi.fcl
fi
/usr/bin/time -v mu2e -c fcl/digi.fcl -s ./input.art -o "$DIG" -n "$NEVTS" > digi.log 2>&1 || { tail -40 digi.log; echo "DIGI FAILED"; exit 3; }
T1=$(date +%s)
mu2e -c fcl/reco_kkts_extrap_nofilter_v04.fcl -s "$DIG" --output "$MCS" > reco.log 2>&1 || { tail -40 reco.log; echo "RECO FAILED"; exit 3; }
T2=$(date +%s)
mu2e -c fcl/ntuple_kkts_extrap_v04.fcl -s "$MCS" -T "$NTS" > ntuple.log 2>&1 || { tail -40 ntuple.log; echo "NTUPLE FAILED"; exit 4; }
T3=$(date +%s)
grep -h "TrigReport Events total" digi.log reco.log ntuple.log
grep -E "Maximum resident" digi.log
echo "times digi $((T1 - T0)) s, reco $((T2 - T1)) s, ntuple $((T3 - T2)) s"

ifdh mkdir_p "$OUTDIR/logs"
ifdh cp "$NTS" "$OUTDIR/$NTS" || { echo "COPY FAILED"; exit 5; }
tar czf "logs.${SEQ}.tgz" digi.log reco.log ntuple.log && ifdh cp "logs.${SEQ}.tgz" "$OUTDIR/logs/logs.${SEQ}.tgz"
echo "=== grid_tv2: done -> $OUTDIR/$NTS"
