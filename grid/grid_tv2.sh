#!/usr/bin/env bash
# Grid worker, T-V2/T-V3: one shard -> digi -> stock extracted reco -> EventNtuple, the same code for both arms.
#   ARM=fast: gen-stream art + fast tracker table -> StrawGasStepsFromTable path A (digi_from_fasttable_alltrig.fcl)
#   ARM=g4:   G4 dts (MDC2025as) -> stock extracted digitisation with the same output selection (digi_g4_alltrig.fcl)
# Shard k = PROCESS: line k+1 of LIST_PNFS ("<input art>[,<input art>...] [<fast table>]" per line: several comma-separated
# inputs run as one art job, with one fast table holding all their events). Code from CODE_TARBALL (muse tarball of
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
SEQ=$(basename "${INPUT%%,*}" .art | awk -F. '{print $NF}')
NTS="nts.ram2aq.tv2${ARM}.v0.${SEQ}.root"
HIT="strawhits.ram2aq.tv2${ARM}.v0.${SEQ}.root"
HAVE_NTS=0; ifdh ls "$OUTDIR/$NTS" >/dev/null 2>&1 && HAVE_NTS=1
HAVE_HIT=0; ifdh ls "$OUTDIR/hits/$HIT" >/dev/null 2>&1 && HAVE_HIT=1
# DUMP=1: also write the StrawHitTableDumper table (digis + stock hits) from the reco job; a shard whose ntuple exists reruns
# digi + reco for the table only (the digitisation is deterministic per input and seed, so the ntuple is not remade)
if [ "$HAVE_NTS" = 1 ] && { [ "${DUMP:-0}" != 1 ] || [ "$HAVE_HIT" = 1 ]; }; then echo "SKIP: $OUTDIR/$NTS exists"; exit 0; fi

ifdh cp "$CODE_TARBALL" ./Code.tar.bz2 && tar xjf ./Code.tar.bz2 && rm -f ./Code.tar.bz2 || { echo "ERROR: code"; exit 2; }
source ./Code/setup.sh || { echo "ERROR: Code/setup.sh"; exit 2; }
which mu2e >/dev/null || { echo "ERROR: mu2e missing"; exit 2; }
# the tarball's setup omits the backing Musing's build dir, which holds the generated mu2e-trig-config menus
BACKBUILD="$(readlink -f ./Code/backing)/build/$MUSE_STUB"
[ -f "$BACKBUILD/mu2e-trig-config/gen/trig_physMenuPSConfig.fcl" ] || { echo "ERROR: no trig menus in $BACKBUILD"; exit 2; }
export FHICL_FILE_PATH="$BACKBUILD:$FHICL_FILE_PATH"
mkdir -p fcl
for f in reco_kkts_extrap_nofilter_v04.fcl reco_smoke_kl_ts.fcl reco_evd_v03.fcl ntuple_kkts_extrap_v04.fcl digi_g4_alltrig.fcl; do
  ifdh cp "$FCL_PNFS/$f" "fcl/$f" || { echo "ERROR: fcl $f"; exit 2; }
done
export FHICL_FILE_PATH="$PWD/fcl:$FHICL_FILE_PATH"

: > inputs.txt
i=0
for IN in ${INPUT//,/ }; do
  ifdh cp "$IN" "./input_$i.art" || { echo "ERROR: stage input $IN"; exit 2; }
  echo "$PWD/input_$i.art" >> inputs.txt; i=$((i + 1))
done
DIG=dig.art; MCS=mcs.art
T0=$(date +%s)
if [ "$ARM" = fast ]; then
  ifdh cp "$TABLE" ./fasttrk.root || { echo "ERROR: stage table"; exit 2; }
  printf '#include "TrkFastSim/fcl/digi_from_fasttable_alltrig.fcl"\nphysics.producers.compressDetStepMCs.trkTableFile : "%s"\n' \
    "$PWD/fasttrk.root" > fcl/digi.fcl
else
  printf '#include "digi_g4_alltrig.fcl"\n' > fcl/digi.fcl
fi
/usr/bin/time -v mu2e -c fcl/digi.fcl -S inputs.txt -o "$DIG" -n "$NEVTS" > digi.log 2>&1 || { tail -40 digi.log; echo "DIGI FAILED"; exit 3; }
T1=$(date +%s)
rm -f input_*.art
RECO=fcl/reco_kkts_extrap_nofilter_v04.fcl
if [ "${DUMP:-0}" = 1 ]; then
  printf '#include "reco_kkts_extrap_nofilter_v04.fcl"\nphysics.analyzers.trkdump : { module_type : StrawHitTableDumper  ProtonBunchTimeTag : @local::physics.producers.makeSH.ProtonBunchTimeTag }\nphysics.EndPath : [ @sequence::physics.EndPath, trkdump ]\nservices.TFileService.fileName : "%s"\n' "$HIT" > fcl/reco_dump.fcl
  RECO=fcl/reco_dump.fcl
fi
mu2e -c $RECO -s "$DIG" --output "$MCS" > reco.log 2>&1 || { tail -40 reco.log; echo "RECO FAILED"; exit 3; }
T2=$(date +%s)
if [ "$HAVE_NTS" = 0 ]; then
  mu2e -c fcl/ntuple_kkts_extrap_v04.fcl -s "$MCS" -T "$NTS" > ntuple.log 2>&1 || { tail -40 ntuple.log; echo "NTUPLE FAILED"; exit 4; }
else
  echo "ntuple exists, not remade" > ntuple.log
fi
T3=$(date +%s)
grep -h "TrigReport Events total" digi.log reco.log ntuple.log
grep -E "Maximum resident" digi.log
echo "times digi $((T1 - T0)) s, reco $((T2 - T1)) s, ntuple $((T3 - T2)) s"

ifdh mkdir_p "$OUTDIR/logs"
[ "$HAVE_NTS" = 0 ] && { ifdh cp "$NTS" "$OUTDIR/$NTS" || { echo "COPY FAILED"; exit 5; }; }
if [ "${DUMP:-0}" = 1 ]; then
  ifdh mkdir_p "$OUTDIR/hits"; ifdh cp "$HIT" "$OUTDIR/hits/$HIT" || { echo "COPY hits FAILED"; exit 5; }
fi
tar czf "logs.${SEQ}.tgz" digi.log reco.log ntuple.log && ifdh cp "logs.${SEQ}.tgz" "$OUTDIR/logs/logs.${SEQ}.tgz"
echo "=== grid_tv2: done -> $OUTDIR/$NTS ${DUMP:+(hits $OUTDIR/hits/$HIT)}"
