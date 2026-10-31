#!/bin/bash
# EventNtuple writer validation on one StrawHitTableDumper (DumpTracks) file, serially on one GPVM:
#   stock  writer closure: the dump's own KKLine fits through reco-ntuple
#   f2     the reco-vr3 chain from digis (fast hits, clusters, LineFinder, F2 own states) through reco-ntuple
#   f1     reco-vr2f1 --own-states --save-tracks (stock-conditioned F1), then reco-ntuple --fit f1
#   replay rerun stock and f2 and compare the content hashes
#   cmp    reco-ntcompare of each fast file against the stock EventNtuple of the same digis, plus the readers
# usage: ntuple_validate.sh <dump.root> <stock nts.root> <conditions.json> <outdir> [steps ...]   (default: all)
# env: PY_JAX (python with JAX; required), TRKFS_COMMIT (recorded when git is unavailable),
#      F1_MAX_TRACKS (first N F1 candidates; the full 29k-event file needs > 9 GB in reco-vr2f1),
#      F1_TRACKS (an existing reco-vr2f1 --own-states --save-tracks f1_tracks.npz of the same dump: skips reco-vr2f1)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
DUMP=${1:?dump}; NTS=${2:?stock ntuple}; COND=${3:?conditions}; O=${4:?outdir}; shift 4
STEPS=${*:-stock f2 f1 replay cmp}
HERE=$(cd "$(dirname "$0")/.." && pwd)
APP=$(cd "$HERE/../.." && pwd)
CORE=${M2FS_CORE:-${M2FS_CODE:?set M2FS_CORE, or M2FS_CODE to the mu2e-fastsim checkout}/core}
export PYTHONPATH=$CORE:$HERE:${CRVEXTRAP:?set CRVEXTRAP to the mu2e_track_crv_mcresponse checkout}
export JAX_PLATFORMS=cpu F1_JAX_CACHE=$O/jax_cache
PY_JAX=${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}
mkdir -p "$O"
run() { local tag=$1; shift; echo "== $tag $(date -Is)"; /usr/bin/time -v "$@" > "$O/$tag.log" 2>&1; local rc=$?
        echo "   rc=$rc $(grep -E 'Elapsed|Maximum resident' "$O/$tag.log" | tr -s ' ' | tr '\n' ' ')"; return $rc; }
for s in $STEPS; do
  case $s in
    stock) run ntuple_stock $PY_JAX -m trkfastsim reco-ntuple "$DUMP" --fit stock --conditions "$COND" --out "$O/fast_stock.root" ;;
    f2)    run ntuple_f2 $PY_JAX -m trkfastsim reco-ntuple "$DUMP" --fit f2 --conditions "$COND" --out "$O/fast_f2.root" \
             --table-out "$O/table_f2.npz" ;;
    f1)    F1T=${F1_TRACKS:-$O/vr2f1/f1_tracks.npz}
           if [ -z "${F1_TRACKS:-}" ]; then
             run vr2f1 $PY_JAX -m trkfastsim reco-vr2f1 "$DUMP" --conditions "$COND" --own-states --save-tracks \
               --outdir "$O/vr2f1" ${F1_MAX_TRACKS:+--max-tracks $F1_MAX_TRACKS} || continue
           fi
           run ntuple_f1 $PY_JAX -m trkfastsim reco-ntuple "$DUMP" --fit f1 --f1-tracks "$F1T" \
             --conditions "$COND" --out "$O/fast_f1.root" ;;
    replay) for f in stock f2 f1; do                 # deterministic replay: same input, same content hash
             [ $f = f1 ] && [ ! -s "$O/fast_f1.root" ] && continue
             run replay_$f $PY_JAX -m trkfastsim reco-ntuple "$DUMP" --fit $f --conditions "$COND" --out "$O/replay_$f.root" \
               $([ $f = f1 ] && echo --f1-tracks "${F1_TRACKS:-$O/vr2f1/f1_tracks.npz}")
             echo "   $f: $(grep -h content_sha256 "$O/ntuple_$f.log" "$O/replay_$f.log" | tr -d ' ",' | sort | uniq -c | tr '\n' ' ')"
           done ;;
    cmp)   for f in stock f2 f1; do
             [ -s "$O/fast_$f.root" ] && run cmp_$f python3 -m trkfastsim reco-ntcompare "$O/fast_$f.root" --compare "$NTS" \
               --outdir "$O/cmp_$f"
           done ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
echo "done $(date -Is)"
