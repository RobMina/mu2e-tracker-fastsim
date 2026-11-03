#!/bin/bash
# EventNtuple writer validation on one StrawHitTableDumper (DumpTracks) file, serially on one GPVM:
#   stock  writer closure: the dump's own KKLine fits through reco-ntuple
#   f2     the reco-vr3 chain from digis (fast hits, clusters, LineFinder, F2 own states) through reco-ntuple
#   replay rerun stock and f2 and compare the content hashes
#   cmp    reco-ntcompare of each fast file against the stock EventNtuple of the same digis, plus the readers
# usage: ntuple_validate.sh <dump.root> <stock nts.root> <conditions.json> <outdir> [steps ...]   (default: all)
# env: PY_JAX (python with JAX; required), TRKFS_COMMIT (recorded when git is unavailable)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
DUMP=${1:?dump}; NTS=${2:?stock ntuple}; COND=${3:?conditions}; O=${4:?outdir}; shift 4
STEPS=${*:-stock f2 replay cmp}
HERE=$(cd "$(dirname "$0")/.." && pwd)
APP=$(cd "$HERE/../.." && pwd)
CORE=${M2FS_CORE:-${M2FS_CODE:?set M2FS_CORE, or M2FS_CODE to the mu2e-fastsim checkout}/core}
export PYTHONPATH=$CORE:$HERE:${CRVEXTRAP:?set CRVEXTRAP to the mu2e_track_crv_mcresponse checkout}
export JAX_PLATFORMS=cpu
PY_JAX=${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}
mkdir -p "$O"
run() { local tag=$1; shift; echo "== $tag $(date -Is)"; /usr/bin/time -v "$@" > "$O/$tag.log" 2>&1; local rc=$?
        echo "   rc=$rc $(grep -E 'Elapsed|Maximum resident' "$O/$tag.log" | tr -s ' ' | tr '\n' ' ')"; return $rc; }
for s in $STEPS; do
  case $s in
    stock) run ntuple_stock $PY_JAX -m trkfastsim reco-ntuple "$DUMP" --fit stock --conditions "$COND" --out "$O/fast_stock.root" ;;
    f2)    run ntuple_f2 $PY_JAX -m trkfastsim reco-ntuple "$DUMP" --fit f2 --conditions "$COND" --out "$O/fast_f2.root" \
             --table-out "$O/table_f2.npz" ;;
    replay) for f in stock f2; do                    # deterministic replay: same input, same content hash
             run replay_$f $PY_JAX -m trkfastsim reco-ntuple "$DUMP" --fit $f --conditions "$COND" --out "$O/replay_$f.root"
             echo "   $f: $(grep -h content_sha256 "$O/ntuple_$f.log" "$O/replay_$f.log" | tr -d ' ",' | sort | uniq -c | tr '\n' ' ')"
           done ;;
    cmp)   for f in stock f2; do
             [ -s "$O/fast_$f.root" ] && run cmp_$f python3 -m trkfastsim reco-ntcompare "$O/fast_$f.root" --compare "$NTS" \
               --outdir "$O/cmp_$f"
           done ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
echo "done $(date -Is)"
