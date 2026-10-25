#!/bin/bash
# Run the full-schedule capture control and candidate sequentially in the verified p109 area.
set -e -o pipefail
AREA=${1:?usage: run_rejection_pair.sh AREA INPUT_LIST OUTDIR NEVENTS}
LIST=${2:?input list}
OUT=${3:?new output directory}
N=${4:?event limit}
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ "$MUSE_WORK_DIR" = "$AREA" ] && [ "$MUSE_ENVSET" = p109 ] || { echo "Wrong muse area/envset"; exit 2; }
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "Need a positive bounded event count"; exit 2; }
case "$OUT" in /exp/mu2e/data/users/ram2aq/trk_fastreco/*) ;; *) echo "Output must be under trk_fastreco data"; exit 2;; esac
INPUT=$(sed -n '1p' "$LIST")
[ -n "$INPUT" ] || { echo "Empty input list"; exit 2; }
mkdir "$OUT"
export FHICL_FILE_PATH="$HERE/art:${FHICL_FILE_PATH}"
PLUGIN="$AREA/build/$MUSE_STUB/TrkFastSim/lib/libtrkfastsim_KinematicLineFitTrace_module.so"
ldd "$PLUGIN" > "$OUT/linkage.txt"
if grep -q 'not found' "$OUT/linkage.txt"; then echo "Unresolved plugin dependencies"; exit 2; fi
while read -r lib arrow resolved rest; do
  case "$lib" in libKinKal_*)
    case "$(readlink -f "$resolved")" in "$AREA/build/$MUSE_STUB/Offline/lib/"*) ;;
      *) echo "Nonlocal KinKal dependency: $lib $resolved"; exit 2;;
    esac;;
  esac
done < "$OUT/linkage.txt"
grep -q 'libKinKal_Fit.so' "$OUT/linkage.txt"
{
  date -Is
  hostname
  printf 'area=%s\nenvset=%s\nstub=%s\ninput=%s\nevents=%s\n' "$AREA" "$MUSE_ENVSET" "$MUSE_STUB" "$INPUT" "$N"
  for repo in Offline KinKal Production mu2e-trig-config EventNtuple; do
    echo "$repo $(git -C "$AREA/$repo" rev-parse HEAD)"
    git -C "$AREA/$repo" status --short
  done
  echo "tracker $(git -C "$HERE" rev-parse HEAD)"
  git -C "$HERE" status --short
  sha256sum "$PLUGIN" "$AREA/KinKal/Fit/Track.hh" "$AREA/TrkFastSim/src/KinematicLineFitTrace_module.cc"
  sha256sum "$HERE/art/TrkFastSim/fcl/trace_rejection_control.fcl" "$HERE/art/TrkFastSim/fcl/trace_rejection_capture.fcl"
  cat "$AREA/build/$MUSE_STUB/.musebuild"
} > "$OUT/manifest.txt"
for arm in control capture; do
  mu2e --debug-config "$OUT/$arm.expanded.fcl" -c "TrkFastSim/fcl/trace_rejection_$arm.fcl" > "$OUT/$arm.config.log" 2>&1
  rc=0
  /usr/bin/time -v mu2e -c "TrkFastSim/fcl/trace_rejection_$arm.fcl" -s "$INPUT" -n "$N" -T "$OUT/$arm.root" > "$OUT/$arm.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$OUT/$arm.exit"
  [ "$rc" -eq 0 ] || { tail -60 "$OUT/$arm.log"; exit "$rc"; }
done
printf 'Both art arms exited 0; finished tree entries and capture equivalence still require checking.\n'
