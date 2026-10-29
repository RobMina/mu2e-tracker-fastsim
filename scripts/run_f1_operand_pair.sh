#!/bin/bash
# Run only after user build confirmation and local-library verification, with an owned idle host.
set -e -o pipefail
AREA=${1:?usage: run_f1_operand_pair.sh AREA INPUT_LIST CONFIG_DIR NEW_OUT NEVENTS}
LIST=${2:?input list}; CONFIG=${3:?target configs}; OUT=${4:?new output}; N=${5:?event count}
EXPECTED=/exp/mu2e/app/users/ram2aq/fastreco_operands_p109_20261028
DATA=/exp/mu2e/data/users/ram2aq
[ "$AREA" = "$EXPECTED" ] && [ "${MUSE_WORK_DIR:-}" = "$AREA" ] && [ "${MUSE_ENVSET:-}" = p109 ] || {
  echo 'Wrong isolated area or environment'; exit 2;
}
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo 'Positive bounded event count required'; exit 2; }
case "$OUT" in "$DATA/trk_f1native_r2_20261027/"*) ;; *) echo 'Output outside the F1-native data area'; exit 2;; esac
[ ! -e "$OUT" ] || { echo 'Existing output refused'; exit 2; }
for f in control.fcl capture.fcl targets.json; do test -f "$CONFIG/$f" || exit 2; done
INPUT=$(sed -n '1p' "$LIST")
[ -n "$INPUT" ] || { echo 'Empty input list'; exit 2; }
PLUGIN=$AREA/build/$MUSE_STUB/TrkFastSim/lib/libtrkfastsim_KinematicLineFitTrace_module.so
KLIB=$AREA/build/$MUSE_STUB/Offline/lib/libKinKal_Fit.so.0.0.0
for source in KinKal/Fit/Track.hh KinKal/Fit/IterationTrace.hh TrkFastSim/src/KinematicLineFitTrace_module.cc \
              TrkFastSim/inc/IterationCaptureWriter.hh TrkFastSim/inc/HitOperandCapture.hh; do
  [ -f "$AREA/$source" ] && [ "$PLUGIN" -nt "$AREA/$source" ] && [ "$KLIB" -nt "$AREA/$source" ] || {
    echo "Build not verified or stale relative to $source"; exit 2;
  }
done
(cd "$AREA" && sha256sum -c capture-source-manifest.sha256)
limit=$(getfattr --only-values -n ceph.quota.max_bytes "$DATA")
used=$(getfattr --only-values -n ceph.dir.rbytes "$DATA")
[ $((limit-used)) -ge $((4*1024*1024*1024)) ] || { echo 'Need 4 GiB data reserve; no automatic deletion'; exit 2; }
mkdir "$OUT"
ldd "$PLUGIN" > "$OUT/linkage.txt"
! grep -q 'not found' "$OUT/linkage.txt" || { echo 'Unresolved library'; exit 2; }
while read -r lib arrow resolved rest; do
  case "$lib" in libKinKal_*|libmu2e_Mu2eKinKal*)
    case "$(readlink -f "$resolved")" in "$AREA/build/$MUSE_STUB/Offline/lib/"*) ;;
      *) echo "Nonlocal dependency $lib $resolved"; exit 2;;
    esac;;
  esac
done < "$OUT/linkage.txt"
grep -q libKinKal_Fit "$OUT/linkage.txt"
PUBLISHER=/exp/mu2e/app/users/ram2aq/analysis/thread-f-publish-20261025/art
for config in trace_f1_candidates trace_rejection_control trace_kkline_mc dump_tracks_mc; do
  test -f "$PUBLISHER/TrkFastSim/fcl/$config.fcl" || { echo "Missing frozen publisher config $config"; exit 2; }
done
export FHICL_FILE_PATH="$CONFIG:$PUBLISHER:$AREA:${FHICL_FILE_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CPU=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
{
  date -Is; hostname
  printf 'area=%s\ninput=%s\nevents=%s\nconfig=%s\n' "$AREA" "$INPUT" "$N" "$CONFIG"
  cat "$AREA/source-pins.txt" "$AREA/capture-source-manifest.sha256" "$AREA/operand-preparation.json"
  sha256sum "$PLUGIN" "$KLIB" "$CONFIG/control.fcl" "$CONFIG/capture.fcl" "$CONFIG/targets.json" "$0"
  for config in trace_f1_candidates trace_rejection_control trace_kkline_mc dump_tracks_mc; do
    sha256sum "$PUBLISHER/TrkFastSim/fcl/$config.fcl"
  done
} > "$OUT/manifest.txt"
for arm in control capture; do
  taskset -c "$CPU" mu2e --debug-config "$OUT/$arm.expanded.fcl" -c "$CONFIG/$arm.fcl" > "$OUT/$arm.config.txt" 2>&1
  rc=0
  /usr/bin/time -v taskset -c "$CPU" mu2e -c "$CONFIG/$arm.fcl" -s "$INPUT" -n "$N" -T "$OUT/$arm.root" > "$OUT/$arm.txt" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$OUT/$arm.exit"
  [ "$rc" -eq 0 ] || { tail -60 "$OUT/$arm.txt"; exit "$rc"; }
done
echo 'OPERAND_PAIR_DONE: standard capture gate and old-tree/sidecar checks required; not numerical closure.'
