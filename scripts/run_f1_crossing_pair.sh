#!/bin/bash
# Run a bounded crossing-capture pair after the user's build and library verification.
set -e -o pipefail
AREA=/exp/mu2e/app/users/ram2aq/fastreco_crossings_p109_20261028
LIST=${1:?input list}; CONFIG=${2:?target configs}; OUT=${3:?new output}; N=${4:?event count}
DATA=/exp/mu2e/data/users/ram2aq
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ "${MUSE_WORK_DIR:-}" = "$AREA" ] && [ "${MUSE_ENVSET:-}" = p109 ] || exit 2
[ "$N" = 100 ] || { echo 'Only the authorized 100-event smoke is supported'; exit 2; }
[ "$(cat "$DATA/agent_locks/$(hostname -s)/owner")" = f1native-crossing-postbuild-20261028 ] || exit 2
OUT=$(realpath -m "$OUT")
case "$OUT" in "$(realpath -e "$DATA/trk_f1native_r2_20261027")/"*) ;; *) exit 2;; esac
[ ! -e "$OUT" ] || exit 2
[ "$(realpath -e "$CONFIG")" = "$DATA/trk_fastreco/p109_iteration_targets_v2_20261026" ] || exit 2
[ "$(realpath -e "$LIST")" = "$DATA/trk_fastreco/au_dig_3f.txt" ] || exit 2
sha256sum -c <<EOF
91c49c9a256a0fd6f7af34620db96ad40cab4336f616316ef31855e130090802  $LIST
b228c36d268e8d49fc48e879cd7cbbc76285ba857ec3325f4992f73f81c77335  $CONFIG/control.fcl
4154132908087c1379314309fb72996bdabc5214b27c734e079c27f77dd4808a  $CONFIG/capture.fcl
f32c3e49bec3746fc15a57bc588a3858a15195820fd1705a4d756cb7504fce90  $CONFIG/targets.json
EOF
for f in control.fcl capture.fcl targets.json; do test -f "$CONFIG/$f" || exit 2; done
for f in f1_crossing_check.py f1_crossing_geometry_check.py f1_crossing_record_check.py; do test -f "$HERE/scripts/$f" || exit 2; done
INPUT=$(sed -n '1p' "$LIST")
[ -n "$INPUT" ] || exit 2
PLUGIN=$AREA/build/$MUSE_STUB/TrkFastSim/lib/libtrkfastsim_KinematicLineFitTrace_module.so
KLIB=$AREA/build/$MUSE_STUB/Offline/lib/libKinKal_Fit.so.0.0.0
while read -r hash source; do
  [ "$PLUGIN" -nt "$AREA/$source" ] && [ "$KLIB" -nt "$AREA/$source" ] || exit 2
done < "$AREA/capture-source-manifest.sha256"
(cd "$AREA" && sha256sum -c capture-source-manifest.sha256)
limit=$(getfattr --only-values -n ceph.quota.max_bytes "$DATA")
used=$(getfattr --only-values -n ceph.dir.rbytes "$DATA")
[ $((limit-used)) -ge $((4*1024*1024*1024)) ] || { echo 'Need 4 GiB reserve; no automatic deletion'; exit 2; }
mkdir "$OUT"
ldd "$PLUGIN" > "$OUT/linkage.txt"
! grep -q 'not found' "$OUT/linkage.txt" || exit 2
while read -r lib arrow resolved rest; do
  case "$lib" in libKinKal_*|libmu2e_Mu2eKinKal*)
    case "$(readlink -f "$resolved")" in "$AREA/build/$MUSE_STUB/Offline/lib/"*) ;; *) exit 2;; esac;;
  esac
done < "$OUT/linkage.txt"
grep -q libKinKal_Fit "$OUT/linkage.txt"
PUBLISHER=/exp/mu2e/app/users/ram2aq/analysis/thread-f-publish-20261025/art
for config in trace_f1_candidates trace_rejection_control trace_kkline_mc dump_tracks_mc; do
  test -f "$PUBLISHER/TrkFastSim/fcl/$config.fcl" || exit 2
done
export FHICL_FILE_PATH="$CONFIG:$PUBLISHER:$AREA:${FHICL_FILE_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CPU=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
{
  date -Is; hostname
  printf 'area=%s\ninput=%s\nevents=%s\nconfig=%s\nquota_used=%s\nquota_limit=%s\n' "$AREA" "$INPUT" "$N" "$CONFIG" "$used" "$limit"
  cat "$AREA/source-pins.txt" "$AREA/capture-source-manifest.sha256"
  sha256sum "$PLUGIN" "$KLIB" "$CONFIG/control.fcl" "$CONFIG/capture.fcl" "$CONFIG/targets.json" "$LIST" "$0"
  sha256sum "$DATA/trk_f1native_r2_20261027/crossing_preparation_v1.json"
  for config in trace_f1_candidates trace_rejection_control trace_kkline_mc dump_tracks_mc; do
    sha256sum "$PUBLISHER/TrkFastSim/fcl/$config.fcl"
  done
} > "$OUT/manifest.txt"
for arm in control capture; do
  taskset -c "$CPU" mu2e --debug-config "$OUT/$arm.expanded.fcl" -c "$CONFIG/$arm.fcl" > "$OUT/$arm.config.txt" 2>&1
  REFERENCE=$DATA/trk_f1native_r2_20261027/operand_smoke_20261028_v2/$arm.expanded.fcl
  if [ "$arm" = control ]; then EXPECTED_HASH=0ebf6d0c47b5a625439544349fd3fa27b52e6a1e95d96c89ec0a61018410a28a;
  else EXPECTED_HASH=68123c10e62928395122f2f7d8e081d4def024f0d867a31d4e9eeb93c070eecc; fi
  printf '%s  %s\n' "$EXPECTED_HASH" "$REFERENCE" | sha256sum -c -
  [ "$(grep -Fc "fileName: \"$OUT/$arm.expanded.fcl\"" "$OUT/$arm.expanded.fcl")" = 1 ] || exit 2
  [ "$(grep -Fc "fileName: \"$REFERENCE\"" "$REFERENCE")" = 1 ] || exit 2
  sed "s|fileName: \"$OUT/$arm.expanded.fcl\"|fileName: \"__DEBUG_OUTPUT__\"|" "$OUT/$arm.expanded.fcl" > "$OUT/$arm.normalized.fcl"
  sed "s|fileName: \"$REFERENCE\"|fileName: \"__DEBUG_OUTPUT__\"|" "$REFERENCE" > "$OUT/$arm.reference-normalized.fcl"
  cmp "$OUT/$arm.normalized.fcl" "$OUT/$arm.reference-normalized.fcl"
  rc=0
  /usr/bin/time -v taskset -c "$CPU" mu2e -c "$CONFIG/$arm.fcl" -s "$INPUT" -n "$N" -T "$OUT/$arm.root" > "$OUT/$arm.txt" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$OUT/$arm.exit"
  [ "$rc" -eq 0 ] || { tail -60 "$OUT/$arm.txt"; exit "$rc"; }
done
(cd "$AREA" && sha256sum -c capture-source-manifest.sha256) > "$OUT/source_after.txt"
stat -c '%s %n' "$OUT/control.root" "$OUT/capture.root" > "$OUT/sizes.txt"
echo 'CROSSING_PAIR_DONE: standard gate and independent new-sidecar checker still required.'
