#!/bin/bash
# Run only the fixed 3000-event crossing pair, preserving both prior captures.
set -e -o pipefail
AREA=/exp/mu2e/app/users/ram2aq/fastreco_crossings_p109_20261028
DATA=/exp/mu2e/data/users/ram2aq
F1N=$DATA/trk_f1native_r2_20261027
CONFIG=$DATA/trk_fastreco/p109_iteration_targets_v2_20261026
LIST=$DATA/trk_fastreco/au_dig_3f.txt
SMOKE=$F1N/crossing_smoke_20261028_v1
REFERENCE=$F1N/operand_3k_20261028_v1
OUT=${1:?new output directory required}
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ "${MUSE_WORK_DIR:-}" = "$AREA" ] && [ "${MUSE_ENVSET:-}" = p109 ] || exit 2
[ "$(cat "$DATA/agent_locks/$(hostname -s)/owner")" = f1native-crossing-full-20261028 ] || exit 2
OUT=$(realpath -m "$OUT")
case "$OUT" in "$(realpath -e "$F1N")/"*) ;; *) exit 2;; esac
[ ! -e "$OUT" ] || { echo 'Existing output refused'; exit 2; }
for f in f1_crossing_full_check.py f1_crossing_geometry_check.py f1_crossing_record_check.py; do test -f "$HERE/scripts/$f" || exit 2; done
sha256sum -c <<EOF
91c49c9a256a0fd6f7af34620db96ad40cab4336f616316ef31855e130090802  $LIST
b228c36d268e8d49fc48e879cd7cbbc76285ba857ec3325f4992f73f81c77335  $CONFIG/control.fcl
4154132908087c1379314309fb72996bdabc5214b27c734e079c27f77dd4808a  $CONFIG/capture.fcl
f32c3e49bec3746fc15a57bc588a3858a15195820fd1705a4d756cb7504fce90  $CONFIG/targets.json
ce3762ab9a5610e5838a5939ccff47c67d362812a41ddb1428eca13cc269cd5d  $AREA/capture-source-manifest.sha256
9b25e7f783b0949f8a7e1c5c27548a0928a59eb45f1d1090d63a6fe082967c84  $F1N/crossing_preparation_v1.json
9b9983124f8df4f48b253158f414f7aa9712d03d1eb0b22e250ea47810f12eb1  $SMOKE/crossing_check_v1.json
cc3441c37613151767bdb9bcbb24a8e7d94813d2859c095a87a15ee4afe85439  $SMOKE/manifest.txt
81a461613d6955e2f948f3323e7db20e6618e078ff731cce464f9bab31a3640a  $SMOKE/capture.root
101623117e0f245ad02982cd01ff88a15b7411469319199fee906ae22232c1da  $REFERENCE/check.json
b97804739a88d739d6fcda45954edd4a88b322e67d2a58929b333971f4c7beb6  $REFERENCE/hit_operands_check_v2.json
413b06ff800be415a207a426d6b48bf068d69d09608aefec79cf7c081aee7fca  $REFERENCE/manifest.txt
c79577c3aeea7e18216091e431f60bd4f19ac11522ec3a06e3f8b871cecc6da0  $F1N/crossing_build_preflight_20261028_v1/verification.txt
EOF
INPUT=$(sed -n '1p' "$LIST")
[ -n "$INPUT" ] && grep -Fxq "input=$INPUT" "$REFERENCE/manifest.txt" || exit 2
PLUGIN=$AREA/build/$MUSE_STUB/TrkFastSim/lib/libtrkfastsim_KinematicLineFitTrace_module.so
KLIB=$AREA/build/$MUSE_STUB/Offline/lib/libKinKal_Fit.so.0.0.0
for artifact in "$PLUGIN" "$KLIB"; do
  saved=$(grep -F "  $artifact" "$SMOKE/manifest.txt")
  [ "$(printf '%s\n' "$saved" | wc -l)" -eq 1 ] && [ -n "$saved" ] || exit 2
  printf '%s\n' "$saved" | sha256sum -c -
done
PREFLIGHT=$F1N/crossing_build_preflight_20261028_v1/verification.txt
for component in Detector Examples Fit General Geometry MatEnv Trajectory; do
  artifact=$AREA/build/$MUSE_STUB/Offline/lib/libKinKal_$component.so.0.0.0
  saved=$(grep -E "^[0-9a-f]{64}  $artifact$" "$PREFLIGHT")
  [ "$(printf '%s\n' "$saved" | wc -l)" -eq 1 ] && [ -n "$saved" ] || exit 2
  printf '%s\n' "$saved" | sha256sum -c -
done
while read -r repo pin; do
  [ "$(git -C "$AREA/$repo" rev-parse HEAD)" = "$pin" ] || exit 2
done < "$AREA/source-pins.txt"
while read -r hash source; do
  [ "$PLUGIN" -nt "$AREA/$source" ] && [ "$KLIB" -nt "$AREA/$source" ] || exit 2
done < "$AREA/capture-source-manifest.sha256"
(cd "$AREA" && sha256sum -c capture-source-manifest.sha256)
SMOKE_BYTES=$(stat -c %s "$SMOKE/capture.root")
[ "$SMOKE_BYTES" -eq 15152564 ] || exit 2
CONTROL_BYTES=$(stat -c %s "$REFERENCE/control.root")
CAPTURE_BYTES=$(stat -c %s "$REFERENCE/capture.root")
[[ "$CONTROL_BYTES" =~ ^[0-9]+$ && "$CAPTURE_BYTES" =~ ^[0-9]+$ ]] || exit 2
BASE_BYTES=$((CONTROL_BYTES+CAPTURE_BYTES))
ALLOWANCE=$((4*1024*1024*1024))
RESERVE=$((4*1024*1024*1024))
SCALED_PROXY=$(((SMOKE_BYTES*147+1)/2 + BASE_BYTES))
[ "$ALLOWANCE" -ge $((2*SCALED_PROXY)) ] || { echo 'Space allowance no longer covers conservative proxy'; exit 2; }
quota_gate() {
  LIMIT=$(getfattr --only-values -n ceph.quota.max_bytes "$DATA")
  USED=$(getfattr --only-values -n ceph.dir.rbytes "$DATA")
  [[ "$LIMIT" =~ ^[0-9]+$ && "$USED" =~ ^[0-9]+$ ]] || exit 2
  [ $((LIMIT-USED)) -ge $((ALLOWANCE+RESERVE)) ] || { echo 'Need 4 GiB output allowance plus 4 GiB reserve; no deletion'; exit 2; }
}
quota_gate
mkdir "$OUT"
{
  printf 'smoke_capture_bytes=%s\nsmoke_targets=2\nfull_targets=147\npredecessor_pair_bytes=%s\n' "$SMOKE_BYTES" "$BASE_BYTES"
  printf 'size_proxy_bytes=%s\noutput_allowance_bytes=%s\nreserve_bytes=%s\nquota_used=%s\nquota_limit=%s\n' "$SCALED_PROXY" "$ALLOWANCE" "$RESERVE" "$USED" "$LIMIT"
  echo 'Size proxy only; no runtime extrapolation or guaranteed output upper bound. Recheck before each arm.'
} > "$OUT/space_plan.txt"
ldd "$PLUGIN" > "$OUT/linkage.txt"
! grep -q 'not found' "$OUT/linkage.txt" || exit 2
while read -r lib arrow resolved rest; do
  case "$lib" in libKinKal_*|libmu2e_Mu2eKinKal*)
    case "$(readlink -f "$resolved")" in "$AREA/build/$MUSE_STUB/Offline/lib/"*) ;; *) exit 2;; esac;;
  esac
done < "$OUT/linkage.txt"
grep -q libKinKal_Fit "$OUT/linkage.txt"
PUBLISHER=/exp/mu2e/app/users/ram2aq/analysis/thread-f-publish-20261025/art
export FHICL_FILE_PATH="$CONFIG:$PUBLISHER:$AREA:${FHICL_FILE_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CPU=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
{
  date -Is; hostname
  printf 'area=%s\ninput=%s\nevents=3000\nconfig=%s\nquota_used=%s\nquota_limit=%s\n' "$AREA" "$INPUT" "$CONFIG" "$USED" "$LIMIT"
  cat "$AREA/source-pins.txt" "$AREA/capture-source-manifest.sha256"
  sha256sum "$PLUGIN" "$KLIB" "$CONFIG/control.fcl" "$CONFIG/capture.fcl" "$CONFIG/targets.json" "$LIST" "$0"
  sha256sum "$SMOKE/crossing_check_v1.json" "$REFERENCE/check.json" "$REFERENCE/hit_operands_check_v2.json"
  sha256sum "$F1N/crossing_preparation_v1.json" "$OUT/space_plan.txt" "$PREFLIGHT"
  for component in Detector Examples Fit General Geometry MatEnv Trajectory; do
    sha256sum "$AREA/build/$MUSE_STUB/Offline/lib/libKinKal_$component.so.0.0.0"
  done
  for f in f1_crossing_full_check.py f1_crossing_check.py f1_crossing_geometry_check.py f1_crossing_record_check.py; do
    sha256sum "$HERE/scripts/$f"
  done
  for config in trace_f1_candidates trace_rejection_control trace_kkline_mc dump_tracks_mc; do
    sha256sum "$PUBLISHER/TrkFastSim/fcl/$config.fcl"
  done
} > "$OUT/manifest.txt"
for arm in control capture; do
  quota_gate
  printf '%s %s %s\n' "$arm" "$USED" "$LIMIT" >> "$OUT/quota_before_arm.txt"
  taskset -c "$CPU" mu2e --debug-config "$OUT/$arm.expanded.fcl" -c "$CONFIG/$arm.fcl" > "$OUT/$arm.config.txt" 2>&1
  REF=$REFERENCE/$arm.expanded.fcl
  if [ "$arm" = control ]; then EXPECTED=ae36805ddb56e68d9abe05358a9ea59857d45887e0baf7a5b1c0154e432db153;
  else EXPECTED=88106635d1ecaa6d802ff4d27db1e5740803eb27baaa90d8f936e2373c5e150c; fi
  printf '%s  %s\n' "$EXPECTED" "$REF" | sha256sum -c -
  [ "$(grep -Fc "fileName: \"$OUT/$arm.expanded.fcl\"" "$OUT/$arm.expanded.fcl")" = 1 ] || exit 2
  [ "$(grep -Fc "fileName: \"$REF\"" "$REF")" = 1 ] || exit 2
  sed "s|fileName: \"$OUT/$arm.expanded.fcl\"|fileName: \"__DEBUG_OUTPUT__\"|" "$OUT/$arm.expanded.fcl" > "$OUT/$arm.normalized.fcl"
  sed "s|fileName: \"$REF\"|fileName: \"__DEBUG_OUTPUT__\"|" "$REF" > "$OUT/$arm.reference-normalized.fcl"
  cmp "$OUT/$arm.normalized.fcl" "$OUT/$arm.reference-normalized.fcl"
  rc=0
  /usr/bin/time -v taskset -c "$CPU" mu2e -c "$CONFIG/$arm.fcl" -s "$INPUT" -n 3000 -T "$OUT/$arm.root" > "$OUT/$arm.txt" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$OUT/$arm.exit"
  [ "$rc" -eq 0 ] || { tail -60 "$OUT/$arm.txt"; exit "$rc"; }
done
(cd "$AREA" && sha256sum -c capture-source-manifest.sha256) > "$OUT/source_after.txt"
stat -c '%s %n' "$OUT/control.root" "$OUT/capture.root" > "$OUT/sizes.txt"
echo 'CROSSING_FULL_PAIR_DONE: full standard and new sidecar checks required; not F1 closure.'
