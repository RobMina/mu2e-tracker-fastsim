#!/bin/bash
# Read-only source/binary/linkage verification after the user's build confirmation.
set -e -o pipefail
AREA=/exp/mu2e/app/users/ram2aq/fastreco_crossings_p109_20261028
OUT=${1:?new data output directory required}
DATA=/exp/mu2e/data/users/ram2aq/trk_f1native_r2_20261027
[ "${MUSE_WORK_DIR:-}" = "$AREA" ] && [ "${MUSE_ENVSET:-}" = p109 ] || exit 2
case "$OUT" in "$DATA/"*) ;; *) exit 2;; esac
[ ! -e "$OUT" ] || exit 2
mkdir "$OUT"
exec > >(tee "$OUT/verification.txt") 2>&1
printf '%s\n' 'Build and matching install explicitly user-confirmed; build-pane markers unavailable, not required again.'
date -Is
hostname
PLUGIN=$AREA/build/$MUSE_STUB/TrkFastSim/lib/libtrkfastsim_KinematicLineFitTrace_module.so
LIBDIR=$AREA/build/$MUSE_STUB/Offline/lib
[ -f "$PLUGIN" ] && [ -f "$LIBDIR/libKinKal_Fit.so.0.0.0" ]
cd "$AREA"
printf '%s  %s\n' ce3762ab9a5610e5838a5939ccff47c67d362812a41ddb1428eca13cc269cd5d capture-source-manifest.sha256 | sha256sum -c -
printf '%s  %s\n' 9b25e7f783b0949f8a7e1c5c27548a0928a59eb45f1d1090d63a6fe082967c84 "$DATA/crossing_preparation_v1.json" | sha256sum -c -
sha256sum -c capture-source-manifest.sha256
while read -r repo pin; do
  [ "$(git -C "$AREA/$repo" rev-parse HEAD)" = "$pin" ] || exit 2
  printf 'pin %s %s\n' "$repo" "$pin"
done < source-pins.txt
while read -r hash source; do
  [ "$PLUGIN" -nt "$source" ] && [ "$LIBDIR/libKinKal_Fit.so.0.0.0" -nt "$source" ] || {
    echo "Stale binary relative to $source"; exit 2;
  }
  stat -c '%y %n' "$source"
done < capture-source-manifest.sha256
stat -c '%y %n' "$PLUGIN" "$LIBDIR/libKinKal_Fit.so.0.0.0"
ldd "$PLUGIN" | tee "$OUT/linkage.txt"
! grep -q 'not found' "$OUT/linkage.txt" || exit 2
while read -r lib arrow resolved rest; do
  case "$lib" in libKinKal_*|libmu2e_Mu2eKinKal*)
    case "$(readlink -f "$resolved")" in "$LIBDIR/"*) ;; *) echo "Nonlocal dependency $lib $resolved"; exit 2;; esac;;
  esac
done < "$OUT/linkage.txt"
grep -q libKinKal_Fit "$OUT/linkage.txt"
for component in Detector Examples Fit General Geometry MatEnv Trajectory; do
  built=$AREA/kinkal_build_p109/lib/libKinKal_$component.so
  installed=$LIBDIR/libKinKal_$component.so.0.0.0
  objcopy --dump-section .text="$OUT/$component.built.text" "$built" /dev/null
  objcopy --dump-section .text="$OUT/$component.installed.text" "$installed" /dev/null
  cmp "$OUT/$component.built.text" "$OUT/$component.installed.text"
  sha256sum "$built" "$installed" "$OUT/$component.built.text" "$OUT/$component.installed.text"
  readelf -d "$installed" | grep -E 'NEEDED|SONAME'
done
sha256sum "$PLUGIN" "$0" capture-source-manifest.sha256 "$DATA/crossing_preparation_v1.json"
sha256sum -c capture-source-manifest.sha256
printf '%s\n' CROSSING_BUILD_VERIFIED > "$OUT/completed.txt"
echo CROSSING_BUILD_VERIFIED
