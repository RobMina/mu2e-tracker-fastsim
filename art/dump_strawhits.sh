#!/bin/bash
# T-VR1 reference: dump stock hit reco on dig files with StrawHitTableDumper, serially on one GPVM (needs the built TrkFastSim area).
# usage: dump_strawhits.sh <file list> <output dir> [mc|data] [first line] [last line] [nevents, -1 = all]
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${TRKFS_CODE:?set TRKFS_CODE to this repository}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$TRKFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
LIST=$(readlink -f "${1:?file list}"); O=${2:?output dir}; KIND=${3:-mc}; A=${4:-1}; B=${5:-$A}; N=${6:--1}
mkdir -p "$O/logs"
sed -n "${A},${B}p" "$LIST" | while read -r F; do
  [ -n "$F" ] || continue
  SEQ=$(basename "$F" .art | awk -F. '{print $NF}')
  OUT=$O/strawhits_ref_$KIND.$SEQ.root
  [ -s "$OUT" ] || mu2e -c TrkFastSim/fcl/dump_strawhits_$KIND.fcl -s "$F" -n "$N" -T "$OUT" > "$O/logs/$SEQ.log" 2>&1
  echo "$SEQ rc=$? $(grep -m1 'Events total' "$O/logs/$SEQ.log") $(date +%T)"
done
echo "done $(date -Is)"
