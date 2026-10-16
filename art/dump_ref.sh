#!/bin/bash
# Dump dts art files to flat tracker truth tables with StrawGasStepTableDumper, serially on one GPVM.
# usage: dump_ref.sh <file list> <output dir>   (list: one dts path per line; gen.* paths use the gen-only fcl)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${TRKFS_CODE:?set TRKFS_CODE to this repository}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$TRKFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
LIST=$(readlink -f "${1:?file list}")
O=${2:?output dir}
mkdir -p "$O/logs"
while read -r F; do
  [ -n "$F" ] || continue
  B=$(basename "$F" .art)
  case "$B" in gen.*) FCL=TrkFastSim/fcl/dump_sgsteps_gen.fcl; OUT=$O/gen_tables.$B.root ;;
               *)     FCL=TrkFastSim/fcl/dump_sgsteps.fcl;     OUT=$O/sgsteps.$B.root ;; esac
  [ -s "$OUT" ] || mu2e -c $FCL -s "$F" -T "$OUT" > "$O/logs/$B.log" 2>&1
  echo "$B rc=$? $(date +%T)"
done < "$LIST"
echo "done $(date -Is)"
