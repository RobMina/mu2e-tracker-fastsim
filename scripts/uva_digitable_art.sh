#!/bin/bash
# Four sequential art jobs share one private node-local CVMFS namespace.
set -euo pipefail
ROOT=${ROOT:?task output root}
MODE=${MODE:?mc or kpp}
TAG=${TAG:?new output tag}
INPUT_ROOT=${INPUT_ROOT:?shared read-only original inputs}
case "$MODE" in
  mc) INPUT="$INPUT_ROOT/trkfs_digi_20261026/ref/dig.v18L_1001.art" ;;
  kpp) INPUT="$INPUT_ROOT/crv_data_124155/unpack_check_head/dig_fullmap_head.art" ;;
  *) exit 2 ;;
esac
OUT="$ROOT/${MODE}_${TAG}"
mkdir "$OUT"
cd "$OUT"
module load apptainer
export CVMFSEXEC_DIST=${CVMFSEXEC_DIST:?saved EL8 CVMFS client}
export OFFLINE_FCL_PATH="$ROOT/runtime/fcl"
export OFFLINE_BINDS="$ROOT:$ROOT"
export OFFLINE_LOG_PREFIX=sequence
AREA=${AREA:-$ROOT/built_area}
RUNNER=${RUNNER:-$ROOT/runtime/offline_sequence.sh}
bash "$RUNNER" "$AREA" \
  "$AREA/TrkFastSim/fcl/dump_digis.fcl" "$INPUT" > runtime.log 2>&1
