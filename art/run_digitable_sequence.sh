#!/bin/bash
# Execute detector replay and stock reco checks in one already-configured runtime.
set -eo pipefail
FCL=${1:?dump FCL}; INPUT=${2:?original art input}
[[ $# == 2 ]] || { echo 'expected one input file' >&2; exit 2; }
[[ -n ${MUSE_WORK_DIR:-} && $(readlink -f "$MUSE_WORK_DIR") == "$(readlink -f "$(dirname "$0")")" ]] || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
run_art() {
  local name=$1 fcl=$2
  shift 2
  fhicl-dump -c "$fcl" > "$name.expanded.fcl"
  local args=(-c "$fcl" -s "$INPUT" -T "$PWD/$name.root")
  [[ -z ${OFFLINE_EVENTS:-} ]] || args+=(-n "$OFFLINE_EVENTS")
  "$OFFLINE_TIME" -v -o "$name.art.time" mu2e "${args[@]}" "$@" > "$name.log" 2>&1
}
run_art digis "$FCL"
printf '#include "TrkFastSim/fcl/replay_digis.fcl"\nphysics.producers.tableSD.tableFile : "%s/digis.root"\n' "$PWD" > replay.fcl
run_art replay replay.fcl
MODE=mc
[[ $PWD != *kpp* ]] || MODE=data
BASE=TrkFastSim/fcl/dump_tracks_mc_saveall.fcl
[[ $MODE != data ]] || BASE=TrkFastSim/fcl/dump_tracks_data_autonomous.fcl
printf '#include "%s"\n' "$BASE" > original_reco.fcl
printf '#include "TrkFastSim/fcl/reco_digis_table_%s.fcl"\nphysics.producers.tableSD.tableFile : "%s/digis.root"\n' "$MODE" "$PWD" > replay_reco.fcl
if [[ ${PAYLOAD:-0} == 1 ]]; then
  printf '#include "TrkFastSim/fcl/reco_payload.fcl"\n' >> original_reco.fcl
  printf '#include "TrkFastSim/fcl/reco_payload.fcl"\n' >> replay_reco.fcl
fi
run_art original_reco original_reco.fcl
run_art replay_reco replay_reco.fcl
printf 'sequence_completed=%s events_limit=%s\n' "$(date -u +%FT%TZ)" "${OFFLINE_EVENTS:-all}" > sequence.done
