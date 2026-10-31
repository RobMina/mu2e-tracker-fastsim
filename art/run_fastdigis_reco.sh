#!/bin/bash
# Stock KKLine on a supplied fast table; prove injected digis match the table separately.
set -eo pipefail
FCL=${1:?reco FCL}; INPUT=${2:?original art event context}
[[ $# == 2 ]] || exit 2
[[ -n ${MUSE_WORK_DIR:-} && $(readlink -f "$MUSE_WORK_DIR") == "$(readlink -f "$(dirname "$0")")" ]] || exit 2
fhicl-dump -c "$FCL" > fast_reco.expanded.fcl
args=(-c "$FCL" -s "$INPUT" -T "$PWD/fast_reco.root")
[[ -z ${OFFLINE_EVENTS:-} ]] || args+=(-n "$OFFLINE_EVENTS")
"$OFFLINE_TIME" -v -o fast_reco.art.time mu2e "${args[@]}" > fast_reco.log 2>&1
printf 'completed=%s\n' "$(date -u +%FT%TZ)" > sequence.done
