#!/bin/bash
# Repeat stock reco only as a seed-payload reproducibility control, not a new replay baseline.
set -eo pipefail
[[ -n ${MUSE_WORK_DIR:-} && $(readlink -f "$MUSE_WORK_DIR") == "$(readlink -f "$(dirname "$0")")" ]] || exit 2
FCL=${1:?saved original reco configuration}; INPUT=${2:?original MC art}
[[ $# == 2 ]] || exit 2
fhicl-dump -c "$FCL" > control.expanded.fcl
"$OFFLINE_TIME" -v -o control.art.time mu2e -c "$FCL" -s "$INPUT" -n 10 -T "$PWD/control.root" > control.log 2>&1
printf 'stock_reproducibility_control_completed\n' > control.done
