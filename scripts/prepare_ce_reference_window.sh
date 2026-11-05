#!/bin/bash
# Copy a disjoint sequential stop-record window without generation or transport.
set -eo pipefail
FCL=${1:?copy FCL}
OUT=${2:?fresh output directory}
INPUT=${3:?original upstream art URL}
test -n "${MUSE_WORK_DIR:-}"
mkdir "$OUT"
cd "$OUT"
printf 'source=%s\nskip_records=2000\ncopy_records=2000\nrecord_window_zero_based=[2000,4000)\nstart_utc=%s\n' "$INPUT" "$(date -u +%FT%TZ)" > receipt.txt
fhicl-dump -c "$FCL" > copy.expanded.fcl
/usr/bin/time -v -o copy.time timeout 900 mu2e -c copy.expanded.fcl --nskip 2000 -n 2000 -s "$INPUT" > copy.log 2>&1
config_dumper -P stop_pool.art > pool.config.txt
sha256sum stop_pool.art copy.expanded.fcl > MANIFEST.sha256
grep 'TrigReport Events total' copy.log
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
