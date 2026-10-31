#!/usr/bin/env bash
# Monitor one authorized cluster and collect only complete products; never submit or cancel jobs.
: "${TRKFS_OWNER:?set TRKFS_OWNER, the owner field of the tracker-stage Mu2e file names}"
set -euo pipefail
AUDIT=${1:?audit directory}
OLD=${2:?saved campaign directory}
DATA=${3:?tracker data directory}
BASE=${4:?tracker pnfs root}
RESULT=${5:?new result directory}
CLUSTER=${6:?numeric cluster id}
TRK=$(cd "$(dirname "$0")/.." && pwd)
[[ "$CLUSTER" =~ ^[0-9]+$ ]]
mkdir "$AUDIT/watch"
trap 'rc=$?; printf "%s\n" "$rc" > "$AUDIT/watch/exit_code"' EXIT
export PYTHONPATH="$TRK:${CORE:?set pinned CORE}:${CRVEXTRAP:?set crvextrap import directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for poll in $(seq 1 90); do
  prefix="$AUDIT/watch/$(printf '%03d' "$poll")"
  date -u | tee "$prefix.time"
  jobsub_q -G mu2e --user "$USER" > "$prefix.queue"
  for item in nts hits logs; do
    sub=""; [[ "$item" == nts ]] || sub="/$item"
    ifdh ls "$BASE/tv2p_fast$sub" 1 > "$prefix.$item"
  done
  complete=0
  for first in $(seq 21 10 220); do
    s=$(printf '001400_%08d' "$first")
    if grep -Fxq "$BASE/tv2p_fast/nts.${TRKFS_OWNER}.tv2fast.v0.$s.root" "$prefix.nts" &&
       grep -Fxq "$BASE/tv2p_fast/hits/strawhits.${TRKFS_OWNER}.tv2fast.v0.$s.root" "$prefix.hits" &&
       grep -Fxq "$BASE/tv2p_fast/logs/logs.$s.tgz" "$prefix.logs"; then complete=$((complete+1)); fi
  done
  echo "complete_groups=$complete/20 (includes retained pilot)"
  grep -E "^$CLUSTER[.]" "$prefix.queue" || true
  if (( complete == 20 )); then
    bash "$TRK/scripts/secondary_tv3_collect.sh" "$OLD" "$DATA" "$BASE" "$RESULT"
    echo 'Collection and paired comparison completed'
    exit 0
  fi
  if ! grep -Eq "^$CLUSTER[.]" "$prefix.queue"; then
    echo 'ERROR: cluster absent but products incomplete; inspect logs, do not resubmit automatically'
    exit 2
  fi
  if grep -E "^$CLUSTER[.]" "$prefix.queue" | grep -Eq '[[:space:]]H[[:space:]]'; then
    echo 'ERROR: authorized cluster has held jobs; inspect, do not resubmit automatically'
    exit 3
  fi
  sleep 120
done
echo 'ERROR: monitoring window ended; inspect queue before any next action'
exit 4
