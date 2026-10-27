#!/bin/bash
# Report only a fully completed, immutable set of closure arms; no fitting or build commands.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$1
REPORT=$2
ARMS=(own_legacy own_stock own_stock_Lcons own_stock_Lprior own_stock_Lbounds own_stock_Lstatusref own_stock_Lseedx own_stock_Ltransit own_stock_xdoca own_legacy_xdoca fixed_legacy fixed_stock fixed_legacy_xstock fixed_stock_xstock fixed_stock_pstock own_stock_xstock own_legacy_xstock own_stock_gasp109 own_legacy_gasp109 fixed_stock_gasp109)
for attempt in $(seq 1 360); do
  ready=true
  for arm in "${ARMS[@]}"; do
    [[ -f "$OUT/$arm/vr2f1.json" ]] || ready=false
  done
  $ready && break
  sleep 20
done
$ready || { echo "incomplete campaign; no report written" >&2; exit 2; }
[[ ! -e "$REPORT" ]] || { echo "refusing existing report $REPORT" >&2; exit 2; }
args=()
for arm in "${ARMS[@]}"; do
  cmp "$OUT/own_legacy/source.sha256" "$OUT/$arm/source.sha256"
  args+=("$arm=$OUT/$arm")
done
bash "$HERE/scripts/run_f1_closure_report.sh" "$REPORT" "${args[@]}"
/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3 "$HERE/scripts/f1_closure_summary.py" "$REPORT" --out "${REPORT%.json}.md"
sha256sum "$HERE/scripts/"f1_closure_{report,summary}.py > "${REPORT%.json}.source.sha256"
