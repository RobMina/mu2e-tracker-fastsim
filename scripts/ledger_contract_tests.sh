#!/usr/bin/env bash
# Run supplied-history tests and legacy controls in isolated processes with a pinned core.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "usage: $0 PYTHON3 CORE_SOURCE NEW_OUTPUT_DIR" >&2
  exit 2
fi
PY=$1
CORE=$(cd "$2" && pwd)
OUT=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$ROOT:$CORE"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
for file in test_transportledger.py test_flighttime.py test_historystate.py test_historysurfaces.py \
  test_rate_ledger.py test_rate_ledger_limits.py test_step_energy_ledger.py test_surfsteps.py test_surfsteps_vertices.py; do
  "$PY" -m pytest -q -p no:cacheprovider "tests/$file" 2>&1 | tee "$OUT/${file%.py}.log"
done
printf '%s\n' "All isolated ledger contract/legacy controls passed (strict legacy xfails retained)." > "$OUT/complete.txt"
