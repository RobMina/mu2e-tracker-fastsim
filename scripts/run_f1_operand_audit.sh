#!/bin/bash
# Run one saved-operand diagnostic on one CPU; caller owns host lock.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=/exp/mu2e/data/users/ram2aq
MODE=$1; OUT=$2; shift 2
case "$MODE" in
  ca) SCRIPT=f1_ca_operand_audit.py ;;
  material) SCRIPT=f1_material_audit.py ;;
  population) SCRIPT=f1_ca_population_audit.py ;;
  captured) SCRIPT=f1_captured_ca_audit.py ;;
  captured-population) SCRIPT=f1_captured_ca_population.py ;;
  *) echo "mode must be ca, material, population, captured or captured-population" >&2; exit 2 ;;
esac
if test -e "$OUT"; then echo "refusing existing operand audit" >&2; exit 2; fi
export PYTHONPATH="$HERE:/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/core"
export MU2E_SEARCH_PATH=/exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
cd "$HERE"
taskset -c "$CORE" "$DATA/crv_fastsim/venv_eaf/bin/python3" "scripts/$SCRIPT" \
  --check "$DATA/trk_fastreco/p109_iteration_3k_20261027/check.json" \
  --native-source /exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026 --out "$OUT" "$@"
