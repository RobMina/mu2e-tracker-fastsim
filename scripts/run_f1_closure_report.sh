#!/bin/bash
# f1_closure_report.py on one CPU with the GPVM JAX venv. Usage: run_f1_closure_report.sh OUT.json NAME=DIR...
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/exp/mu2e/app/users/ram2aq/analysis
DATA=/exp/mu2e/data/users/ram2aq/trk_fastreco
OUT=$1; shift
args=(); for a in "$@"; do args+=(--arm "$a"); done
export PYTHONPATH="$HERE:$ROOT/mu2e-fastsim/core" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
exec taskset -c "$CORE" /exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3 "$HERE/scripts/f1_closure_report.py" "${args[@]}" \
  --trace ${F1C_TRACE:-$DATA/p109_capture_3k_20261025/control.root} \
  --paired ${F1C_PAIRED:-$DATA/p109_prior_all_20261025/paired.npz} --out "$OUT"
