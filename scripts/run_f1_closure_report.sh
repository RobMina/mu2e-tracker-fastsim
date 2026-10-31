#!/bin/bash
# f1_closure_report.py on one CPU with the GPVM JAX venv. Usage: run_f1_closure_report.sh OUT.json NAME=DIR...
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout (core/ is put on PYTHONPATH)}"
: "${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}"
DATA=${TRK_FASTRECO_DATA:-${M2FS_DATA:?set M2FS_DATA (or TRK_FASTRECO_DATA) to the data area}/trk_fastreco}
OUT=$1; shift
args=(); for a in "$@"; do args+=(--arm "$a"); done
export PYTHONPATH="$HERE:$M2FS_CODE/core" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
exec taskset -c "$CORE" "$PY_JAX" "$HERE/scripts/f1_closure_report.py" "${args[@]}" \
  --trace ${F1C_TRACE:-$DATA/p109_capture_3k_20261025/control.root} \
  --paired ${F1C_PAIRED:-$DATA/p109_prior_all_20261025/paired.npz} --out "$OUT"
