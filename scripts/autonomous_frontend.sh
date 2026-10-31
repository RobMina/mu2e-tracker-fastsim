#!/bin/bash
# Evaluate detector-only hit and seed ports against dump columns in separate diagnostic commands.
set -e -o pipefail
DUMP=${1:?dump ROOT}; COND=${2:?conditions JSON}; SETTINGS=${3:?mc or data}; OUT=${4:?new directory}
case "$SETTINGS" in mc|data) ;; *) echo 'settings must be mc or data' >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")/.." && pwd)
CORE=${M2FS_CORE:-${M2FS_CODE:?set M2FS_CORE, or M2FS_CODE to the mu2e-fastsim checkout}/core}
export PYTHONPATH="$HERE:$CORE" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=${PY_JAX:?set PY_JAX to a Python with JAX, NumPy and uproot}
CPU=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
mkdir "$OUT"
taskset -c "$CPU" "$PY" -m trkfastsim reco-vr1 "$DUMP" --conditions "$COND" --settings "$SETTINGS" --outdir "$OUT/hits"
taskset -c "$CPU" "$PY" -m trkfastsim reco-vr1b "$DUMP" --conditions "$COND" --settings "$SETTINGS" --jax --outdir "$OUT/seeds"
