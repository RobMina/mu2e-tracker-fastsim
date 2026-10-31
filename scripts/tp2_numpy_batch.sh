#!/bin/bash
# One-core numpy throughput for the same ten-shard input batch as the device benchmark.
set -euo pipefail
O=${1:?new output directory}
D=${M2FS_DATA:?set M2FS_DATA to the data area}
TRK=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-${PY_JAX:?set PY or PY_JAX to a Python with JAX, NumPy and uproot}}
: "${CORE_DIR:?pinned core directory}"
[ ! -e "$O" ] || { echo "refusing existing output $O"; exit 2; }
export PYTHONPATH=$CORE_DIR:$TRK JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_cpu_parallel_codegen_split_count=1"
exec taskset -c "${CPU_CORE:-0}" "$PY" -m trkfastsim jaxcheck \
  "$D"/trk_fastsim/gen_as_v1/gen.001400_0000{1001..1010}.root \
  --gdml "$D/crv_fastsim/geometry/mu2e_extracted_v04.gdml" --stage C --seed 1 \
  --concat --repeat 0 --numpy-repeat 2 --outdir "$O"
