#!/bin/bash
# steps --backend jax against numpy on one CPU core (GPVM): jaxcheck per gen_as_v1 shard, both backends pinned to one
# core (taskset, single-threaded XLA and BLAS), two passes each (reproducibility; the first jax pass compiles), tables
# saved for the cross-device comparison with a GPU run (jaxcheck --cross).
# usage: tracker/scripts/jax_steps_cpu.sh OUTDIR [SHARD ...]   (shards default 1001-1005; env PY, CORE (cpu index), GDML,
#   LANES: jax muons per chunk, default 1024 here: a GPVM has ~11 GB; the outputs do not depend on it)
set -u
O=${1:?outdir}; shift
[ $# -gt 0 ] || set -- 1001 1002 1003 1004 1005
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
TRK=${TRK:-$(cd "$(dirname "$0")/.." && pwd)}
D=${M2FS_DATA:-/exp/mu2e/data/users/ram2aq}
PY=${PY:-$D/crv_fastsim/venv_eaf/bin/python3}
GDML=${GDML:-$D/crv_fastsim/geometry/mu2e_extracted_v04.gdml}
export PYTHONPATH=${CORE_DIR:-$M2FS_CODE/core}:$TRK JAX_PLATFORMS=cpu
# one codegen part per executable: the default splits multiply the memory maps past vm.max_map_count (65530)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_cpu_parallel_codegen_split_count=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$O"
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$(dirname "$O")/jaxcache_cpu}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
{ hostname; lscpu | grep -E "^Model name|^CPU\(s\)"; "$PY" -c "import jax, numpy; print('jax', jax.__version__, jax.devices(), 'numpy', numpy.__version__)"
  echo "trk $TRK"; } > "$O/env.txt" 2>&1
rc=0
for s in "$@"; do
  f=$D/trk_fastsim/gen_as_v1/gen.001400_0000$s.root
  /usr/bin/time -f "%e s %M kB" -o "$O/jc_$s.time" taskset -c "${CORE:-0}" "$PY" -m trkfastsim jaxcheck "$f" --gdml "$GDML" \
    --stage C --seed 1 --repeat "${REPEAT:-2}" --numpy-repeat "${NUMPY_REPEAT:-2}" --lanes "${LANES:-1024}" --save --outdir "$O/jc_$s" > "$O/jc_$s.log" 2>&1 || { echo "FAILED $s"; exit 1; }
  echo "== $s $(cat "$O/jc_$s.time")"; grep -E "pass|reproducible|PrimaryFilter" "$O/jc_$s.log" | cut -c1-220
done
echo "exit=$rc"
exit "$rc"
