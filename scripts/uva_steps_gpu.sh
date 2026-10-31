#!/bin/bash
#SBATCH -J m2fs_trkgpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:55:00
# UVA (Slurm, one whole A100-80): steps --backend jax on the GPU against numpy (jaxcheck per gen shard, warm passes
# timed apart from the compiling one), the cross-device comparison with a CPU JAX run of the same seed (--cross), and
# the throughput at large batches (--concat, several lane counts). Submitted and fed by uva_steps_stream.sh: the code
# tarball and the inputs (gen tables, GDML, CPU tables) are streamed into job tmpfs and deleted on exit; only stats
# (json/txt) are written to the output dir, which the stream script copies back and removes. The stream script passes
# the account (-A $FASTSIMKIT_SLURM_ACCOUNT), the log path ($M2FS_UVA/kpp/slurm-%j.out) and M2FS_UVA.
# env: SHARDS (1001-1005), VALIDATION_SHARDS (SHARDS), LANES_LIST (65536), REPEAT (2: compile + warm)
F=${M2FS_UVA:-${FASTSIMKIT_SCRATCH:?set M2FS_UVA or FASTSIMKIT_SCRATCH to the project work area on the cluster}}
SIF=${SIF:-$F/mu2efastsim.sif}
umask 077
O=$F/kpp/trkgpu_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-none}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
module load apptainer >/dev/null 2>&1
L=${TMPDIR:-/tmp}/crvfs_trk_j${SLURM_JOB_ID:-0}_$$; mkdir -p $L/code/tracker; chmod 700 $L   # --pwd needs the dir
cleanup() { rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$O/cleanup.txt"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
OR=$(readlink -f $O)
A() { apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      --env JAX_COMPILATION_CACHE_DIR=${JAXCACHE:-$L/jaxcache} --env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
      --env PYTHONPATH=$L/code/core:$L/code/tracker --env PYTHONDONTWRITEBYTECODE=1 -B $L -B $OR --pwd $L/code/tracker "$SIF" "$@"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out=$O sif=$SIF"
nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm,mig.mode.current --format=csv,noheader; nvidia-smi -L
lscpu | grep -E "^Model name"; echo "cpus $(nproc)"
A python3 -c "import jax, sys; print(jax.__version__, jax.devices()); sys.exit(jax.default_backend() != 'gpu')" 2>&1 | tail -3
A python3 -c "import jax, sys; sys.exit(jax.default_backend() != 'gpu')" 2>/dev/null \
  || { echo "JAX is not on the GPU: abort"; exit 5; }
echo "$(hostname) $L $O" > $F/kpp/inbox_${SLURM_JOB_ID}   # read by uva_steps_stream.sh
echo "waiting for $L/ready (up to 20 min)"
for i in $(seq 240); do [ -f $L/ready ] && break; sleep 5; done
rm -f $F/kpp/inbox_${SLURM_JOB_ID}
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
mkdir -p $L/code && tar -C $L/code -xzf $L/code.tgz && rm -f $L/code.tgz && ls $L/code && cat $L/code/VERSION.txt
mkdir -p $L/cross && for t in $L/fasttrkCjx.*.root; do [ -f "$t" ] && mv "$t" $L/cross/; done
ls -la $L $L/cross | grep -v ready | awk 'NR>1{print $5, $9}'
A python3 -c "import trkfastsim, mu2efastsim, pandas, uproot; print(trkfastsim.__file__, mu2efastsim.__file__, pandas.__version__, uproot.__version__)"
G=$L/mu2e_extracted_v04.gdml
sec() { echo; echo "==== $* $(date +%T)"; }
rc=0
sec "pytest transport/world/validation kernels (GPU)"
A python3 -m pytest -q -p no:cacheprovider tests/test_steps_jax.py tests/test_world_jax.py tests/test_slab_rounds_jax.py tests/test_jaxcheck_clusters.py tests/test_jaxcheck_audit.py > $O/pytest.txt 2>&1 || { tail -60 $O/pytest.txt; exit 8; }
tail -3 $O/pytest.txt
for s in ${VALIDATION_SHARDS:-${SHARDS:-1001 1002 1003 1004 1005}}; do
  sec "jaxcheck shard $s: numpy reference vs jax (GPU), cross vs CPU JAX"
  if [ "${NUMPY_REPEAT:-1}" = 0 ]; then
    mkdir -p $L/validation/jc_$s
    [ -f "$L/fasttrkCnp.001400_0000$s.root" ] || { echo "missing saved numpy reference $s"; exit 11; }
    mv $L/fasttrkCnp.001400_0000$s.root $L/validation/jc_$s/
  fi
  A python3 -m trkfastsim jaxcheck $L/gen.001400_0000$s.root --gdml $G --stage C --seed 1 --repeat ${REPEAT:-2} \
    --numpy-repeat "${NUMPY_REPEAT:-1}" --cross $L/cross --save --outdir $L/validation/jc_$s > $O/jc_$s.log 2>&1 || { echo "FAILED $s"; exit 6; }
  grep -E "pass|reproducible|PrimaryFilter|cross-device|kernel times" $O/jc_$s.log | cut -c1-400
  mkdir -p $O/jc_$s
  cp $L/validation/jc_$s/jaxcheck.{json,txt} $O/jc_$s/
done
A python3 scripts/jaxcheck_merge.py $OR/combined.json $L/validation/jc_* > $O/combined.txt 2>&1 || exit 9
A python3 scripts/jaxcheck_audit.py --require-cross --out $OR/audit.json $OR/jc_*/jaxcheck.json > $O/audit.txt 2>&1 || rc=10
for ln in ${LANES_LIST:-65536}; do
  sec "throughput: all shards in one call, lanes $ln"
  A python3 -m trkfastsim jaxcheck $L/gen.001400_0000*.root --concat --lanes $ln --gdml $G --stage C --seed 1 \
    --repeat ${REPEAT:-2} --numpy-repeat 0 --kernel-times --outdir $OR/concat_l$ln > $O/concat_l$ln.log 2>&1 || { echo "FAILED concat $ln"; rc=7; }
  grep -E "pass|reproducible|kernel times" $O/concat_l$ln.log | cut -c1-300; tail -2 $O/concat_l$ln.log | cut -c1-300
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "outdir $O elapsed $SECONDS s"; echo exit=$rc; exit $rc
