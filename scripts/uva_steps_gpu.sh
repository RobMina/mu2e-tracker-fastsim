#!/bin/bash
#SBATCH -J m2fs_trkgpu
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:55:00
#SBATCH -o /scratch/%u/mu2efastsim/kpp/slurm-%j.out
# UVA (Slurm, one whole A100-80): steps --backend jax on the GPU against numpy (jaxcheck per gen shard, warm passes
# timed apart from the compiling one), the cross-device comparison with a CPU JAX run of the same seed (--cross), and
# the throughput at large batches (--concat, several lane counts). Submitted and fed by uva_steps_stream.sh: the code
# tarball and the inputs (gen tables, GDML, CPU tables) are streamed into job tmpfs and deleted on exit; only stats
# (json/txt) are written to the output dir, which the stream script copies back and removes.
# env: SHARDS (default 1001 1002 1003 1004 1005), LANES_LIST (default 4096 65536), REPEAT (2: compile + warm)
F=${M2FS_UVA:-/scratch/$USER/mu2efastsim}
SIF=${SIF:-$F/mu2efastsim.sif}
umask 077
O=$F/kpp/trkgpu_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-none}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
module load apptainer >/dev/null 2>&1
L=${TMPDIR:-/tmp}/crvfs_trk_j${SLURM_JOB_ID:-0}_$$; mkdir -p $L/code/tracker; chmod 700 $L   # --pwd needs the dir
trap 'rm -rf $L; echo "$(date -Is) $(hostname) deleted $L" >> $O/cleanup.txt; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
OR=$(readlink -f $O)
A() { apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      --env JAX_COMPILATION_CACHE_DIR=${JAXCACHE:-$L/jaxcache} --env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
      --env PYTHONPATH=$L/code/core:$L/code/tracker --env PYTHONDONTWRITEBYTECODE=1 -B $L -B $OR --pwd $L/code/tracker "$SIF" "$@"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out=$O sif=$SIF"
nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm,mig.mode.current --format=csv,noheader; nvidia-smi -L
lscpu | grep -E "^Model name"; echo "cpus $(nproc)"
A python -c "import jax, sys; print(jax.__version__, jax.devices()); sys.exit(jax.default_backend() != 'gpu')" 2>&1 | tail -3
A python -c "import jax, sys; sys.exit(jax.default_backend() != 'gpu')" 2>/dev/null \
  || { echo "JAX is not on the GPU: abort"; exit 5; }
echo "$(hostname) $L $O" > $F/kpp/inbox_${SLURM_JOB_ID}   # read by uva_steps_stream.sh
echo "waiting for $L/ready (up to 20 min)"
for i in $(seq 240); do [ -f $L/ready ] && break; sleep 5; done
rm -f $F/kpp/inbox_${SLURM_JOB_ID}
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
mkdir -p $L/code && tar -C $L/code -xzf $L/code.tgz && rm -f $L/code.tgz && ls $L/code && cat $L/code/VERSION.txt
mkdir -p $L/cross && for t in $L/fasttrkCjx.*.root; do [ -f "$t" ] && mv "$t" $L/cross/; done
ls -la $L $L/cross | grep -v ready | awk 'NR>1{print $5, $9}'
A python -c "import trkfastsim, mu2efastsim, pandas, uproot; print(trkfastsim.__file__, mu2efastsim.__file__, pandas.__version__, uproot.__version__)"
G=$L/mu2e_extracted_v04.gdml
sec() { echo; echo "==== $* $(date +%T)"; }
rc=0
sec "pytest tests/test_steps_jax.py (GPU)"
A python -m pytest -q -p no:cacheprovider tests/test_steps_jax.py > $O/pytest.txt 2>&1 || rc=8
tail -3 $O/pytest.txt
for s in ${SHARDS:-1001 1002 1003 1004 1005}; do
  sec "jaxcheck shard $s: numpy (job CPU) vs jax (GPU), cross vs CPU JAX"
  A python -m trkfastsim jaxcheck $L/gen.001400_0000$s.root --gdml $G --stage C --seed 1 --repeat ${REPEAT:-2} \
    --numpy-repeat 1 --cross $L/cross --kernel-times --outdir $OR/jc_$s > $O/jc_$s.log 2>&1 || { echo "FAILED $s"; rc=6; }
  grep -E "pass|reproducible|PrimaryFilter|cross-device|kernel times" $O/jc_$s.log | cut -c1-400
done
for ln in ${LANES_LIST:-4096 65536}; do
  sec "throughput: all shards in one call, lanes $ln"
  A python -m trkfastsim jaxcheck $L/gen.001400_0000*.root --concat --lanes $ln --gdml $G --stage C --seed 1 \
    --repeat ${REPEAT:-2} --numpy-repeat 0 --kernel-times --outdir $OR/concat_l$ln > $O/concat_l$ln.log 2>&1 || { echo "FAILED concat $ln"; rc=7; }
  grep -E "pass|reproducible|kernel times" $O/concat_l$ln.log | cut -c1-300; tail -2 $O/concat_l$ln.log | cut -c1-300
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "outdir $O elapsed $SECONDS s"; echo exit=$rc; exit $rc
