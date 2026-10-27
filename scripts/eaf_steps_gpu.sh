#!/bin/bash
# steps --backend jax on a Fermilab EAF A100 session (run in the EAF terminal): jaxcheck per gen_as_v1 shard (numpy on
# the session CPU vs jax on the GPU, warm passes timed apart from the compiling one, cross-device comparison with the
# CPU JAX tables of jax_steps_cpu.sh) and the throughput at large batches (--concat, several lane counts).
# Inputs are read in place from /exp/mu2e/data; outputs go to $D/trkfs_gpu_eaf_<time>.
# usage: bash tracker/scripts/eaf_steps_gpu.sh [SHARD ...]   (default 1001-1005)
# env: VENV, OUT, CPU_OUT (jax_steps_cpu.sh outdir, for --cross), REPEAT (3), LANES_LIST (4096 16384 65536), CPU=1 (no GPU)
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
TRK=${TRK:-$(cd "$(dirname "$0")/.." && pwd)}
D=${M2FS_DATA:-/exp/mu2e/data/users/ram2aq}
VENV=${VENV:-$D/crv_fastsim/venv_eaf}
BASE=/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python
JAXPKG="jax[cuda12]==0.11.2"; [ -n "${CPU:-}" ] && JAXPKG="jax==0.11.2"
[ -x "$VENV/bin/python" ] || "$BASE" -m venv --system-site-packages "$VENV" || exit 2
"$VENV/bin/python" -c "import jax" 2>/dev/null || "$VENV/bin/python" -m pip install -q "$JAXPKG" || exit 3
PY=$VENV/bin/python
export PYTHONPATH=$M2FS_CODE/core:$TRK XLA_PYTHON_CLIENT_PREALLOCATE=false
GDML=$D/crv_fastsim/geometry/mu2e_extracted_v04.gdml
CPU_OUT=${CPU_OUT:-$D/trkfs_gpu_20261026/cpu_gpvm03}
O=${OUT:-$D/trkfs_gpu_eaf_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$O" || exit 4
[ $# -gt 0 ] || set -- 1001 1002 1003 1004 1005
{ hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
  nvidia-smi -L 2>/dev/null; "$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"; } | tee "$O/env.txt"
if [ -z "${CPU:-}" ] && ! grep -q -i cuda "$O/env.txt"; then echo "JAX sees no GPU: start the session with the A100 profile"; exit 5; fi
rc=0
(cd "$TRK" && "$PY" -m pytest -q -p no:cacheprovider tests/test_steps_jax.py > "$O/pytest.txt" 2>&1) || rc=8; tail -2 "$O/pytest.txt"
G=()
for s in "$@"; do
  f=$D/trk_fastsim/gen_as_v1/gen.001400_0000$s.root; G+=("$f")
  "$PY" -m trkfastsim jaxcheck "$f" --gdml "$GDML" --stage C --seed 1 --repeat "${REPEAT:-3}" --numpy-repeat 1 \
    --cross "$CPU_OUT/jc_$s" --outdir "$O/jc_$s" > "$O/jc_$s.log" 2>&1 || { echo "FAILED $s"; rc=6; }
  echo "== $s"; grep -E "pass|reproducible|PrimaryFilter|cross-device" "$O/jc_$s.log" | cut -c1-300
done
for ln in ${LANES_LIST:-4096 16384 65536}; do
  "$PY" -m trkfastsim jaxcheck "${G[@]}" --concat --lanes "$ln" --gdml "$GDML" --stage C --seed 1 --repeat "${REPEAT:-3}" \
    --numpy-repeat 0 --outdir "$O/concat_l$ln" > "$O/concat_l$ln.log" 2>&1 || { echo "FAILED concat $ln"; rc=7; }
  echo "== concat lanes $ln"; grep -E "pass" "$O/concat_l$ln.log" | cut -c1-300
done
echo "outdir $O"; echo "exit=$rc"; exit $rc
