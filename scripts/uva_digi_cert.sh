#!/bin/bash
# Run a bounded real-table benchmark in the project image on a Slurm allocation.
set -euo pipefail
CODE=${CODE:?core and tracker snapshot directory}
OUT=${OUT:?new output directory}
INPUTS=${INPUTS:?shared input root}
BACKEND=${BACKEND:-jax}
DEVICE=${DEVICE:-cpu}
LIMIT=${LIMIT:-200}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
SHARDS=${SHARDS:-21}
SIF=${SIF:?project image}
module load apptainer
nv=(); platform=cpu
if [[ $DEVICE == gpu ]]; then nv=(--nv); platform=cuda; fi
extra=()
if [[ ${PROFILE_WARM:-0} == 1 ]]; then extra+=(--profile-warm); fi
if [[ ${PROFILE_STAGES:-0} == 1 ]]; then extra+=(--profile-stages); fi
if [[ -n ${WAVEFORM_LANES:-} ]]; then extra+=(--waveform-lanes "$WAVEFORM_LANES"); fi
if [[ -n ${STRAW_BATCH:-} ]]; then extra+=(--straw-batch "$STRAW_BATCH"); fi
if [[ -n ${CLUSTER_BATCH:-} ]]; then extra+=(--cluster-batch "$CLUSTER_BATCH"); fi
if [[ -n ${SATURATION_TILE:-} ]]; then extra+=(--saturation-tile "$SATURATION_TILE"); fi
steps=(); events=()
for shard in $SHARDS; do
  printf -v suffix '001400_%08d.root' "$shard"
  steps+=("$INPUTS/tracker_tv3_default_l/trk_straw_occ_20261026/fast_g4ref_v2_L/fasttrkCL.grp.$suffix")
  events+=("$INPUTS/tracker_tv3_default_l/trk_secondaries_r2_20261027/tv3_full_paired_v1/fast/strawhits.ram2aq.tv2fast.v0.$suffix")
done
cache=()
if [[ -n ${JAX_COMPILATION_CACHE_DIR:-} ]]; then
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  cache=(--env "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR")
fi
apptainer exec "${nv[@]}" "${cache[@]}" --cleanenv --env "JAX_PLATFORMS=$platform,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONUNBUFFERED=1,PYTHONPATH=$CODE/tracker:$CODE/core,XLA_PYTHON_CLIENT_PREALLOCATE=false" \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" -B "$CODE" -B "$INPUTS:$INPUTS:ro" -B "$(dirname "$OUT")" --pwd "$CODE/tracker" "$SIF" \
  python3 scripts/digi_real_bench.py --steps "${steps[@]}" --events "${events[@]}" \
  --cond "$INPUTS/tracker_digi_reference_v2_20261026/v2/digicond.npz" --backend "$BACKEND" --seed "$SEED" --limit "$LIMIT" --repeats "$REPEATS" --outdir "$OUT" "${extra[@]}"
