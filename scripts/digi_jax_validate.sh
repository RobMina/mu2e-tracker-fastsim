#!/bin/bash
# Paired validation on existing digitiser references; never rerun stock or replace an output directory.
set -euo pipefail
O=${1:?new output directory}
SHARDS=${SHARDS:-1001}
F2_CHUNK=${F2_CHUNK:-50}
D=${M2FS_DATA:-/exp/mu2e/data/users/ram2aq}/trkfs_digi_20261026
T=${M2FS_DATA:-/exp/mu2e/data/users/ram2aq}/trk_fastsim/fast_as_v18L
PY=${PY:-python3}
mkdir "$O"
steps=(); refs=(); numpy=(); floor=()
for s in $SHARDS; do
  steps+=("$T/fasttrkCL.001400_0000${s}.root")
  refs+=("$D/ref/strawhits_ref.v18L_${s}.root")
  numpy+=("$D/v2/fast/fastdigi.fasttrkCL.001400_0000${s}.root")
  floor+=("$D/ref/strawhits_ref.v18L_s9_${s}.root")
done
"$PY" -m trkfastsim digi "${steps[@]}" --cond "$D/v2/digicond.npz" --events "${refs[@]}" --backend jax --seed 1 --outdir "$O/jax"
"$PY" -m trkfastsim digicmp "$O/jax/"*.root --ref "${numpy[@]}" --floor "${floor[@]}" --hit-cond "$D/v2/cmp/hitcond.json" --drift-source "${refs[0]}" --f2 --f2-chunk "$F2_CHUNK" --f2-fixed-shapes --f2-progress --require-same-events --outdir "$O/vs_numpy"
"$PY" -m trkfastsim digicmp "$O/jax/"*.root --ref "${refs[@]}" --floor "${floor[@]}" --f2 --f2-chunk "$F2_CHUNK" --f2-fixed-shapes --f2-progress --require-same-events --outdir "$O/vs_stock"
