#!/bin/bash
# Reuse existing digitiser outputs for a strict-population fixed-shape F2 comparison.
set -euo pipefail
O=${1:?new output directory}
D=${M2FS_DATA:-/exp/mu2e/data/users/ram2aq}
R=$D/trkfs_digi_20261026
P=$D/trk_digijax_r2_20261027/pilot1001_v1/jax
PY=${PY:-python3}
mkdir "$O"
"$PY" -u -m trkfastsim digicmp "$P/fastdigi.fasttrkCL.001400_00001001.root" \
  --ref "$R/v2/fast/fastdigi.fasttrkCL.001400_00001001.root" \
  --floor "$R/ref/strawhits_ref.v18L_s9_1001.root" --hit-cond "$R/v2/cmp/hitcond.json" \
  --drift-source "$R/ref/strawhits_ref.v18L_1001.root" \
  --f2 --f2-chunk 50 --f2-fixed-shapes --f2-progress --require-same-events --outdir "$O"
