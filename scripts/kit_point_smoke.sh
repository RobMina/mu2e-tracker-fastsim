#!/bin/bash
# One existing cached shard through the fixed tracker point on one CPU; no builds or stock reruns.
set -e
OUT=${1:?new output directory}
DATA=${M2FS_DATA:?set the local data root}
HERE=$(cd "$(dirname "$0")/.." && pwd)
CORE=${M2FS_CORE:?set the fastsim core checkout}
PY=${PY_JAX:?set a Python interpreter with JAX, uproot, pandas and scipy}
export PYTHONPATH="$CORE:$HERE" PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec taskset -c "${CORE_ID:-1}" "$PY" -m trkfastsim.kitpoint \
  --gen "$DATA/crv_fastsim/g4ref_v1_tables/gen.001400_00000011.root" \
  --gdml "$DATA/crv_fastsim/geometry/mu2e_extracted_v04.gdml" \
  --digi-cond "$DATA/trkfs_digi_20261026/v2/digicond.npz" \
  --reco-cond "$DATA/trk_fastreco/cond_mc_au.json" \
  --drift-source "$DATA/trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root" \
  --seed 1 --outdir "$OUT" --workdir /tmp --retain-ntuple
