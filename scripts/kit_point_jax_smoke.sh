#!/bin/bash
# One cached primary shard through the opt-in JAX digitiser; reuse the saved NumPy point as reference.
set -euo pipefail
OUT=${1:?new output directory}
DATA=${M2FS_DATA:-/exp/mu2e/data/users/ram2aq}
CORE=${M2FS_CORE:?matching pinned core snapshot}
PY=${PY:?Python with JAX and uproot}
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$HERE:$CORE" PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$PY" -u -m trkfastsim.kitpoint --backend jax \
  --gen "$DATA/crv_fastsim/g4ref_v1_tables/gen.001400_00000011.root" \
  --gdml "$DATA/crv_fastsim/geometry/mu2e_extracted_v04.gdml" \
  --digi-cond "$DATA/trkfs_digi_20261026/v2/digicond.npz" \
  --reco-cond "$DATA/trk_fastreco/cond_mc_au.json" \
  --drift-source "$DATA/trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root" \
  --seed 1 --outdir "$OUT" --workdir /tmp --retain-ntuple
