#!/bin/bash
# F1 Kalman fit (kalman_jax) on an EAF A100 MIG slice (run in the EAF terminal by core/eaf/eafhub.py run):
# reco-vr2f1 on the SaveAllFits dump (stock seed, states and crossings), tables read in place from /exp.
# FILTERS: filter forms to compare (cov = covariance + RTS over crossings, info = two-filter information form).
set -u
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
V=${M2FS_VENV:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf}/bin/python
D=${TRK_FASTRECO_DATA:-/exp/mu2e/data/users/ram2aq/trk_fastreco}
export F1_JAX_CACHE=${F1_JAX_CACHE:-$D/jax_cache}
NTRK=${NTRK:-8192}
CHUNK=${CHUNK:-4096}
O=$D/gpu_f1_$(date +%Y%m%d_%H%M%S); mkdir -p $O
cd $M2FS_CODE && export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker
nvidia-smi -L; $V -c "import jax; print(jax.devices())"
for FL in ${FILTERS:-cov}; do
  F1_FILTER=$FL $V -m trkfastsim reco-vr2f1 $D/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root --conditions $D/cond_mc_au.json \
    --max-tracks $NTRK --chunk $CHUNK --outdir $O/vr2f1_$FL > $O/vr2f1_$FL.log 2>&1; echo "vr2f1 $FL rc=$?"
done
echo "out $O"
