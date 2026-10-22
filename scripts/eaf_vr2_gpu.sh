#!/bin/bash
# F2 JAX line fit and JAX hit reco on an EAF A100 MIG slice (run in the EAF terminal by core/eaf/eafhub.py run):
# reco-vr2/vr1/vr1b --jax and reco-vr3 on the dump tables read in place from /exp.
set -u
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
V=${M2FS_VENV:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf}/bin/python
D=${TRK_FASTRECO_DATA:-/exp/mu2e/data/users/ram2aq/trk_fastreco}
O=$D/gpu_$(date +%Y%m%d_%H%M%S); mkdir -p $O
cd $M2FS_CODE && export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker
nvidia-smi -L; $V -c "import jax; print(jax.devices())"
$V -m trkfastsim reco-vr2 $D/trk2/tracks_ref_mc.001400_0000000{0,1,2}.root --conditions $D/cond_mc_au.json --jax --outdir $O/vr2 > $O/vr2.log 2>&1; echo "vr2 rc=$?"
$V -m trkfastsim reco-vr1 $D/strawhits_ref_mc.001400_0000000{0,1,2}.root --conditions $D/cond_mc_au.json --settings mc --jax --outdir $O/vr1 > $O/vr1.log 2>&1; echo "vr1 rc=$?"
$V -m trkfastsim reco-vr1b $D/trk2/tracks_ref_mc.001400_0000000{0,1,2}.root --conditions $D/cond_mc_au.json --jax --outdir $O/vr1b > $O/vr1b.log 2>&1; echo "vr1b rc=$?"
$V -m trkfastsim reco-vr3 $D/trk2/tracks_ref_mc.001400_0000000{0,1,2}.root --conditions $D/cond_mc_au.json --outdir $O/vr3 > $O/vr3.log 2>&1; echo "vr3 rc=$?"
echo "out $O"
