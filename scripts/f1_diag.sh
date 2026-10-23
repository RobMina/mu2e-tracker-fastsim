#!/bin/bash
# F1 vs stock diagnostic variants on the first NTRK FitOK tracks of the SaveAllFits dump (CPU or GPU).
set -u
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
V=${M2FS_VENV:-/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf}/bin/python
D=${TRK_FASTRECO_DATA:-/exp/mu2e/data/users/ram2aq/trk_fastreco}
export F1_JAX_CACHE=${F1_JAX_CACHE:-$D/jax_cache}
NTRK=${NTRK:-3000}; CHUNK=${CHUNK:-1024}
O=${OUT:-$D/f1diag_$(date +%Y%m%d_%H%M%S)}; mkdir -p $O
cd $M2FS_CODE && export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker
run() { name=$1; shift; env "$@" $V -m trkfastsim reco-vr2f1 $D/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root \
  --conditions $D/cond_mc_au.json --max-tracks $NTRK --chunk $CHUNK --outdir $O/$name > $O/$name.log 2>&1; echo "$name rc=$?"; }
for v in ${VARIANTS:-base xstock noscat}; do
  case $v in
    base) run base F1_X=1 ;;
    xstock) run xstock F1_XACT=stock ;;
    noscat) run noscat F1_SCATSCALE=0 ;;
    scat2) run scat2 F1_SCATSCALE=2 ;;
    scat05) run scat05 F1_SCATSCALE=0.5 ;;
    pstock) run pstock F1_PATHS=stock F1_XACT=stock ;;
    scat07) run scat07 F1_SCATSCALE=0.7 ;;
    saxis) run saxis F1_STRAWAXIS=$D/align/TrkAlignStraw.csv ;;
    saxis_pstock) run saxis_pstock F1_STRAWAXIS=$D/align/TrkAlignStraw.csv F1_PATHS=stock F1_XACT=stock ;;
    saxis_xstock) run saxis_xstock F1_STRAWAXIS=$D/align/TrkAlignStraw.csv F1_XACT=stock ;;
    saxis_seedx) run saxis_seedx F1_STRAWAXIS=$D/align/TrkAlignStraw.csv F1_NOHIT_IN_SEED=1 ;;
    tight) run tight F1_CONVSCALE=0.001 F1_MAXNITER=40 ;;
    scat14) run scat14 F1_SCATSCALE=1.4 ;;
  esac
done
echo "out $O"
