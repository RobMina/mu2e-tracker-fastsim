#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Compare closed paired CE outputs after their explicit stage histories pass.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:20:00
#SBATCH -J r2_ce_compare_s32
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s33
W=$Y/out/ce_confirm_compare_s33
IMAGE=$B/mu2efastsim.sif.prev-20261104
for arm in g4 entry chain; do
  [ -s "$Y/out/ce_confirm_reco_${arm}_s32/COMPLETE" ]
  (cd "$Y/out/ce_confirm_reco_${arm}_s32" && sha256sum -c closed.sha256)
done
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
for mode in entry chain; do
  [ -s "$Y/out/ce_confirm_${mode}_s31/COMPLETE" ]
  (cd "$Y/out/ce_confirm_${mode}_s31" && sha256sum -c transport.sha256)
done
for arm in g4 entry chain; do
  runpy -m trkfastsim.validation.cehistory --verify-report "$Y/out/ce_confirm_reco_${arm}_s32/history/cehistory.json"
done
for mode in entry chain; do
  CLAIM=diagnostic
  [ "$mode" != chain ] || CLAIM=complete-birth
  runpy -m trkfastsim.validation.cereco \
    g4="$Y/out/ce_confirm_reco_g4_s32/nts.001430_00006200.root" \
    fast="$Y/out/ce_confirm_reco_${mode}_s32/nts.001430_00006200.root" \
    --states "$Y/out/ce_confirm_${mode}_s31/states.root" --claim "$CLAIM" --outdir "$W/$mode"
done
runpy -m trkfastsim.validation.ceaudit --entry-events "$W/entry/events.csv.gz" --chain-events "$W/chain/events.csv.gz" \
  --entry-dir "$Y/out/ce_confirm_entry_s31" --chain-dir "$Y/out/ce_confirm_chain_s31" --outdir "$W/diagnostic"
sha256sum entry/* chain/* diagnostic/* > compare.sha256
date -u +%FT%TZ > COMPLETE
