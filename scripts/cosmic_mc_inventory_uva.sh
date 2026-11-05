#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Inventory actual saved cosmic MC branches before claiming an exact reconstructed-primary link.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:10:00
#SBATCH -J r2_mcinv_s36
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s36
W=$Y/out/cosmic_mc_inventory_s36
IMAGE=$B/mu2efastsim.sif.prev-20261104
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_fieldmcinventory.py)
runpy -m trkfastsim.validation.fieldmcinventory \
  g4="$Y/out/art3v2_g4fp_unbiased/nts.001430_00000000.root" \
  fast="$Y/out/art3v2_fast_unbiased/nts.001430_00000000.root" --outdir "$W/inventory"
sha256sum inventory/* > closed.sha256
date -u +%FT%TZ > COMPLETE
