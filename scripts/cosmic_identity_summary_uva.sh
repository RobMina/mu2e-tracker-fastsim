#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Keep pilot separate while aggregating the nine fixed saved identity/singleton controls.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:15:00
#SBATCH -J r2_idsummary_s41
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s41
W=$Y/out/cosmic_identity_summary_s41
IMAGE=$B/mu2efastsim.sif.prev-20261104
ARGS=("pilot=$Y/out/cosmic_identity_pilot_s40/audit")
(cd "$Y/out/cosmic_identity_pilot_s40" && sha256sum -c closed.sha256)
while IFS= read -r stem; do
  D=$Y/out/cosmic_identity_parents_s40_$stem
  [ -s "$D/COMPLETE" ]
  (cd "$D" && sha256sum -c closed.sha256)
  ARGS+=("$stem=$D/audit")
done < "$Y/jobs/r2_cosmic_parents_s15.txt"
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_fieldidentityparents.py)
runpy -m trkfastsim.validation.fieldidentityparents "${ARGS[@]}" --pilot pilot --outdir "$W/summary"
sha256sum summary/* > closed.sha256
date -u +%FT%TZ > COMPLETE
