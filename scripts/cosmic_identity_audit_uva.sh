#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Join saved selected-track contributor IDs to injector primaries without changing the frozen core finding.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -J r2_identity_s37
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s37
W=$Y/out/cosmic_identity_pilot_s37
IMAGE=$B/mu2efastsim.sif.prev-20261104
[ -s "$Y/out/cosmic_mc_inventory_s36/COMPLETE" ]
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$Y/out/cosmic_mc_inventory_s36" && sha256sum -c closed.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_fieldidentity.py)
/usr/bin/time -v -o audit.time apptainer exec --cleanenv -B "$B:$B" "$IMAGE" \
  python3 -m trkfastsim.validation.fieldidentity \
  --g4-ntuple "$Y/out/art3v2_g4fp_unbiased/nts.001430_00000000.root" \
  --fast-ntuple "$Y/out/art3v2_fast_unbiased/nts.001430_00000000.root" \
  --field-dir "$Y/out/unbiased_mu_s7" --baseline-events "$Y/out/unbiased_mu_reco_s7/events.csv.gz" --outdir "$W/audit"
sha256sum audit/* > closed.sha256
date -u +%FT%TZ > COMPLETE
