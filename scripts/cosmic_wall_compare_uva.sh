#!/bin/bash
# Saved-only development comparison; reuse every retained legacy/G4 decision and the fixed denominator.
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s53
H=$Y/out/cosmic_wall_pilot_s52
R=$Y/out/cosmic_wall_reco_s53
W=$Y/out/cosmic_wall_compare_s53
IMAGE=$B/mu2efastsim.sif.prev-20261104
[ -s "$R/COMPLETE" ] && [ -s "$H/COMPLETE" ]
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$H/transport" && sha256sum -c ../closed.sha256)
(cd "$R" && sha256sum -c closed.sha256 && sha256sum -c raw.sha256 && sha256sum -c inputs.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_fieldwallcompare.py tests/test_fieldwallhistory.py)
/usr/bin/time -v -o compare.time apptainer exec --cleanenv -B "$B:$B" "$IMAGE" \
  python3 -m trkfastsim.validation.fieldwallcompare \
  --candidate-table "$H/transport/fasttrk_field.root" --legacy-dir "$Y/out/unbiased_mu_s7" \
  --candidate-ntuple "$R/nts.001430_00000000.root" \
  --legacy-ntuple "$Y/out/art3v2_fast_unbiased/nts.001430_00000000.root" \
  --g4-ntuple "$Y/out/art3v2_g4fp_unbiased/nts.001430_00000000.root" \
  --baseline-events "$Y/out/unbiased_mu_reco_s7/events.csv.gz" --outdir "$W/compare"
(cd compare && sha256sum fieldwallcompare.json *.csv.gz > ../closed.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
date -u +%FT%TZ > COMPLETE
