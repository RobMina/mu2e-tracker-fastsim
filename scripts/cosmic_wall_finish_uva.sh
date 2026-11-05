#!/bin/bash
# Saved-only filter audit, stage reconciliation and comparison; never rerun completed transport/art stages.
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s56
H=$Y/out/cosmic_wall_pilot_s52
R=$Y/out/cosmic_wall_reco_s53
W=$Y/out/cosmic_wall_finish_s56
IMAGE=$B/mu2efastsim.sif.prev-20261104
[ -s "$H/COMPLETE" ] && [ -s "$R/NTUPLE_CLOSED" ]
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$R" && sha256sum -c raw.sha256 && sha256sum -c inputs.sha256)
(cd "$H/transport" && sha256sum -c ../closed.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
sha256sum "$R"/{inj.art,dig.art,mcs-CH.art,nts.001430_00000000.root,inject.log,digi.log,reco.log,ntuple.log,inject.time,digi.time,reco.time,ntuple.time,raw.sha256,inputs.sha256} "$IMAGE" > inputs.sha256
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_fieldwallhistory.py tests/test_fieldwallcompare.py)
runpy -m trkfastsim.validation.fieldwallhistory --diagnose-filter \
  --source-table "$Y/out/tables/cosmiccryall_ac/sgsteps.001430_00000000.root" \
  --candidate-table "$H/transport/fasttrk_field.root" --stage-dir "$R" --outdir "$W/filter"
runpy -m trkfastsim.validation.fieldwallhistory \
  --source-table "$Y/out/tables/cosmiccryall_ac/sgsteps.001430_00000000.root" \
  --candidate-table "$H/transport/fasttrk_field.root" --stage-dir "$R" --legacy-dir "$Y/out" --outdir "$W/history"
/usr/bin/time -v -o compare.time apptainer exec --cleanenv -B "$B:$B" "$IMAGE" \
  python3 -m trkfastsim.validation.fieldwallcompare \
  --candidate-table "$H/transport/fasttrk_field.root" --legacy-dir "$Y/out/unbiased_mu_s7" \
  --candidate-ntuple "$R/nts.001430_00000000.root" \
  --legacy-ntuple "$Y/out/art3v2_fast_unbiased/nts.001430_00000000.root" \
  --g4-ntuple "$Y/out/art3v2_g4fp_unbiased/nts.001430_00000000.root" \
  --baseline-events "$Y/out/unbiased_mu_reco_s7/events.csv.gz" --outdir "$W/compare"
sha256sum -c inputs.sha256
(cd "$R" && sha256sum -c raw.sha256 && sha256sum -c inputs.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
sha256sum filter/* history/* compare/* > closed.sha256
date -u +%FT%TZ > COMPLETE
