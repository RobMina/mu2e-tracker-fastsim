#!/bin/bash
# One enabled development transport on the original pilot entries; retained references are read-only.
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s52
W=$Y/out/cosmic_wall_pilot_s52
IMAGE=$B/mu2efastsim.sif.prev-20261104
[ -s "$Y/out/cosmic_source_pilot_s45/COMPLETE" ]
[ -s "$S/tracker/CENSUS.sha256" ]
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
echo "2df2ca48753e0a36b4c529b658e28ac6a3f10c9a64d0a52ccbadf306e9212fa0  $B/hardlib_proto/data/geometry/mu2e_run1a_v01.gdml" | sha256sum -c -
echo "4061bda2c19a7bcc8f56d3d3ad34b49988febd451bd91808ec61b1befa33ae6f  $Y/maps/DSMap.header" | sha256sum -c -
echo "4d81d17e4e85cfe70fd48d6834f7c48b8dcf0febf84ea77014716960ba010e2e  $Y/maps/DSMap.bin" | sha256sum -c -
(cd "$Y/out/cosmic_source_pilot_s45/audit" && sha256sum -c ../closed.sha256)
(cd "$Y/out/cosmic_path_pilot_s36" && sha256sum -c closed.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_fieldwallpilot.py tests/test_fieldwall_defaults.py tests/test_fieldwall_transport.py tests/test_fieldwall_geometry.py)
/usr/bin/time -v -o transport.time apptainer exec --cleanenv -B "$B:$B" "$IMAGE" \
  python3 -m trkfastsim.validation.fieldwallpilot \
  --input-table "$Y/out/tables/cosmiccryall_ac/sgsteps.001430_00000000.root" \
  --legacy-dir "$Y/out/unbiased_mu_s7" --source-report "$Y/out/cosmic_source_pilot_s45/audit/fieldsourceaudit.json" \
  --path-report "$Y/out/cosmic_path_pilot_s36/audit/fieldpathaudit.json" \
  --gdml "$B/hardlib_proto/data/geometry/mu2e_run1a_v01.gdml" --map "$Y/maps/DSMap.header" --outdir "$W/transport"
(cd transport && sha256sum fieldwallpilot.json fasttrk_field.root states.root *.csv.gz > ../closed.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
date -u +%FT%TZ > COMPLETE
