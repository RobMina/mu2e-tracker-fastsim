#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Observe material on frozen stochastic paths without changing transport or rerunning stock reconstruction.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:45:00
#SBATCH -J r2_material_s42
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s42
W=$Y/out/cosmic_material_pilot_s42
IMAGE=$B/mu2efastsim.sif.prev-20261104
[ -s "$Y/out/cosmic_path_pilot_s36/COMPLETE" ]
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
echo "2df2ca48753e0a36b4c529b658e28ac6a3f10c9a64d0a52ccbadf306e9212fa0  $B/hardlib_proto/data/geometry/mu2e_run1a_v01.gdml" | sha256sum -c -
(cd "$Y/out/cosmic_path_pilot_s36" && sha256sum -c closed.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_fieldmaterial.py tests/test_fieldpathaudit.py tests/test_fieldcrossing.py tests/test_fieldsteps.py)
/usr/bin/time -v -o audit.time apptainer exec --cleanenv -B "$B:$B" "$IMAGE" \
  python3 -m trkfastsim.validation.fieldmaterial --field-dir "$Y/out/unbiased_mu_s7" \
  --input-table "$Y/out/tables/cosmiccryall_ac/sgsteps.001430_00000000.root" \
  --gdml "$B/hardlib_proto/data/geometry/mu2e_run1a_v01.gdml" --map "$Y/maps/DSMap.header" \
  --outdir "$W/audit" --entry first --oracle-chords 256
sha256sum audit/fieldmaterial.json audit/replay/*.root audit/replay/fieldcompare.json > closed.sha256
date -u +%FT%TZ > COMPLETE
