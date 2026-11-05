#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Run frozen first-passage CE transport only after exact closed-reference admission.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:30:00
#SBATCH -J r2_ce_confirm_s31
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s31
E=$B/ele_y3_20261101/code_v19/umbrella/beam
D=$Y/out/ce_reference_s26
C=$Y/out/ce_closed_audit_s30
IMAGE=$B/mu2efastsim.sif.prev-20261104
case ${SLURM_ARRAY_TASK_ID:?array index required} in
  0) MODE=entry ;;
  1) MODE=chain ;;
  *) exit 2 ;;
esac
W=$Y/out/ce_confirm_${MODE}_s31
[ -s "$C/COMPLETE" ]
mkdir "$W"
cd "$W"
sha256sum -c "$Y/inputs/ce_confirm_s31/frozen.sha256"
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker:$E
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_JAX_PLATFORMS=cpu
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_ceadmit.py)
runpy -m trkfastsim.validation.ceadmit --census "$C/census/cecensus.json" --source "$C/source.json" --primary "$C/primary/ceinput.json" --out "$W/admission.json"
runpy -m trkfastsim.validation.fieldce "$D/sgsteps.001430_00006200.root" \
  --mode "$MODE" --seed 202621100 --outdir "$W" \
  --reference-selection unfiltered --expected-starts 2000 \
  --gdml "$B/hardlib_proto/data/geometry/mu2e_run1a_v01.gdml" \
  --maps "$B/ele_y3_20261101/maps/DSMap.header" "$B/ele_y3_20261101/maps/TSdMap.header" \
  --offline-root "$B/fastsim_docs_w1_20261031/dryrun/out/beam_pub/offline" \
  --geom Offline/Mu2eG4/geom/geom_2021_PhaseI_v03.txt
sha256sum admission.json handoff.json states.root fasttrk_field.root g4prim_field.root > transport.sha256
date -u +%FT%TZ > COMPLETE
