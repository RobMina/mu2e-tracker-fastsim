#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Confirm saved cosmic identity and singleton diagnostics on the nine fixed parents without rerunning physics.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH --array=0-8%2
#SBATCH -J r2_idparents_s40
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s40
STEM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$Y/jobs/r2_cosmic_parents_s15.txt")
[[ "$STEM" =~ ^001430_[0-9]{8}$ ]] && [ "$STEM" != 001430_00000000 ]
W=$Y/out/cosmic_identity_parents_s40_$STEM
IMAGE=$B/mu2efastsim.sif.prev-20261104
[ -s "$Y/out/cosmic_identity_pilot_s40/COMPLETE" ]
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
apptainer exec --cleanenv -B "$B:$B" "$IMAGE" \
  python3 -m trkfastsim.validation.fieldidentity \
  --g4-ntuple "$Y/out/art3v2_g4fp_parents_s15_$STEM/nts.$STEM.root" \
  --fast-ntuple "$Y/out/art3v2_fast_parents_s15_$STEM/nts.$STEM.root" \
  --field-dir "$Y/out/cosmic_field_s15_$STEM" \
  --baseline-events "$Y/out/cosmic_reco_s15_$STEM/events.csv.gz" --outdir "$W/audit"
sha256sum audit/* > closed.sha256
date -u +%FT%TZ > COMPLETE
