#!/bin/bash
# CE tracker steps of both arms for one dts file: fast (umbrella beamfastsim.cetracker: cechain -> fieldsteps) and Geant4 (trkfastsim.validation.cearms).
# Usage (Slurm array over the lines "SEQUENCER TABLE" of LIST): sbatch --array=0-N trk1_ce_steps_uva.sh LIST OUT_ROOT SEED
# Environment: FASTSIMKIT_SCRATCH, CODE (dir with core/, beam/ and tracker code), IMAGE (fastsim SIF), GDML, MAPS ("DS TSd"),
# OFFLINE_ROOT and CE_GEOM (cechain geometry file); writes OUT_ROOT/<SEQUENCER>/.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 04:00:00
set -eo pipefail
module load apptainer
LIST=$1; O=$2; SEED=$3
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the scratch root}")
read -r SEQ TAB < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
TAB=$(readlink -f "$TAB")
W=$O/$SEQ
mkdir -p "$W"
export APPTAINERENV_PYTHONPATH=${CODE:?}/core:$CODE/beam:${TRACKER_CODE:-$CODE/tracker}
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
RUN=(apptainer exec --cleanenv ${NV:+--nv} -B "$B:$B" "${IMAGE:?}" python3)
# shellcheck disable=SC2086
/usr/bin/time -v -o "$W/handoff.time" "${RUN[@]}" -m beamfastsim.cetracker "$TAB" --outdir "$W" --seed "$SEED" \
  --gdml "${GDML:?}" --maps ${MAPS:?} --offline-root "${OFFLINE_ROOT:?}" --geom "${CE_GEOM:?}" ${CETRACKER_ARGS:-} > "$W/handoff.log" 2>&1
/usr/bin/time -v -o "$W/g4arm.time" "${RUN[@]}" -m trkfastsim.validation.cearms "$TAB" --outdir "$W" > "$W/g4arm.log" 2>&1
