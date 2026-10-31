#!/usr/bin/env bash
# Consume an audited guarded table with the existing historical stock runtime.
set -eo pipefail
R=${1:?output root}
T=${2:?immutable tracker snapshot}
PRODUCER=${3:?completed guarded producer directory}
OUT=${4:?new stock consumer directory}
PREFLIGHT=$R/evidence/ordered_stock_preflight_v31
FCLS=$R/evidence/ordered_stock_chain_v32/fcl
: "${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH to the project work area on the Slurm cluster (holds inputs/ and mu2efastsim.sif)}"
SIF=${SIF:-$FASTSIMKIT_SCRATCH/mu2efastsim.sif}
module load apptainer
[[ -s "$PRODUCER/complete.json" && -s "$PRODUCER/ordered_steps.root" && -f "$PREFLIGHT/COMPLETE" ]] || exit 2
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd -L)
E="JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$T:$R/code/core:$R/code/crv"
run() { apptainer exec --cleanenv --env "$E" -B "$R" "$SIF" python3 "$@"; }
cd "$T"
run -m scripts.ordered_guarded_audit "$PRODUCER" --out "$OUT/producer_audit.json"
TABLE=$PRODUCER/ordered_steps.root
GEN=$FASTSIMKIT_SCRATCH/inputs/offline_q1_shard22/gen.art
run -m scripts.ordered_stock_preflight "$TABLE" --source-dump "$PREFLIGHT/source.root" --out "$OUT/prefix_gate.json" --expected-events 1000 --source-log "$PREFLIGHT/source.log" --source-art "$GEN" --trusted-source-sha256 cfdc329b7aa346c86d1994fa7cf07b1b2f4692bd44fc9b7db79a27ae9155924f
cp "$PREFLIGHT/offline_uva.sh" "$OUT/offline_uva.sh"
mkdir "$OUT/fcl"
for name in ntuple_kkts_extrap_v04.fcl reco_evd_v03.fcl reco_kkts_extrap_nofilter_v04.fcl reco_smoke_kl_ts.fcl; do cp "$FCLS/$name" "$OUT/fcl/$name"; done
A=$R/evidence/tracker_runtime_stage_v30/Code_tv2.tar.bz2
sha256sum "$A" "$TABLE" "$GEN" "$OUT/prefix_gate.json" "$OUT/producer_audit.json" "$OUT/offline_uva.sh" "$OUT/fcl/"*.fcl > "$OUT/input.sha256"
export CVMFSEXEC_DIST=${CVMFSEXEC_DIST:?set CVMFSEXEC_DIST to the cvmfsexec distribution offline_uva.sh runs under}
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_FCL_PATH="$OUT/fcl" OFFLINE_EVENTS=1000 OFFLINE_WARM_REPEAT=0
cd "$OUT"
printf '#include "TrkFastSim/fcl/digi_from_fasttable_alltrig.fcl"\nphysics.producers.compressDetStepMCs.trkTableFile : "%s"\noutputs.Output.fileName : "%s"\nservices.SeedService.baseSeed : 20261023\n' "$TABLE" "$OUT/dig.art" > fcl/digi.fcl
unset OFFLINE_OUTPUT_FILE
export OFFLINE_LOG_PREFIX=digi
sha256sum "$OUT/fcl/digi.fcl" > "$OUT/generated_fcl.sha256"
bash "$OUT/offline_uva.sh" "$A" "$OUT/fcl/digi.fcl" "$GEN" > digi.log 2>&1
printf '#include "reco_kkts_extrap_nofilter_v04.fcl"\noutputs.KinematicLineOutput.fileName : "%s"\nphysics.analyzers.trkdump : { module_type : StrawHitTableDumper ProtonBunchTimeTag : @local::physics.producers.makeSH.ProtonBunchTimeTag }\nphysics.EndPath : [ @sequence::physics.EndPath, trkdump ]\n' "$OUT/mcs.art" > fcl/reco.fcl
sha256sum "$OUT/fcl/reco.fcl" >> "$OUT/generated_fcl.sha256"
export OFFLINE_LOG_PREFIX=reco OFFLINE_OUTPUT_FILE="$OUT/strawhits.root"
bash "$OUT/offline_uva.sh" "$A" "$OUT/fcl/reco.fcl" "$OUT/dig.art" > reco.log 2>&1
printf '#include "ntuple_kkts_extrap_v04.fcl"\nservices.DbService.purpose : Sim_best\nservices.DbService.version : v1_5\n' > fcl/ntuple.fcl
sha256sum "$OUT/fcl/ntuple.fcl" >> "$OUT/generated_fcl.sha256"
export OFFLINE_LOG_PREFIX=ntuple OFFLINE_OUTPUT_FILE="$OUT/ntuple.root"
bash "$OUT/offline_uva.sh" "$A" "$OUT/fcl/ntuple.fcl" "$OUT/mcs.art" > ntuple.log 2>&1
cd "$T"
run -m scripts.ordered_stock_output_audit "$TABLE" --ntuple "$OUT/ntuple.root" --hits "$OUT/strawhits.root" --out "$OUT/coverage.json" > "$OUT/coverage.log" 2>&1
sha256sum -c "$OUT/input.sha256"
sha256sum -c "$OUT/generated_fcl.sha256"
sha256sum "$OUT/"*.root "$OUT/fcl/"*.fcl > "$OUT/products.sha256"
printf '%s\n' 'Bounded guarded all-lane producer consumer test; not inclusive G4/rate or secondary closure.' > "$OUT/scope.txt"
touch "$OUT/COMPLETE"
