#!/usr/bin/env bash
# Consume an exact-prefix-gated ordered table with the existing historical stock runtime.
set -eo pipefail
R=${1:?output root}
FCLS=${2:?frozen historical FCL directory}
OUT=${3:?new output directory}
PREFLIGHT=${4:?completed prefix and ten-event preflight directory}
[[ -f "$PREFLIGHT/COMPLETE" && -s "$PREFLIGHT/prefix_gate.json" ]] || { echo 'stock preflight is not complete'; exit 2; }
module load apptainer
mkdir "$OUT"
OUT=$(readlink -f "$OUT")
cp "$PREFLIGHT/offline_uva.sh" "$OUT/offline_uva.sh"
mkdir "$OUT/fcl"
cp "$FCLS/"*.fcl "$OUT/fcl/"
A=$R/evidence/tracker_runtime_stage_v30/Code_tv2.tar.bz2
TABLE=$R/evidence/full_midpoint1000_v26/ordered_steps.root
GEN=/scratch/ram2aq/mu2efastsim/inputs/offline_q1_shard22/gen.art
sha256sum "$A" "$TABLE" "$GEN" "$PREFLIGHT/prefix_gate.json" "$OUT/offline_uva.sh" "$OUT/fcl/"*.fcl > "$OUT/input.sha256"
export CVMFSEXEC_DIST=/scratch/ram2aq/mu2efastsim/offline_uva_s11_20261029/runtime/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_FCL_PATH="$OUT/fcl" OFFLINE_EVENTS=1000 OFFLINE_WARM_REPEAT=0
cd "$OUT"
printf '#include "TrkFastSim/fcl/digi_from_fasttable_alltrig.fcl"\nphysics.producers.compressDetStepMCs.trkTableFile : "%s"\noutputs.Output.fileName : "%s"\nservices.SeedService.baseSeed : 20261023\n' "$TABLE" "$OUT/dig.art" > fcl/digi.fcl
unset OFFLINE_OUTPUT_FILE
export OFFLINE_LOG_PREFIX=digi
bash "$OUT/offline_uva.sh" "$A" "$OUT/fcl/digi.fcl" "$GEN" > digi.log 2>&1
printf '#include "reco_kkts_extrap_nofilter_v04.fcl"\noutputs.KinematicLineOutput.fileName : "%s"\nphysics.analyzers.trkdump : { module_type : StrawHitTableDumper ProtonBunchTimeTag : @local::physics.producers.makeSH.ProtonBunchTimeTag }\nphysics.EndPath : [ @sequence::physics.EndPath, trkdump ]\n' "$OUT/mcs.art" > fcl/reco.fcl
export OFFLINE_LOG_PREFIX=reco OFFLINE_OUTPUT_FILE="$OUT/strawhits.root"
bash "$OUT/offline_uva.sh" "$A" "$OUT/fcl/reco.fcl" "$OUT/dig.art" > reco.log 2>&1
printf '#include "ntuple_kkts_extrap_v04.fcl"\nservices.DbService.purpose : Sim_best\nservices.DbService.version : v1_5\n' > fcl/ntuple.fcl
export OFFLINE_LOG_PREFIX=ntuple OFFLINE_OUTPUT_FILE="$OUT/ntuple.root"
bash "$OUT/offline_uva.sh" "$A" "$OUT/fcl/ntuple.fcl" "$OUT/mcs.art" > ntuple.log 2>&1
T=$R/code/tracker_continuation_v26
E="JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$R/code/stock_chain_v32:$T:$R/code/core:$R/code/crv"
apptainer exec --cleanenv --env "$E" -B "$R" /scratch/ram2aq/mu2efastsim/mu2efastsim.sif python3 -m scripts.ordered_stock_output_audit "$TABLE" --ntuple "$OUT/ntuple.root" --hits "$OUT/strawhits.root" --out "$OUT/coverage.json" > coverage.log 2>&1
sha256sum -c "$OUT/input.sha256"
sha256sum "$OUT/"*.root "$OUT/fcl/"*.fcl > "$OUT/products.sha256"
touch "$OUT/COMPLETE"
