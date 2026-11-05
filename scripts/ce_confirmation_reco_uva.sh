#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Replay the historical primary-only CE adapter and retain every intermediate and stage ledger.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH -J r2_ce_lh_s32
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s32
D=$Y/out/ce_reference_s26
C=$Y/out/ce_closed_audit_s30
A=$B/trk_ledger_s3_20261028/evidence/tracker_runtime_stage_v30/Code_tv2.tar.bz2
R=$B/trk_field_y4_20261101/runtime/offline_uva_chain.sh
IMAGE=$B/mu2efastsim.sif.prev-20261104
case ${SLURM_ARRAY_TASK_ID:?array index required} in
  0) ARM=g4; H=$Y/out/ce_confirm_entry_s31; TAB=$H/g4prim_field.root ;;
  1) ARM=entry; H=$Y/out/ce_confirm_entry_s31; TAB=$H/fasttrk_field.root ;;
  2) ARM=chain; H=$Y/out/ce_confirm_chain_s31; TAB=$H/fasttrk_field.root ;;
  *) exit 2 ;;
esac
[ -s "$C/COMPLETE" ] && [ -s "$H/COMPLETE" ]
W=$Y/out/ce_confirm_reco_${ARM}_s32
mkdir "$W"
cd "$W"
echo "dcb2bd3bb0581fcf422ec081aee5ec603aa54602996042e751187fc829c09a19  $A" | sha256sum -c -
echo "34311e1e6dc3668779b6f9a84ab988ca4ee293fd00a7faa6e25a0c11be55296a  $R" | sha256sum -c -
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$D" && sha256sum -c transport.sha256)
(cd "$H" && sha256sum -c transport.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
(cd "$S/tracker" && apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 -m pytest -q -p no:cacheprovider tests/test_cehistory.py)
mkdir fcl
GEOM='services.GeometryService.inputFile : "Offline/Mu2eG4/geom/geom_run1_a.txt"'
DB='services.DbService.purpose : Sim_best
services.DbService.version : v1_1'
cat > fcl/inject.fcl <<EOF
#include "Offline/fcl/standardServices.fcl"
process_name : FieldInject
source : { module_type : RootInput }
services : @local::Services.Sim
physics : {
  producers : { compressDetStepMCs : { module_type : StrawGasStepsFromTable trkTableFile : "$TAB" crvTableFile : ""
                genParticles : "compressDetStepMCs" trkKeyOffset : 0 primaryKey : 1 makeTrajectories : true }
                FindMCPrimary : { module_type : FindMCPrimary SimParticles : "compressDetStepMCs" PrimaryProcess : "mu2ePrimary" } }
  filters : { trkSteps : { module_type : DetectorStepFilter StrawGasSteps : [ "compressDetStepMCs" ] CaloShowerSteps : [ ]
                CrvSteps : [ ] MinimumTrkStepEnergy : 0.0 MinimumCaloStepEnergy : 0.0 MinimumCrvStepEnergy : 0.0
                MinimumCrvSteps : 1 MinimumPartMom : 0.0 MaximumPartMom : 1.0e9 MinimumCaloPartMom : 0.0
                MaximumCaloPartMom : 1.0e9 KeepPDG : [ ] MinimumTrkSteps : 1 MinimumSumCaloStepE : 1.0e9 MinimumSumCaloE : 1.0e9 } }
  analyzers : { current : { module_type : EventLister } }
  p1 : [ compressDetStepMCs, trkSteps, FindMCPrimary ]
  e1 : [ current, out ]
  trigger_paths : [ p1 ]
  end_paths : [ e1 ]
}
outputs : { out : { module_type : RootOutput fileName : "inj.art" SelectEvents : [ p1 ]
  outputCommands : [ "keep *_*_*_*", "drop *_compressDetStepMCs_*_*", "drop *_FindMCPrimary_*_*",
                     "keep *_*_*_FieldInject", "keep mu2e::GenParticles_*_*_*" ] } }
services.SeedService.baseSeed : 1
$GEOM
services.GeometryService.bFieldFile : "Offline/Mu2eG4/geom/bfgeom_no_tsu_ps_v01.txt"
EOF
cat > fcl/digi.fcl <<EOF
#include "Production/JobConfig/digitize/OffSpill.fcl"
physics.filters.TriggerableCHPrescale.prescaleFactor : 1
outputs.Output.SelectEvents : [ "DigitizePath" ]
outputs.Output.fileName : "dig.art"
$GEOM
$DB
services.SeedService.baseSeed : 20267302
EOF
cat > fcl/reco.fcl <<EOF
#include "Production/JobConfig/recoMC/OffSpill.fcl"
outputs.LoopHelixOutput.fileName : "mcs-LH.art"
physics.EndPath : [ @sequence::Reconstruction.OffSpillEndSequence, LoopHelixOutput ]
$GEOM
$DB
EOF
cat > fcl/ntuple.fcl <<EOF
#include "EventNtuple/fcl/from_mcs-mockdata.fcl"
physics.EventNtuplePath : [ @sequence::EventNtuple.PathOff ]
physics.analyzers.EventNtuple.trk.fits : [ { @table::Off trkDtDtTag : "" } ]
physics.analyzers.EventNtuple.FitType : LoopHelix
physics.analyzers.EventNtuple.trk.fillTrkDtDt : false
physics.producers.MergeKKOff.KalSeedCollections : [ "KKDe" ]
$GEOM
$DB
EOF
cat > chain.sh <<EOF
set -eo pipefail
export TMPDIR=\$(dirname "\$OFFLINE_CACHE_PATH")/tmp
export SQLITE_TMPDIR=\$TMPDIR
mkdir -p "\$TMPDIR"
for c in inject digi reco ntuple; do fhicl-dump -c fcl/\$c.fcl > fcl/\$c.expanded.fcl; done
"\$OFFLINE_TIME" -v -o inject.time mu2e -c fcl/inject.expanded.fcl -s "$D/ce_reference.art" > inject.log 2>&1
date -u +%FT%TZ > INJECT_CLOSED
"\$OFFLINE_TIME" -v -o digi.time mu2e -c fcl/digi.expanded.fcl -s inj.art > digi.log 2>&1
date -u +%FT%TZ > DIGI_CLOSED
"\$OFFLINE_TIME" -v -o reco.time mu2e -c fcl/reco.expanded.fcl -s dig.art > reco.log 2>&1
date -u +%FT%TZ > RECO_CLOSED
"\$OFFLINE_TIME" -v -o ntuple.time mu2e -c fcl/ntuple.expanded.fcl -s mcs-LH.art -T nts.001430_00006200.root > ntuple.log 2>&1
date -u +%FT%TZ > NTUPLE_CLOSED
sha256sum inj.art dig.art mcs-LH.art nts.001430_00006200.root fcl/*.expanded.fcl > closed.sha256
EOF
export CVMFSEXEC_DIST=$B/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$B:$B" OFFLINE_CHAIN="$W/chain.sh" OFFLINE_LOG_PREFIX=reco
bash "$R" "$A" "$W/fcl/digi.fcl" > runner.log 2>&1
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 -m trkfastsim.validation.cehistory \
  --births "$H/states.root" --table "$TAB" --inject-art "$W/inj.art" --digi-art "$W/dig.art" \
  --reco-art "$W/mcs-LH.art" --ntuple "$W/nts.001430_00006200.root" --fileindex-byteswap --outdir "$W/history"
date -u +%FT%TZ > COMPLETE
