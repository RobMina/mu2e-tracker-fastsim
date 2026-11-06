#!/bin/bash
# CE arm through stock art: inject an injector table into its dts events, stock OffSpill digitisation, stock recoMC OffSpill (LoopHelix KKDe), EventNtuple.
# Usage (Slurm array over the dts sequencers in LIST): sbatch -A YOUR_ACCOUNT --array=0-N ce_reco_slurm.sh ARM LIST HANDOFF_ROOT OUT_ROOT TABLE_NAME
#   LIST: lines "SEQUENCER DTS_PATH"; HANDOFF_ROOT/<SEQUENCER>/TABLE_NAME is the injector table; output OUT_ROOT/<ARM>/<SEQUENCER>/.
# Site paths come from the environment: FASTSIMKIT_SCRATCH (scratch root with the runtime, tarball, cvmfsexec).
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 03:00:00
set -eo pipefail
module load apptainer
ARM=$1; LIST=$2; H=$3; O=$4; TABNAME=$5
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the scratch root}")
read -r SEQ DTS < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
DTS=$(readlink -f "$DTS")
TAB=$(readlink -f "$H/$SEQ/$TABNAME")
A=${OFFLINE_TARBALL:?muse tarball with the TrkFastSim injector modules}
R=${OFFLINE_CHAIN_RUNNER:?offline_uva runner with OFFLINE_CHAIN support}
W=$O/$ARM/$SEQ
mkdir -p "$W/fcl"
cd "$W"
[ -s "$TAB" ] && [ -s "$DTS" ]
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
  p1 : [ compressDetStepMCs, trkSteps, FindMCPrimary ]
  e1 : [ out ]
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
services.SeedService.baseSeed : ${DIGI_SEED:-20261105}
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
cd "$W"
"\$OFFLINE_TIME" -v -o inject.time mu2e -c fcl/inject.fcl -s "$DTS" > inject.log 2>&1
"\$OFFLINE_TIME" -v -o digi.time mu2e -c fcl/digi.fcl -s inj.art > digi.log 2>&1
"\$OFFLINE_TIME" -v -o reco.time mu2e -c fcl/reco.fcl -s dig.art > reco.log 2>&1
"\$OFFLINE_TIME" -v -o ntuple.time mu2e -c fcl/ntuple.fcl -s mcs-LH.art -T nts.$SEQ.root.tmp > ntuple.log 2>&1
grep -h "TrigReport Events total" inject.log digi.log reco.log ntuple.log > events.txt || true
mv nts.$SEQ.root.tmp nts.$SEQ.root
rm -f inj.art mcs-LH.art
[ "${KEEP_DIGI:-0}" = 1 ] || rm -f dig.art
EOF
export CVMFSEXEC_DIST=${CVMFSEXEC_DIST:-$B/environments/cvmfsexec-el8}
export OFFLINE_IMAGE=${OFFLINE_IMAGE:?pinned EL9 worker image directory on CVMFS}
export OFFLINE_BINDS="$B:$B" OFFLINE_CHAIN="$W/chain.sh" OFFLINE_LOG_PREFIX=reco
bash "$R" "$A" "$W/fcl/digi.fcl" > runner.log 2>&1
