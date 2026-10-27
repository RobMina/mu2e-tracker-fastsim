#!/bin/bash
# Fast digitisation validation on one GPVM, serially: conditions -> fast digis on the stock reference events -> digicmp.
# usage: digi_validate.sh <data dir> <refdir> <table dir> <seq[,seq...]> <out dir> [python]
#   data dir: holds patha_digi_dump.fcl (fhicl-dump of TrkFastSim/fcl/digi_from_fasttable_alltrig.fcl) and the
#             dbTool CSVs db_TrkPreampStraw.csv, db_TrkAlignStrawSim.csv
#   refdir:   art/run_digi_ref.sh outputs strawhits_ref.v18L_<seq>.root and (floor) strawhits_ref.v18L_s9_<seq>.root
#   out dir:  digicond.npz, fast/, cmp/
#   python:   default python3 (pyenv ana); a JAX python (venv_eaf, JAX_PLATFORMS=cpu) adds the F2 yield (F2=1)
# The digitisation step runs on one core (taskset, one BLAS/OpenMP thread), so its printed ms/event is per core.
set -e
D=${1:?data dir}; R=${2:?refdir}; T=${3:?table dir}; SEQS=${4:?seq list, e.g. 1001,1002}; O=${5:?out dir}; PY=${6:-python3}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${M2FS_CORE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/core}:$HERE" PYTHONDONTWRITEBYTECODE=1
GDML=${GDML:-/exp/mu2e/data/users/ram2aq/crv_fastsim/geometry/mu2e_extracted_v04.gdml}
GC=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00/Offline/GlobalConstantsService/data/globalConstants_01.txt
mkdir -p "$O"
[ -s "$O/digicond.npz" ] || $PY -m trkfastsim digi-cond "$D/patha_digi_dump.fcl" --preamp "$D/db_TrkPreampStraw.csv" \
  --align "$D/db_TrkAlignStrawSim.csv" --gdml "$GDML" --global-constants "$GC" --out "$O/digicond.npz" --outdir "$O" \
  --check-ref "$R/strawhits_ref.v18L_${SEQS%%,*}.root"
TABS=(); REFS=(); FLOORS=(); FAST=()
for s in ${SEQS//,/ }; do
  TABS+=("$T/fasttrkCL.001400_0000$s.root"); REFS+=("$R/strawhits_ref.v18L_$s.root"); FLOORS+=("$R/strawhits_ref.v18L_s9_$s.root")
  FAST+=("$O/fast/fastdigi.fasttrkCL.001400_0000$s.root")
done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c ${CORE:-1} \
  $PY -m trkfastsim digi "${TABS[@]}" --cond "$O/digicond.npz" --events "${REFS[@]}" --seed "${SEED:-1}" --outdir "$O/fast"
$PY -m trkfastsim digicmp "${FAST[@]}" --ref "${REFS[@]}" --floor "${FLOORS[@]}" --outdir "$O/cmp" ${F2:+--f2}
