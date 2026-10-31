#!/bin/bash
# Fast steps (stage C defaults + secondary library) on a G4 reference's gen tables, then the paired secondary census.
# usage (GPVM): tracker/scripts/straw_occ_steps.sh OUTDIR [REF=g4ref_v2] [LIB] [TAG] [GROUP=10]; code from TRK (tracker) and
# CORE dirs. The REF gen tables are first merged GROUP subruns per file (gen.grp.<first subrun>.root): one steps output per
# group, which is also the table of one grouped path A grid job (grid_tv2.sh), and one library-sampler set-up per group.
# STEPS_FLAGS: extra steps options (e.g. --offaxis-edge-anchor); SKIP_STEPS=1: census only, on existing fast tables;
# CENSUS_TAG: census output tag (default TAG).
O=$1; REF=${2:-g4ref_v2}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana || exit 1
set -eu
python3 -c 'import numpy, awkward, uproot'
source "${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout that holds site.env}/site.env"
LIB=${3:-$M2FS_DATA/trk_fastsim/lib/trklib_mdc2025as_20f_v2.root}
TAG=${4:-L}
GROUP=${5:-10}
TRK=${TRK:-$(cd "$(dirname "$0")/.." && pwd)}
CORE=${CORE:-$TRK/../straw-core}
export PYTHONPATH=$CORE:$TRK:${CRVEXTRAP:?set CRVEXTRAP to the mu2e_track_crv_mcresponse checkout}
export PYTHONHASHSEED=0
F=$O/fast_${REF}_$TAG
GD=$O/gen_${REF}_grp$GROUP
mkdir -p $F $GD
mapfile -t GEN < <(ls $CRVFS_DATA/${REF}_tables/gen.001400_*.root)
for ((i = 0; i < ${#GEN[@]}; i += GROUP)); do
  S0=$(basename "${GEN[$i]}" .root | sed 's/^gen\.//')
  [ -s $GD/gen.grp.$S0.root ] || python3 $TRK/scripts/merge_tables.py $GD/gen.grp.$S0.root "${GEN[@]:$i:$GROUP}" > /dev/null || exit 1
done
if [ "${SKIP_STEPS:-0}" != 1 ]; then
  /usr/bin/time -f "%e s %M kB" python3 -m trkfastsim steps "$GD/gen.grp.*.root" --outdir $F --seed 1 \
    --stage C --sliced-msc --library "$LIB" ${STEPS_FLAGS:-} > $F/steps.log 2>&1 || { tail -20 $F/steps.log; exit 1; }
  tail -1 $F/steps.log
fi
/usr/bin/time -f "%e s %M kB" python3 -m trkfastsim seccensus "$F/fasttrk*.root" --compare "$M2FS_DATA/trk_fastsim/${REF}_tables/sgsteps.*.root" \
  --paired --outdir $O/seccensus_${REF}_${CENSUS_TAG:-$TAG} > $O/seccensus_${REF}_${CENSUS_TAG:-$TAG}.log 2>&1
tail -3 $O/seccensus_${REF}_${CENSUS_TAG:-$TAG}.log
