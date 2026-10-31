#!/bin/bash
# G4-against-G4 floors of every T-V2/T-V3 row (tv2compare on the even against the odd prod_as20 files of the tv2_g4 arm),
# and the unpaired secondary census of a fast 10x arm against prod_as20 (the library's own training sample).
# usage (GPVM): tracker/scripts/straw_occ_floors.sh OUTDIR [FAST_TABLE_DIR]
O=$1
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana || exit 1
set -eu
python3 -c 'import numpy, awkward, uproot'
source "${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout that holds site.env}/site.env"
: "${TRKFS_OWNER:?set TRKFS_OWNER, the owner field of the tracker-stage Mu2e file names}"
FT=${2:-${M2FS_DATA:?set M2FS_DATA in site.env}/trk_fastsim/fast_as_v18L}
TRK=${TRK:-$(cd "$(dirname "$0")/.." && pwd)}
CORE=${CORE:-$TRK/../straw-core}
export PYTHONPATH=$CORE:$TRK:${CRVEXTRAP:?set CRVEXTRAP to the mu2e_track_crv_mcresponse checkout}
D=$M2FS_DATA/trk_fastsim
mapfile -t N < <(ls $D/tv2/tv2_g4/nts.*.root)
E=(); OD=(); SE=(); SO=(); HE=(); HO=()
for i in "${!N[@]}"; do
  seq=$(basename "${N[$i]}" .root | awk -F. '{print $NF}')
  s=$D/prod_as20/sgsteps.dts.mu2e.CosmicCRYExtracted.MDC2025as.$seq.root
  h=$D/tv2/tv2_g4/hits/strawhits.${TRKFS_OWNER}.tv2g4.v0.$seq.root
  if (( i % 2 == 0 )); then E+=("${N[$i]}"); SE+=("$s"); HE+=("$h"); else OD+=("${N[$i]}"); SO+=("$s"); HO+=("$h"); fi
done
python3 -m trkfastsim tv2compare "${E[@]}" --compare "${OD[@]}" --steps "${SE[@]}" --steps-ref "${SO[@]}" \
  --hits "${HE[@]}" --hits-ref "${HO[@]}" --outdir $O/tv2_floor_g4_even_vs_odd > $O/tv2_floor_g4_even_vs_odd.log 2>&1
echo "floor rc=$?"
python3 -m trkfastsim seccensus "$FT/fasttrkCL.*.root" --compare "$D/prod_as20/sgsteps.*.root" \
  --outdir $O/seccensus_$(basename $FT)_vs_as20 > $O/seccensus_$(basename $FT)_vs_as20.log 2>&1
echo "census rc=$?"
