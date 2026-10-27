#!/bin/bash
# tv2compare reruns on existing unpaired grid outputs against tv2_g4 (prod_as20): L3 (tv2_fastL3, the only fast arm with
# hit tables; fetched from pnfs by explicit name) for the panel occupancy with clustered errors and the pre-fix binning,
# L5 (tv2_fastL5) for the factorised yield and jackknife widths, and the G4 even/odd-file floors of every row.
# usage (GPVM): tracker/scripts/straw_occ_tv2_rerun.sh OUTDIR
O=$1
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops || exit 1
pyenv ana || exit 1
set -eu
python3 -c 'import numpy, awkward, uproot'
source /exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/site.env
TRK=${TRK:-$(cd "$(dirname "$0")/.." && pwd)}
CORE=${CORE:-$TRK/../straw-core}
export PYTHONPATH=$CORE:$TRK:/exp/mu2e/app/users/ram2aq/analysis/mu2e_track_crv_mcresponse
D=$M2FS_DATA/trk_fastsim
L3=$O/tv2_fastL3
mkdir -p $L3/nts $L3/hits $L3/tables
for k in $(seq 1001 1020); do
  S=$(printf "001400_%08d" $k)
  for f in "nts/nts.ram2aq.tv2fast.v0.$S.root:" "hits/strawhits.ram2aq.tv2fast.v0.$S.root:hits/" "tables/fasttrkCL.$S.root:tables/"; do
    loc=${f%%:*}; sub=${f##*:}
    [ -s $L3/$loc ] || ifdh cp $TRKFS_PNFS/tv2_fastL3/$sub$(basename $loc) $L3/$loc > /dev/null 2>&1 || echo "fetch failed: $loc"
  done
done
mapfile -t N < <(ls $D/tv2/tv2_g4/nts.*.root)
SR=(); HR=(); E=(); OD=(); SE=(); SO=(); HE=(); HO=()
for i in "${!N[@]}"; do
  seq=$(basename "${N[$i]}" .root | awk -F. '{print $NF}')
  s=$D/prod_as20/sgsteps.dts.mu2e.CosmicCRYExtracted.MDC2025as.$seq.root
  h=$D/tv2/tv2_g4/hits/strawhits.ram2aq.tv2g4.v0.$seq.root
  SR+=("$s"); HR+=("$h")
  if (( i % 2 == 0 )); then E+=("${N[$i]}"); SE+=("$s"); HE+=("$h"); else OD+=("${N[$i]}"); SO+=("$s"); HO+=("$h"); fi
done
run() { local tag=$1; shift; /usr/bin/time -f "%e s %M kB" python3 -m trkfastsim tv2compare "$@" --outdir $O/$tag > $O/$tag.log 2>&1; echo "$tag rc=$?"; }
run tv2_L3_hits "$L3"/nts/nts.*.root --compare "${N[@]}" --steps "$L3"/tables/fasttrkCL.*.root --steps-ref "${SR[@]}" \
  --hits "$L3"/hits/*.root --hits-ref "${HR[@]}"
run tv2_L5 $D/tv2/tv2_fastL5/nts.*.root --compare "${N[@]}" --steps $D/fast_as_v16Lx/fasttrkCL.*.root --steps-ref "${SR[@]}"
run tv2_floor_g4_even_vs_odd "${E[@]}" --compare "${OD[@]}" --steps "${SE[@]}" --steps-ref "${SO[@]}" --hits "${HE[@]}" --hits-ref "${HO[@]}"
# T-V1b census unpaired (each arm's own selection) on the paired campaign's fast tables, against the paired run
/usr/bin/time -f "%e s %M kB" python3 -m trkfastsim seccensus "$O/fast_g4ref_v2_L/fasttrkCL.*.root" \
  --compare "$D/g4ref_v2_tables/sgsteps.*.root" --outdir $O/seccensus_g4ref_v2_L_unpaired > $O/seccensus_g4ref_v2_L_unpaired.log 2>&1
echo "census unpaired rc=$?"
