#!/bin/bash
# Stage C in-tracker scattering A/B on the 20 G4-reference gen shards: steps + v1compare + worldcheck per arm.
# Usage (umbrella root, site.env sourced, python with uproot): tracker/scripts/stagec_msc_ab.sh ARM [ARM ...]
# Arms: base (fast_c1SIs, existing outputs: v1compare/worldcheck rerun only); msc / tail (support-structure
# scattering only, outputs of 2026-10-19 15:20, before the straw/air scattering: rerun needs that code); wall (in-tracker
# scattering in support, straws and air, outputs of 2026-10-19 21:40: air was dropped after), walltail (wall + tail);
# vac (in-tracker scattering in support and straws, vacuum gaps as in the GDML: the current default), vactail (vac + tail). Outputs under $M2FS_DATA/trk_fastsim; one line per arm in stagec_msc_ab.log.
set -u
D=$M2FS_DATA/trk_fastsim
G20="$CRVFS_DATA/g4ref_v1_tables/gen.001400_*.root"
REF="$D/g4ref_v1_tables/sgsteps.*.root"
for arm in "$@"; do
  case $arm in
    base) O=$D/fast_c1SIs; flags="--no-interior-msc" ;;
    msc)  O=$D/fast_c1SIsM; flags="" ;;
    tail) O=$D/fast_c1SIsMT; flags="--msc-tail" ;;
    wall) O=$D/fast_c1SIsW; flags="" ;;
    walltail) O=$D/fast_c1SIsWT; flags="--msc-tail" ;;
    vac) O=$D/fast_c1SIsV; flags="" ;;
    vactail) O=$D/fast_c1SIsVT; flags="--msc-tail" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  tag=$(basename $O | sed 's/^fast_//')
  mkdir -p $O
  if [ "$arm" != base ]; then
    rm -f $O/*.root
    /usr/bin/time -f "%e" -o $O/time.log python3 -m trkfastsim steps $G20 --outdir $O --seed 1 --stage C --sliced-msc \
      $flags > $O/steps.log 2>&1 || { echo "$arm steps failed rc=$?"; tail -20 $O/steps.log; exit 1; }
  fi
  python3 -m trkfastsim v1compare "$O/fasttrkC.*.root" --compare "$REF" --outdir $D/plots/v1compare_$tag > $O/v1.log 2>&1 \
    || { echo "$arm v1compare failed"; tail -20 $O/v1.log; exit 1; }
  python3 -m trkfastsim worldcheck "$O/fasttrkC.*.root" --compare "$REF" --outdir $D/plots/worldcheck_$tag > $O/world.log 2>&1 \
    || { echo "$arm worldcheck failed"; tail -20 $O/world.log; exit 1; }
  echo "$arm $tag steps_wall_s=$(cat $O/time.log 2>/dev/null | tail -1) nfiles=$(ls $O/fasttrkC.*.root | wc -l)" | tee -a $D/stagec_msc_ab.log
done
