#!/bin/bash
# Stage C in-tracker scattering A/B on the 20 G4-reference gen shards: steps + v1compare + worldcheck per arm.
# Usage (umbrella root, site.env sourced, python with uproot): tracker/scripts/stagec_msc_ab.sh ARM [ARM ...]
# Arms: base (fast_c1SIs, existing outputs: v1compare/worldcheck rerun only); msc / tail (support-structure
# scattering only, outputs of 2026-10-19 15:20, before the straw/air scattering: rerun needs that code); wall (in-tracker
# scattering in support, straws and air, outputs of 2026-10-19 21:40: air was dropped after), walltail (wall + tail);
# vac / vactail (support and straws, vacuum gaps, nuclear-only tail: outputs of 2026-10-19 22:00, before the wire);
# wire (vac + the sense wire, no tail), wiretail (wire + nuclear and electron single-scattering tail);
# default (no flags: the current defaults, written to fast_c1SIsD); matfix (same, to fast_c1SIsF: the defaults after the
# core material fixes of 2026-10-21, CONCRETE_MARS iron and the GDML tracker alloys); intop (same, to fast_c1SIsT: the
# interior transport starting at the top of the support rings, above the plane); rescore_<SUFFIX> (rescore_base, rescore_X, ...): rerun the comparisons on an
# existing fast_c1SIs<SUFFIX> directory without regenerating it. Outputs under $M2FS_DATA/trk_fastsim; one line per arm in stagec_msc_ab.log.
set -u
D=$M2FS_DATA/trk_fastsim
G20="$CRVFS_DATA/g4ref_v1_tables/gen.001400_*.root"
REF="$D/g4ref_v1_tables/sgsteps.*.root"
for arm in "$@"; do
  case $arm in
    base) O=$D/fast_c1SIs; flags="--no-interior-msc --no-msc-tail" ;;
    msc)  O=$D/fast_c1SIsM; flags="--no-msc-tail" ;;
    tail) O=$D/fast_c1SIsMT; flags="--msc-tail" ;;
    wall) O=$D/fast_c1SIsW; flags="--no-msc-tail" ;;
    walltail) O=$D/fast_c1SIsWT; flags="--msc-tail" ;;
    vac) O=$D/fast_c1SIsV; flags="--no-msc-tail" ;;
    vactail) O=$D/fast_c1SIsVT; flags="--msc-tail" ;;
    wire) O=$D/fast_c1SIsX; flags="--no-msc-tail" ;;
    wiretail) O=$D/fast_c1SIsXT; flags="--msc-tail" ;;
    default) O=$D/fast_c1SIsD; flags="" ;;
    matfix) O=$D/fast_c1SIsF; flags="" ;;
    intop) O=$D/fast_c1SIsT; flags="" ;;
    rescore_*) O=$D/fast_c1SIs${arm#rescore_}; [ "${arm#rescore_}" = base ] && O=$D/fast_c1SIs; flags="" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  tag=$(basename $O | sed 's/^fast_//')
  mkdir -p $O
  case $arm in base|rescore_*) regen=0 ;; *) regen=1 ;; esac
  if [ $regen = 1 ]; then
    rm -f $O/*.root
    /usr/bin/time -f "%e" -o $O/time.log python3 -m trkfastsim steps $G20 --outdir $O --seed 1 --stage C --sliced-msc \
      $flags > $O/steps.log 2>&1 || { echo "$arm steps failed rc=$?"; tail -20 $O/steps.log; exit 1; }
  fi
  python3 -m trkfastsim v1compare "$O/fasttrkC.*.root" --compare "$REF" --outdir $D/plots/v1compare_$tag > $O/v1.log 2>&1 \
    || { echo "$arm v1compare failed"; tail -20 $O/v1.log; exit 1; }
  python3 -m trkfastsim worldcheck "$O/fasttrkC.*.root" --compare "$REF" --outdir $D/plots/worldcheck_$tag > $O/world.log 2>&1 \
    || { echo "$arm worldcheck failed"; tail -20 $O/world.log; exit 1; }
  python3 -m trkfastsim msccheck "$O/fasttrkC.*.root" --compare "$REF" --outdir $D/plots/msccheck_$tag > $O/msc.log 2>&1 \
    || { echo "$arm msccheck failed"; tail -20 $O/msc.log; exit 1; }
  echo "$arm $tag steps_wall_s=$(cat $O/time.log 2>/dev/null | tail -1) nfiles=$(ls $O/fasttrkC.*.root | wc -l)" | tee -a $D/stagec_msc_ab.log
done
