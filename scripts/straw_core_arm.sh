#!/bin/bash
# One 10x fast arm on gen_as_v1 (defaults + G4 secondary library) with --straw-var-scale S, then msccheck --spans vs prod_as20.
# usage (umbrella setup + PYTHONPATH on this checkout): tracker/scripts/straw_core_arm.sh TAG SCALE [extra steps flags...]
set -u
T=$1; S=$2; shift 2
D=$M2FS_DATA/trk_fastsim; O=$D/fast_$T
mkdir -p "$O"; rm -f "$O"/*.root
/usr/bin/time -f "%e" -o "$O/time.log" python3 -m trkfastsim steps "$D/gen_as_v1/gen.*.root" --outdir "$O" --seed 1 \
  --stage C --sliced-msc --library "$D/lib/trklib_mdc2025as_20f_v2.root" --straw-var-scale "$S" "$@" > "$O/steps.log" 2>&1 \
  || { echo "steps failed"; tail -20 "$O/steps.log"; exit 1; }
echo "steps done wall $(cat "$O/time.log") s, $(ls "$O"/fasttrkCL.*.root | wc -l) files"
python3 -m trkfastsim msccheck "$O/fasttrkCL.*.root" --compare "$D/prod_as20/sgsteps.*.root" --spans 1,4,10,20 \
  --outdir "$D/plots/msccheck_spans_$T" > "$O/spans.log" 2>&1 || { echo "spans failed"; tail -20 "$O/spans.log"; exit 1; }
cat "$O/spans.log"
