#!/bin/bash
# arrivalcheck variants of the stage C world scattering on g4ref_v1 (umbrella root, site.env sourced, pyenv ana).
# usage: tracker/scripts/arrival_variants.sh OUTDIR [NAME=FLAGS ...]  (default: notail, nosliced, band250); FLAGS are word-split (no globs)
set -u
O=${1:?outdir}; shift
[ $# -gt 0 ] || set -- "notail=--no-msc-tail" "nosliced=--no-sliced" "band250=--band-mm 250"
G="$CRVFS_DATA/g4ref_v1_tables/gen.001400_*.root"; R="$M2FS_DATA/trk_fastsim/g4ref_v1_tables/sgsteps.*.root"
for c in "$@"; do
  n=${c%%=*}; f=${c#*=}
  python3 -m trkfastsim arrivalcheck "$G" --compare "$R" $f --outdir "$O/$n" > "$O/$n.log" 2>&1
  echo "$n rc=$?"
done
