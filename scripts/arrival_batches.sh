#!/bin/bash
# arrivalcheck of the current fast defaults against a G4 reference on shards [FIRST, LAST] (one GPVM; umbrella root,
# site.env sourced, pyenv ana). Extra arrivalcheck flags after OUTDIR (e.g. --crv "...").
# usage: tracker/scripts/arrival_batches.sh REF FTAG FIRST LAST OUTDIR [flags...]
set -u
REF=$1; FTAG=$2; FIRST=$3; LAST=$4; O=$5; shift 5
G=(); R=()
for k in $(seq "$FIRST" "$LAST"); do
  S=$(printf "001400_%08d" "$k")
  G+=("$CRVFS_DATA/${REF}_tables/gen.$S.root")
  R+=("$M2FS_DATA/trk_fastsim/${REF}_tables/sgsteps.dts.$CRVFS_OWNER.CosmicCRYExtractedRef.$FTAG.$S.root")
done
python3 -m trkfastsim arrivalcheck "${G[@]}" --compare "${R[@]}" --outdir "$O" "$@"
