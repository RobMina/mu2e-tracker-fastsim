#!/bin/bash
# fatecheck of the current fast defaults against a G4 reference on shards [FIRST, LAST] (one GPVM; umbrella root, site.env
# sourced, pyenv ana): gen tables from $CRVFS_DATA/${REF}_tables, G4 tracker tables from $M2FS_DATA/trk_fastsim/${REF}_tables.
# usage: tracker/scripts/fate_batches.sh REF FTAG FIRST LAST OUTDIR [SEED=1]   (seeds: file i of the batch uses (SEED, i))
set -u
REF=$1; FTAG=$2; FIRST=$3; LAST=$4; O=$5; SEED=${6:-1}
G=(); R=()
for k in $(seq "$FIRST" "$LAST"); do
  S=$(printf "001400_%08d" "$k")
  G+=("$CRVFS_DATA/${REF}_tables/gen.$S.root")
  R+=("$M2FS_DATA/trk_fastsim/${REF}_tables/sgsteps.dts.$CRVFS_OWNER.CosmicCRYExtractedRef.$FTAG.$S.root")
done
python3 -m trkfastsim fatecheck "${G[@]}" --compare "${R[@]}" --seed "$SEED" --outdir "$O"
