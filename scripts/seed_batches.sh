#!/bin/bash
# seedcheck of the current defaults on G4 reference shards [FIRST, LAST] (one GPVM; umbrella root, site.env, pyenv ana).
# usage: tracker/scripts/seed_batches.sh REF FTAG FIRST LAST OUTDIR [seedcheck flags...]
set -u
REF=$1; FTAG=$2; FIRST=$3; LAST=$4; O=$5; shift 5
S=(); C=()
for k in $(seq "$FIRST" "$LAST"); do
  q=$(printf "001400_%08d" "$k")
  S+=("$M2FS_DATA/trk_fastsim/${REF}_tables/sgsteps.dts.$CRVFS_OWNER.CosmicCRYExtractedRef.$FTAG.$q.root")
  C+=("$CRVFS_DATA/${REF}_tables/crvsteps.$q.root")
done
python3 -m trkfastsim seedcheck "${S[@]}" --crv "${C[@]}" --outdir "$O" "$@"
