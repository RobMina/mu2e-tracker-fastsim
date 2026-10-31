#!/bin/bash
# Grid lists for a grouped T-V2/T-V3 path A campaign (grid_tv2.sh: several comma-separated inputs per job, one merged
# fast table per job). usage (GPVM, pyenv ana, site.env): tracker/scripts/tv2_group_lists.sh MODE TABLES OUTDIR TAG [GROUP]
#   MODE g4ref_v2: paired lists on the g4ref_v2 subruns 21-220 (gen stream for the fast arm, dts for the G4 arm)
#   MODE gen_as_v1: fast list on the gen_as_v1 subruns 1001-1020 (one file per job, no merge; stage with TABLES_DIR=TABLES)
# TABLES: dir of the fast tables, g4ref_v2: fasttrkCL.grp.001400_<first subrun>.root (straw_occ_steps.sh, GROUP subruns
# each), gen_as_v1: fasttrkCL.001400_<subrun>.root. Writes OUTDIR/list_<TAG>_{fast,g4}.txt; stage the tables with
# submit_tv2.sh TABLES_DIR=TABLES.
: "${CRVFS_OWNER:?set CRVFS_OWNER, the owner field of the CRV-stage (g4ref) Mu2e file names}"
: "${TRKFS_OWNER:?set TRKFS_OWNER, the owner field of the tracker-stage Mu2e file names}"
set -u
MODE=$1; TAB=$2; O=$3; TAG=$4; G=${5:-10}
mkdir -p "$O"
LF=$O/list_${TAG}_fast.txt; LG=$O/list_${TAG}_g4.txt
: > "$LF"
case $MODE in
  g4ref_v2)
    : > "$LG"
    for start in $(seq 21 "$G" 220); do
      gens=(); dts=()
      for k in $(seq "$start" $((start + G - 1))); do
        S=$(printf "001400_%08d" "$k")
        gens+=("$CRVFS_PNFS/g4ref_v2/gen.${CRVFS_OWNER}.CosmicCRYExtractedRef.v1.$S.art")
        dts+=("$CRVFS_PNFS/g4ref_v2/dts.${CRVFS_OWNER}.CosmicCRYExtractedRef.v1.$S.art")
      done
      M=fasttrkCL.grp.$(printf "001400_%08d" "$start").root
      if [ -s "$TAB/$M" ]; then echo "$(IFS=,; echo "${gens[*]}") $TRKFS_PNFS/${TAG}_fast/tables/$M" >> "$LF"
      else echo "no fast table $TAB/$M: fast line skipped"; fi
      echo "$(IFS=,; echo "${dts[*]}")" >> "$LG"
    done ;;
  gen_as_v1)
    for k in $(seq 1001 1020); do
      S=$(printf "001400_%08d" "$k")
      echo "$TRKFS_PNFS/gen_as_v1/gen.${TRKFS_OWNER}.CosmicCRYExtractedGen.v0.$S.art $TRKFS_PNFS/${TAG}_fast/tables/fasttrkCL.$S.root" >> "$LF"
    done ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
wc -l "$LF" ${LG:+$LG} 2>/dev/null
