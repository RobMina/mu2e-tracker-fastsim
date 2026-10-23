#!/bin/bash
# Follow a G4 reference grid campaign and dump each shard to CRV and tracker tables as soon as both of its art files
# are on pnfs, until every subrun in [FIRST, LAST] is dumped (or MAXH hours pass). One GPVM, serial.
# usage (umbrella root, site.env sourced): tracker/scripts/dump_g4ref_follow.sh REF FTAG FIRST LAST [MAXH=6]
# CRV tables -> $CRVFS_DATA/${REF}_tables (crv art/dump_g4ref.sh); tracker tables -> $M2FS_DATA/trk_fastsim/${REF}_tables
REF=${1:?ref}; FTAG=${2:?ftag}; FIRST=${3:?first}; LAST=${4:?last}; MAXH=${5:-6}
P=$CRVFS_PNFS/$REF; T=$M2FS_DATA/trk_fastsim/${REF}_tables; C=$CRVFS_DATA/${REF}_tables
mkdir -p "$T" "$C"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops >/dev/null 2>&1 || true
set -u                                   # after the Mu2e setup, which reads unset variables
END=$(( $(date +%s) + MAXH * 3600 ))
while :; do
  have=$(ifdh ls "$P/" 2>/dev/null | grep -o "dts\.[^ ]*\.$FTAG\.001400_[0-9]*\.art" | sed 's/.*_0*\([0-9]*\)\.art/\1/' | sort -n)
  todo=0
  for k in $have; do
    [ "$k" -ge "$FIRST" ] && [ "$k" -le "$LAST" ] || continue
    S=$(printf "001400_%08d" "$k")
    if [ ! -s "$C/crvsteps.$S.root" ] || [ ! -s "$C/gen.$S.root" ]; then
      ( REF=$REF FTAG=$FTAG bash "$CRVFS_CODE/art/dump_g4ref.sh" "$k" "$k" ) | grep -v '^done'
    fi
    if [ ! -s "$T/sgsteps.dts.$CRVFS_OWNER.CosmicCRYExtractedRef.$FTAG.$S.root" ]; then
      echo "$P/dts.$CRVFS_OWNER.CosmicCRYExtractedRef.$FTAG.$S.art" > "$T/list.$S.txt"
      ( bash "$TRKFS_CODE/art/dump_ref.sh" "$T/list.$S.txt" "$T" ) | grep -v '^done'
      rm -f "$T/list.$S.txt"
    fi
    todo=$((todo + 1))
  done
  ndone=0                                # tables of this follower's range only
  for k in $(seq "$FIRST" "$LAST"); do
    [ -s "$T/sgsteps.dts.$CRVFS_OWNER.CosmicCRYExtractedRef.$FTAG.$(printf "001400_%08d" "$k").root" ] && ndone=$((ndone + 1))
  done
  echo "$(date +%T) shards on pnfs $(echo $have | wc -w), tracker tables $ndone of $((LAST - FIRST + 1))"
  [ "$ndone" -ge $((LAST - FIRST + 1)) ] && { echo "all dumped"; break; }
  [ "$(date +%s)" -gt "$END" ] && { echo "stopping after $MAXH h"; break; }
  sleep 300
done
