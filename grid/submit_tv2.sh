#!/usr/bin/env bash
# submit_tv2.sh --arm fast|g4 --list LIST.txt --tarball <Code.tar.bz2|--use-staged> [--test | --cluster [-N n]] [--nevts N] [--tag TAG]
# T-V2/T-V3: digi -> stock reco -> EventNtuple with one code tarball for both arms (grid_tv2.sh).
#   fast: LIST lines "<gen art on pnfs> <fast table on pnfs>"; g4: LIST lines "<dts art>" (prod_as20 dts_list.txt).
# Stages the list, the reco/ntuple fcls (mu2e_track_crv_mcresponse/fcl) and digi_g4_alltrig.fcl to $TRKFS_PNFS/<tag>/,
# and the tarball once to $TRKFS_PNFS/Code_tv2.tar.bz2 (build it with muse tarball on mu2ebuild02; never here).
# fast arm: TABLES_DIR=<dir of fasttrkC.*.root> stages the tables to <tag>/tables on pnfs (skipped if present).
# Prereq on the submit host: setupmu2e-art.sh, muse setup ops, valid token, site.env. Run from a data-area dir.
set -euo pipefail
ARM=""; LIST=""; TARBALL=""; MODE="test"; NJOBS=""; NEVTS=-1; TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="${2:?}"; shift ;; --list) LIST="${2:?}"; shift ;; --tarball) TARBALL="${2:?}"; shift ;;
    --test) MODE="test" ;; --cluster) MODE="cluster" ;; -N) NJOBS="${2:?}"; shift ;;
    --nevts) NEVTS="${2:?}"; shift ;; --tag) TAG="${2:?}"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac; shift
done
[[ "$ARM" == fast || "$ARM" == g4 ]] || { echo "--arm fast|g4"; exit 1; }
[[ -s "$LIST" && -n "$TARBALL" ]] || { echo "--list and --tarball required"; exit 1; }
TAG=${TAG:-tv2_$ARM}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$HERE/grid_tv2.sh"; chmod +x "$WRAPPER"
BASE="${TRKFS_PNFS:?source site.env}"
OUTDIR="$BASE/$TAG"; FCL_PNFS="$OUTDIR/fcl"; LIST_PNFS="$OUTDIR/list.txt"; PNFS_TARBALL="$BASE/Code_tv2.tar.bz2"
OUTDIR_TOKEN="$(echo "$OUTDIR" | sed 's|^/pnfs/mu2e|/mu2e|')"
RECO_FCL="${M2FS_CODE:?}/../mu2e_track_crv_mcresponse/fcl"

ifdh mkdir_p "$FCL_PNFS" 2>/dev/null || true
for f in reco_kkts_extrap_nofilter_v04.fcl reco_smoke_kl_ts.fcl reco_evd_v03.fcl ntuple_kkts_extrap_v04.fcl; do
  ifdh rm "$FCL_PNFS/$f" 2>/dev/null || true; ifdh cp "$RECO_FCL/$f" "$FCL_PNFS/$f"
done
ifdh rm "$FCL_PNFS/digi_g4_alltrig.fcl" 2>/dev/null || true
ifdh cp "$HERE/../art/TrkFastSim/fcl/digi_g4_alltrig.fcl" "$FCL_PNFS/digi_g4_alltrig.fcl"
ifdh rm "$LIST_PNFS" 2>/dev/null || true; ifdh cp "$LIST" "$LIST_PNFS"
if [[ "$ARM" == fast && -n "${TABLES_DIR:-}" ]]; then   # TABLES_DIR: local fast tables named as in the list's 2nd column
  ifdh mkdir_p "$OUTDIR/tables" 2>/dev/null || true
  awk '{print $2}' "$LIST" | while read -r T; do
    ifdh ls "$T" >/dev/null 2>&1 || ifdh cp "$TABLES_DIR/$(basename "$T")" "$T"
  done
fi
if [[ "$TARBALL" == "--use-staged" ]]; then
  ifdh ls "$PNFS_TARBALL" >/dev/null 2>&1 || { echo "ERROR: no staged $PNFS_TARBALL"; exit 1; }
elif ifdh ls "$PNFS_TARBALL" >/dev/null 2>&1; then
  echo "tarball already staged: $PNFS_TARBALL (ifdh rm it to refresh)"
else
  ifdh cp "$TARBALL" "$PNFS_TARBALL"
fi
NTOT=$(grep -c . "$LIST"); [[ "$MODE" == test ]] && N=1 || N=${NJOBS:-$NTOT}
echo "arm=$ARM mode=$MODE jobs=$N of $NTOT nevts=$NEVTS out=$OUTDIR tarball=$PNFS_TARBALL"
jobsub_submit -N "$N" -G mu2e --role=Analysis --expected-lifetime=4h \
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC \
  --lines='+FERMIHTC_AutoRelease=True' --memory=4000MB --disk=10000MB \
  --need-storage-modify="$OUTDIR_TOKEN" \
  -e ARM="$ARM" -e LIST_PNFS="$LIST_PNFS" -e FCL_PNFS="$FCL_PNFS" -e CODE_TARBALL="$PNFS_TARBALL" -e NEVTS="$NEVTS" \
  "file://$WRAPPER" "$OUTDIR"
echo "Monitor: jobsub_q -G mu2e --user $USER"
