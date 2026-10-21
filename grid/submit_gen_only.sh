#!/usr/bin/env bash
# submit_gen_only.sh [--test | --cluster] [-N njobs] [--nevts N] [--first-subrun S] [--tag TAG]
# CRY primaries without Geant4 transport for fast-sim runs of a chosen live time (no tarball: Musing SimJob from CVMFS).
# Shard k: subrun FIRST_SUBRUN+k, baseSeed SEED_OFFSET+subrun (subrun 1001 -> 20261020), NEVTS events ->
# $TRKFS_PNFS/<tag>/gen.*.art; the art files are
# then dumped to gen tables on a GPVM with grid/dump_gen_only.sh.
# Sizing (grid, gen_as_v1 2026-10-20): 500 k events in 4.5-17 min of CPU per job by node type, ~1.9 GB RSS, 73 MB of
# art per job. Default 20 x 500 k = 10 M generated, ~156 s live time (as the 20 prod_as20 files).
# Prereq on the submit host: setupmu2e-art.sh, muse setup ops, valid token, site.env. Run from a data-area dir.
set -euo pipefail
MODE="test"; NJOBS=20; NEVTS=500000; FIRST_SUBRUN=1001; SEED_OFFSET=20260019; TAG=gen_as_v1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) MODE="test" ;; --cluster) MODE="cluster" ;;
    -N) NJOBS="${2:?}"; shift ;; --nevts) NEVTS="${2:?}"; shift ;; --first-subrun) FIRST_SUBRUN="${2:?}"; shift ;;
    --tag) TAG="${2:?}"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac; shift
done
OUTDIR="${TRKFS_PNFS:?source site.env}/$TAG"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$HERE/grid_gen_only.sh"; chmod +x "$WRAPPER"
FCL_PNFS="$OUTDIR/fcl"
OUTDIR_TOKEN="$(echo "$OUTDIR" | sed 's|^/pnfs/mu2e|/mu2e|')"

ifdh mkdir_p "$FCL_PNFS" 2>/dev/null || true
ifdh rm "$FCL_PNFS/gen_only_extracted.fcl" 2>/dev/null || true
ifdh cp "$HERE/gen_only_extracted.fcl" "$FCL_PNFS/gen_only_extracted.fcl"

[[ "$MODE" == "test" ]] && NJOBS=1
echo "Mode=$MODE njobs=$NJOBS nevts/job=$NEVTS subruns=$FIRST_SUBRUN..$((FIRST_SUBRUN + NJOBS - 1)) seeds=$((SEED_OFFSET + FIRST_SUBRUN))..$((SEED_OFFSET + FIRST_SUBRUN + NJOBS - 1)) out=$OUTDIR"
jobsub_submit -N "$NJOBS" -G mu2e --role=Analysis --expected-lifetime=2h \
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC \
  --lines='+FERMIHTC_AutoRelease=True' --memory=3000MB --disk=2000MB \
  --need-storage-modify="$OUTDIR_TOKEN" \
  -e NEVTS="$NEVTS" -e SEED_OFFSET="$SEED_OFFSET" -e FIRST_SUBRUN="$FIRST_SUBRUN" -e FCL_PNFS="$FCL_PNFS" \
  -e MUSING=MDC2025ay -e OWNER="${TRKFS_OWNER:-$USER}" \
  "file://$WRAPPER" "$OUTDIR"
echo "Monitor: jobsub_q -G mu2e --user $USER"
