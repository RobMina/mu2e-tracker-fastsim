#!/bin/bash
# Summarize completed census and saved-reco validation outputs without rereading ROOT inputs.
O=${1:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
pyenv ana || exit 1
set -eu
python3 "$HERE/straw_occ_report.py" \
  L="$O/seccensus_g4ref_v2_L_audit_20261027/seccensus.json" \
  Le="$O/seccensus_g4ref_v2_Le/seccensus.json" \
  L_own_selection="$O/seccensus_g4ref_v2_L_unpaired/seccensus.json" \
  L3="$O/tv2_L3_hits/tv2compare.json" L5="$O/tv2_L5/tv2compare.json" \
  G4_floor="$O/tv2_floor_g4_even_vs_odd/tv2compare.json" --out "$O/report_audit_20261027.json"
