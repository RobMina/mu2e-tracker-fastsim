#!/bin/bash
# Audit completed off-axis placement tables without rerunning transport.
O=${1:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
TRK=$(cd "$HERE/.." && pwd)
CORE=${CORE:-$TRK/../straw-core}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
pyenv ana || exit 1
set -eu
export PYTHONPATH="$TRK:$CORE"
python3 "$HERE/straw_occ_identity.py" "$O/fast_g4ref_v2_L" "$O/fast_g4ref_v2_Le" --out "$O/edge_identity_20261027.json"
