#!/bin/bash
# Fetch one completed grouped path-A output and validate it against the saved G4 arm.
O=${1:?campaign output directory}
FIRST=${2:-21}
TRK=$(cd "$(dirname "$0")/.." && pwd)
CORE=${CORE:-$TRK/../straw-core}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup ops || exit 1
pyenv ana || exit 1
source /exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/site.env
set -euo pipefail
export PYTHONPATH="$TRK:$CORE:/exp/mu2e/app/users/ram2aq/analysis/mu2e_track_crv_mcresponse"
S=$(printf '001400_%08d' "$FIRST")
R="$O/paired_test_$S"
mkdir -p "$R"
for arm in fast g4; do
  for item in "nts.ram2aq.tv2${arm}.v0.$S.root" "hits/strawhits.ram2aq.tv2${arm}.v0.$S.root"; do
    dest="$R/$(basename "$item")"
    if [ ! -s "$dest" ]; then
      ifdh cp "$TRKFS_PNFS/tv2p_$arm/$item" "$dest.partial"
      mv "$dest.partial" "$dest"
    fi
  done
done
F="$O/fast_g4ref_v2_L/fasttrkCL.grp.$S.root"
if [ ! -s "$R/coverage.json" ]; then
  python3 "$TRK/scripts/tv2_output_census.py" --steps "$F" \
    --ntuple "$R/nts.ram2aq.tv2fast.v0.$S.root" --hits "$R/strawhits.ram2aq.tv2fast.v0.$S.root" \
    --out "$R/coverage.json"
fi
refs=()
for k in $(seq "$FIRST" "$((FIRST + 9))"); do
  sr=$(printf '001400_%08d' "$k")
  refs+=("$M2FS_DATA/trk_fastsim/g4ref_v2_tables/sgsteps.dts.ram2aq.CosmicCRYExtractedRef.v1.$sr.root")
done
python3 -m trkfastsim tv2compare "$R/nts.ram2aq.tv2fast.v0.$S.root" \
  --compare "$R/nts.ram2aq.tv2g4.v0.$S.root" --steps "$F" --steps-ref "${refs[@]}" \
  --hits "$R/strawhits.ram2aq.tv2fast.v0.$S.root" --hits-ref "$R/strawhits.ram2aq.tv2g4.v0.$S.root" \
  --paired --outdir "$R/comparison" > "$R/comparison.log" 2>&1
cat "$R/comparison.log"
