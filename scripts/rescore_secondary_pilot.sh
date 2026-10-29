#!/bin/bash
# Rescore an existing grouped pilot without fetching, simulation, reconstruction or submission.
set -euo pipefail
DATA=${1:?reference tracker data directory}
SAVED=${2:?saved secondary campaign directory}
OUT=${3:?new output directory}
GDML=${4:?matching geometry file}
FIRST=${5:-21}
S=$(printf '001400_%08d' "$FIRST")
TRK=$(cd "$(dirname "$0")/.." && pwd)
R="$SAVED/paired_test_$S"
F="$SAVED/fast_g4ref_v2_L/fasttrkCL.grp.$S.root"
if [ -e "$OUT" ]; then echo "Refusing existing output: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"
refs=()
for k in $(seq "$FIRST" "$((FIRST + 9))"); do
  sr=$(printf '001400_%08d' "$k")
  refs+=("$DATA/g4ref_v2_tables/sgsteps.dts.ram2aq.CosmicCRYExtractedRef.v1.$sr.root")
done
python3 "$TRK/scripts/tv2_output_census.py" --steps "$F" \
  --ntuple "$R/nts.ram2aq.tv2fast.v0.$S.root" --hits "$R/strawhits.ram2aq.tv2fast.v0.$S.root" \
  --out "$OUT/coverage_fast.json" > "$OUT/coverage_fast.txt" 2>&1
python3 "$TRK/scripts/tv2_output_census.py" --steps "${refs[@]}" \
  --ntuple "$R/nts.ram2aq.tv2g4.v0.$S.root" --hits "$R/strawhits.ram2aq.tv2g4.v0.$S.root" \
  --out "$OUT/coverage_g4.json" > "$OUT/coverage_g4.txt" 2>&1
SECONDS=0
python3 -m trkfastsim tv2compare "$R/nts.ram2aq.tv2fast.v0.$S.root" \
  --compare "$R/nts.ram2aq.tv2g4.v0.$S.root" --steps "$F" --steps-ref "${refs[@]}" \
  --hits "$R/strawhits.ram2aq.tv2fast.v0.$S.root" --hits-ref "$R/strawhits.ram2aq.tv2g4.v0.$S.root" \
  --paired --outdir "$OUT/tv2" > "$OUT/tv2.txt" 2>&1
echo "tv2compare_seconds=$SECONDS" | tee "$OUT/timing.txt"
SECONDS=0
python3 -m trkfastsim seccensus "$F" --compare "${refs[@]}" --paired --gdml "$GDML" \
  --outdir "$OUT/census" > "$OUT/census.txt" 2>&1
echo "seccensus_seconds=$SECONDS" | tee -a "$OUT/timing.txt"
python3 "$TRK/scripts/straw_occ_report.py" "pilot=$OUT/tv2/tv2compare.json" \
  "census=$OUT/census/seccensus.json" --out "$OUT/report.json" > "$OUT/report.txt"
