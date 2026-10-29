#!/usr/bin/env bash
# Collect completed default-arm products, audit every group, and compare all twenty paired groups.
set -euo pipefail
OLD=${1:?saved campaign directory}
DATA=${2:?tracker data directory}
BASE=${3:?tracker pnfs root}
OUT=${4:?new result directory}
TRK=$(cd "$(dirname "$0")/.." && pwd)
mkdir "$OUT"
exec > >(tee "$OUT/collect.txt") 2>&1
used=$(getfattr --only-values -n ceph.dir.rbytes /exp/mu2e/data/users/ram2aq)
limit=$(getfattr --only-values -n ceph.quota.max_bytes /exp/mu2e/data/users/ram2aq)
(( limit-used > 4000000000 )) || { echo 'ERROR: need 4 GB free for products and headroom'; exit 1; }
for arm in fast g4; do
  ifdh ls "$BASE/tv2p_$arm" 1 > "$OUT/${arm}_nts_listing.txt"
  ifdh ls "$BASE/tv2p_$arm/hits" 1 > "$OUT/${arm}_hits_listing.txt"
  ifdh ls "$BASE/tv2p_$arm/logs" 1 > "$OUT/${arm}_logs_listing.txt"
  for first in $(seq 21 10 220); do
    s=$(printf '001400_%08d' "$first")
    grep -Fxq "$BASE/tv2p_$arm/nts.ram2aq.tv2$arm.v0.$s.root" "$OUT/${arm}_nts_listing.txt"
    grep -Fxq "$BASE/tv2p_$arm/hits/strawhits.ram2aq.tv2$arm.v0.$s.root" "$OUT/${arm}_hits_listing.txt"
    grep -Fxq "$BASE/tv2p_$arm/logs/logs.$s.tgz" "$OUT/${arm}_logs_listing.txt"
  done
done
N=(); NR=(); S=(); SR=(); H=(); HR=()
mkdir "$OUT/fast" "$OUT/g4" "$OUT/coverage" "$OUT/logs"
for first in $(seq 21 10 220); do
  seq=$(printf '001400_%08d' "$first")
  f="$OLD/fast_g4ref_v2_L/fasttrkCL.grp.$seq.root"
  refs=()
  for k in $(seq "$first" "$((first+9))"); do
    s=$(printf '001400_%08d' "$k")
    refs+=("$DATA/g4ref_v2_tables/sgsteps.dts.ram2aq.CosmicCRYExtractedRef.v1.$s.root")
  done
  for arm in fast g4; do
    for item in "nts.ram2aq.tv2$arm.v0.$seq.root" "hits/strawhits.ram2aq.tv2$arm.v0.$seq.root"; do
      name=${item##*/}; dest="$OUT/$arm/$name"
      saved="$OLD/paired_test_$seq/$name"
      if [[ -s "$saved" ]]; then ln -s "$saved" "$dest"
      else ifdh cp "$BASE/tv2p_$arm/$item" "$dest.partial"; mv "$dest.partial" "$dest"; fi
    done
    ifdh cp "$BASE/tv2p_$arm/logs/logs.$seq.tgz" "$OUT/logs/$arm.$seq.tgz"
    tar tzf "$OUT/logs/$arm.$seq.tgz"
    for log in digi.log reco.log ntuple.log; do
      tar xOzf "$OUT/logs/$arm.$seq.tgz" "$log" > "$OUT/logs/$arm.$seq.$log"
    done
    if [[ "$arm" == fast ]]; then steps=("$f"); else steps=("${refs[@]}"); fi
    python3 "$TRK/scripts/tv2_output_census.py" --steps "${steps[@]}" \
      --ntuple "$OUT/$arm/nts.ram2aq.tv2$arm.v0.$seq.root" \
      --hits "$OUT/$arm/strawhits.ram2aq.tv2$arm.v0.$seq.root" \
      --out "$OUT/coverage/$arm.$seq.json" > "$OUT/coverage/$arm.$seq.txt" 2>&1
  done
  N+=("$OUT/fast/nts.ram2aq.tv2fast.v0.$seq.root"); NR+=("$OUT/g4/nts.ram2aq.tv2g4.v0.$seq.root")
  H+=("$OUT/fast/strawhits.ram2aq.tv2fast.v0.$seq.root"); HR+=("$OUT/g4/strawhits.ram2aq.tv2g4.v0.$seq.root")
  S+=("$f"); SR+=("${refs[@]}")
  echo "coverage complete: $seq (highest cycles, unique selected identities including zero-track rows)"
done
sha256sum "${N[@]}" "${NR[@]}" "${H[@]}" "${HR[@]}" > "$OUT/product_hashes.txt"
sha256sum "$TRK/trkfastsim/analyses/tv2compare.py" "$TRK/trkfastsim/analyses/occupancy.py" \
  "$TRK/trkfastsim/analyses/clusterband.py" "$TRK/scripts/tv2_output_census.py" "$0" > "$OUT/analysis_hashes.txt"
SECONDS=0
python3 -m trkfastsim tv2compare "${N[@]}" --compare "${NR[@]}" --steps "${S[@]}" --steps-ref "${SR[@]}" \
  --hits "${H[@]}" --hits-ref "${HR[@]}" --paired --outdir "$OUT/comparison" > "$OUT/comparison.txt" 2>&1
echo "tv2compare_seconds=$SECONDS" | tee "$OUT/timing.txt"
touch "$OUT/COMPLETE"
cat "$OUT/comparison.txt"
