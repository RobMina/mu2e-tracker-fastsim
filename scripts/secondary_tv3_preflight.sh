#!/usr/bin/env bash
# Audit the saved default-arm campaign without submitting or replacing existing products.
: "${CRVFS_OWNER:?set CRVFS_OWNER, the owner field of the CRV-stage (g4ref) Mu2e file names}"
: "${TRKFS_OWNER:?set TRKFS_OWNER, the owner field of the tracker-stage Mu2e file names}"
set -euo pipefail
OLD=${1:?saved campaign directory}
OUT=${2:?new audit directory}
BASE=${3:?tracker pnfs root}
GEN=${4:?generator campaign pnfs directory}
TRK=$(cd "$(dirname "$0")/.." && pwd)
mkdir "$OUT"
exec > >(tee "$OUT/preflight.txt") 2>&1
jobsub_q -G mu2e --user "$USER" > "$OUT/queue.txt"
cat "$OUT/queue.txt"
if grep -q 'grid_tv2' "$OUT/queue.txt"; then echo 'ERROR: existing TV2 jobs; inspect before proceeding'; exit 1; fi
ifdh ls "$BASE/tv2p_fast" 1 > "$OUT/fast.txt"
ifdh ls "$BASE/tv2p_fast/hits" 1 > "$OUT/fast_hits.txt"
ifdh ls "$BASE/tv2p_g4" 1 > "$OUT/g4.txt"
ifdh ls "$BASE/tv2p_g4/hits" 1 > "$OUT/g4_hits.txt"
ifdh ls "$BASE/tv2p_fast/tables" 1 > "$OUT/tables.txt"
ifdh ls "$GEN" 1 > "$OUT/generator.txt"
ifdh cp "$BASE/tv2p_fast/list.txt" "$OUT/list_staged_original.txt"
cmp "$OLD/list_tv2p_fast.txt" "$OUT/list_staged_original.txt"
cmp "$OLD/joblogs_fast_test/grid_tv2.sh" "$TRK/grid/grid_tv2.sh"
cp "$TRK/grid/grid_tv2.sh" "$OUT/grid_tv2.sh"
chmod +x "$OUT/grid_tv2.sh"
: > "$OUT/list_remaining19.txt"
: > "$OUT/table_hashes.txt"
n=0
while read -r inputs table extra; do
  [[ -z "$extra" && -n "$inputs" && -n "$table" ]]
  first=$((21 + 10*n)); seq=$(printf '001400_%08d' "$first")
  expected=""
  for k in $(seq "$first" "$((first+9))"); do
    s=$(printf '001400_%08d' "$k")
    path="$GEN/gen.${CRVFS_OWNER}.CosmicCRYExtractedRef.v1.$s.art"
    grep -Fxq "$path" "$OUT/generator.txt"
    expected+="${expected:+,}$path"
  done
  [[ "$inputs" == "$expected" ]]
  [[ "$table" == "$BASE/tv2p_fast/tables/fasttrkCL.grp.$seq.root" ]]
  grep -Fxq "$table" "$OUT/tables.txt"
  staged=$(sha256sum "$table"); saved=$(sha256sum "$OLD/fast_g4ref_v2_L/${table##*/}")
  [[ "${staged%% *}" == "${saved%% *}" ]]
  printf '%s\n' "$staged" >> "$OUT/table_hashes.txt"
  for kind in nts hits; do
    if [[ "$kind" == nts ]]; then prefix=nts; sub=""; else prefix=strawhits; sub=hits/; fi
    fast="$BASE/tv2p_fast/${sub}$prefix.${TRKFS_OWNER}.tv2fast.v0.$seq.root"
    ref="$BASE/tv2p_g4/${sub}$prefix.${TRKFS_OWNER}.tv2g4.v0.$seq.root"
    if [[ "$kind" == nts ]]; then fl="$OUT/fast.txt"; rl="$OUT/g4.txt"; else fl="$OUT/fast_hits.txt"; rl="$OUT/g4_hits.txt"; fi
    grep -Fxq "$ref" "$rl"
    if (( n == 0 )); then grep -Fxq "$fast" "$fl"
    elif grep -Fxq "$fast" "$fl"; then echo "ERROR: remaining output already exists: $fast"; exit 1; fi
  done
  if (( n > 0 )); then printf '%s %s\n' "$inputs" "$table" >> "$OUT/list_remaining19.txt"; fi
  printf 'group %02d first=%s table hash matches default-L; reference complete; fast=%s\n' "$n" "$first" "$([[ $n == 0 ]] && echo pilot || echo missing)"
  n=$((n+1))
done < "$OUT/list_staged_original.txt"
[[ "$n" == 20 && $(wc -l < "$OUT/list_remaining19.txt") == 19 ]]
for arm in fast g4; do
  mkdir "$OUT/fcl_$arm"
  for name in reco_kkts_extrap_nofilter_v04.fcl reco_smoke_kl_ts.fcl reco_evd_v03.fcl ntuple_kkts_extrap_v04.fcl digi_g4_alltrig.fcl; do
    ifdh cp "$BASE/tv2p_$arm/fcl/$name" "$OUT/fcl_$arm/$name"
    if [[ "$arm" == g4 ]]; then cmp "$OUT/fcl_fast/$name" "$OUT/fcl_g4/$name"; fi
  done
done
ifdh ll "$BASE/Code_tv2.tar.bz2" > "$OUT/tarball_listing.txt"
ifdh cp "$BASE/Code_tv2.tar.bz2" "$OUT/Code_tv2.tar.bz2"
sha256sum "$OUT/Code_tv2.tar.bz2" "$OUT/grid_tv2.sh" "$OUT/list_remaining19.txt" "$OUT"/fcl_fast/*.fcl > "$OUT/provenance.sha256"
tar tjf "$OUT/Code_tv2.tar.bz2" > "$OUT/tarball_contents.txt"
printf '\nquota used/limit bytes: '
getfattr --only-values -n ceph.dir.rbytes "${M2FS_DATA:?set M2FS_DATA to the data area whose quota is checked}"
printf ' / '
getfattr --only-values -n ceph.quota.max_bytes "${M2FS_DATA:?set M2FS_DATA to the data area whose quota is checked}"
printf '\nREADY: 19 missing default-L groups, pilot retained, 20 G4 groups retained; no submission.\n'
touch "$OUT/READY"
