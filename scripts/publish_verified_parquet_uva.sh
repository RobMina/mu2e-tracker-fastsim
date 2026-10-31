#!/usr/bin/env bash
set -euo pipefail
R=${1:?owned output root}
STAGE=${2:?completed validated environment}
SHARED=${3:?new shared destination}
TOOLS=${4:?versioned replay tools}
[[ ! -e "$SHARED" ]] || { echo "refusing existing shared destination" >&2; exit 1; }
[[ -f "$STAGE/installation-check.json" && -f "$STAGE/MANIFEST.sha256" ]]
(
  cd "$STAGE"
  sha256sum -c MANIFEST.sha256 > "$R/evidence/publication_hashes_${SLURM_JOB_ID:?}.txt"
  sha256sum -c base-image.sha256
)
chmod u+w "$STAGE"
cp "$TOOLS/scripts/install_shared_parquet_uva.sh" "$STAGE/install_shared_parquet_uva_v2.sh"
printf 'Publication recovery: move validated directory before removing owner write permission.\n' \
  > "$STAGE/publication-note.txt"
mv -T -n "$STAGE" "$SHARED"
[[ ! -e "$STAGE" ]] || { echo "publication refused: destination appeared" >&2; exit 1; }
chmod -R a-w "$SHARED"
printf 'Published shared environment: %s\n' "$SHARED"
bash "$TOOLS/scripts/replay_saved_rate_parquet_uva.sh" "$R" \
  /scratch/ram2aq/mu2efastsim/inputs /scratch/ram2aq/mu2efastsim/mu2efastsim.sif "$SHARED" "$TOOLS"
