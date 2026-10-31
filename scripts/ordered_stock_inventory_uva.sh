#!/usr/bin/env bash
# Inspect an existing trusted runtime tarball before attempting any stock consumer.
set -euo pipefail
R=${1:?output root}
TARBALL=${2:?existing trusted muse tarball}
OUT=${3:?new evidence directory}
mkdir "$OUT"
sha256sum "$TARBALL" > "$OUT/tarball.sha256"
tar tjf "$TARBALL" > "$OUT/tarball_contents.txt"
grep -E 'StrawGasStepsFromTable|StrawHitTableDumper|KLTruthSeed|EventNtuple|Code/(setup.sh|backing)|TrkFastSim|CrvFastSim' "$OUT/tarball_contents.txt" > "$OUT/relevant_paths.txt" || true
cat "$OUT/relevant_paths.txt"
if ! grep -Eq 'lib.*StrawGasStepsFromTable.*\.so' "$OUT/tarball_contents.txt"; then
  printf '%s\n' 'BLOCKED: trusted staged tarball contains no StrawGasStepsFromTable plugin. No stock job attempted, no build requested or run.' | tee "$OUT/status.txt"
else
  printf '%s\n' 'PLUGIN_FILE_PRESENT: runtime loadability and source/event-coverage gates still required; no stock job attempted.' | tee "$OUT/status.txt"
fi
sha256sum -c "$OUT/tarball.sha256"
printf 'root=%s\njob=%s\n' "$R" "${SLURM_JOB_ID:?}" > "$OUT/job.txt"
