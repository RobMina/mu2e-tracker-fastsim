#!/bin/bash
# Transport-ledger arms on the paired g4ref_v1 + g4ref_v2 generator tables.
# usage: scripts/rate_ledger_arms.sh OUT GDML GENDIR_V1 GENDIR_V2 G4MUONS ARMSPEC...
#   ARMSPEC = NAME:PRESET:CORE[:history]; environment: NPROC=1 NCHUNK=4 SEED=1 PY=python3
# Run from the tracker directory. File i of the sorted v1 + v2 list uses seed (SEED, i).
# Restart skips only validated chunks; partial/mismatched outputs are never overwritten.
set -euo pipefail
if [ "$#" -lt 6 ]; then
  echo "usage: $0 OUT GDML GENDIR_V1 GENDIR_V2 G4MUONS ARMSPEC..." >&2
  exit 2
fi
OUT=$1; GDML=$2; V1=$3; V2=$4; G4=$5; shift 5
NPROC=${NPROC:-1}; NCHUNK=${NCHUNK:-4}; SEED=${SEED:-1}; PY=${PY:-python3}
HOST=$(hostname)
case "$HOST" in
  mu2egpvm*)
    [ "$NPROC" = 1 ] || echo "# $HOST: forcing NPROC=1 (requested $NPROC)" >&2
    NPROC=1 ;;
esac
export NPROC
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
export ARROW_IO_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
export LC_ALL=C
for value in "$NPROC" "$NCHUNK"; do
  case "$value" in ''|*[!0-9]*) echo "NPROC and NCHUNK must be positive integers" >&2; exit 2 ;; esac
  [ "$((10#$value))" -gt 0 ] || { echo "NPROC and NCHUNK must be positive integers" >&2; exit 2; }
done
NPROC=$((10#$NPROC)); NCHUNK=$((10#$NCHUNK))
shopt -s nullglob
V1_FILES=("$V1"/gen.001400_*.root); V2_FILES=("$V2"/gen.001400_*.root)
if [ "${#V1_FILES[@]}" -eq 0 ] || [ "${#V2_FILES[@]}" -eq 0 ]; then
  echo "No generator inputs in one or both directories: $V1, $V2" >&2
  exit 2
fi
FILES=("${V1_FILES[@]}" "${V2_FILES[@]}")
N=${#FILES[@]}
PER=$(( (N + NCHUNK - 1) / NCHUNK ))
echo "# $N gen files, $NCHUNK chunks of $PER, $NPROC processes" >&2
JOBS=()
for spec in "$@"; do
  IFS=: read -r NAME PRESET CORE HIST <<< "$spec"
  if [ -z "$NAME" ] || [ -z "$PRESET" ] || [ -z "$CORE" ] || { [ -n "$HIST" ] && [ "$HIST" != history ]; }; then
    echo "Invalid ARMSPEC: $spec (expected NAME:PRESET:CORE[:history])" >&2
    exit 2
  fi
  mkdir -p "$OUT/$NAME"
  for ((c = 0; c < NCHUNK; c++)); do
    i0=$(( c * PER )); [ "$i0" -ge "$N" ] && break
    JOBS+=("$NAME|$PRESET|$CORE|$HIST|$i0")
  done
done
run_one() {
  local NAME PRESET CORE HIST I0 TAG DIR LOG existing=0 path rc
  IFS='|' read -r NAME PRESET CORE HIST I0 <<< "$1"
  shift
  local F=("$@") H=()
  TAG=$(printf %04d "$I0"); DIR="$OUT/$NAME"; LOG="$DIR/fast.$TAG.log"
  # A log alone also counts as an interrupted attempt; retain it for diagnosis.
  for path in "$DIR/muons.$TAG.parquet" "$DIR/files.$TAG.jsonl" "$DIR/segments.$TAG.parquet" "$LOG"; do
    if [ -e "$path" ] || [ -L "$path" ]; then existing=1; fi
  done
  if [ "$existing" -eq 1 ]; then
    if PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CORE:." "$PY" -m trkfastsim.analyses.ratearms check "$DIR" \
      --inputs "${F[@]:$I0:$PER}" --index0 "$I0" --tag "$TAG" --arm "$PRESET" --seed "$SEED" --gdml "$GDML" --g4 "$G4"; then
      if [ "$HIST" = history ] && [ ! -f "$DIR/segments.$TAG.parquet" ]; then
        echo "Partial output for $NAME chunk $TAG: requested history is missing; nothing overwritten or deleted." >&2
        return 1
      fi
      echo "# $NAME chunk $TAG validated, skipped"
      return 0
    else
      rc=$?
      echo "Partial or mismatched output for $NAME chunk $TAG (check exit=$rc); refusing to overwrite/delete. Inspect the existing files and use a new output directory if necessary." >&2
      return "$rc"
    fi
  fi
  [ "$HIST" != history ] || H=(--history-steps)
  # noclobber also protects the log if another runner starts this chunk concurrently.
  if (set -o noclobber
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CORE:." "$PY" -m trkfastsim.analyses.ratearms fast "${F[@]:$I0:$PER}" --gdml "$GDML" \
        --arm "$PRESET" --seed "$SEED" --index0 "$I0" --g4 "$G4" --outdir "$DIR" ${H[@]+"${H[@]}"} > "$LOG" 2>&1); then
    echo "$NAME $I0 exit=0"
  else
    rc=$?
    echo "$NAME $I0 exit=$rc (see $LOG)" >&2
    return "$rc"
  fi
}
export -f run_one
export GDML G4 OUT SEED PY PER
# The single-worker path stops immediately and preserves the worker exit status.
if [ "$NPROC" -eq 1 ]; then
  for job in "${JOBS[@]}"; do
    run_one "$job" "${FILES[@]}"
  done
else
  # NUL-delimited jobs preserve spaces. Exit 255 tells xargs to stop scheduling on failure;
  # already-running workers may finish. pipefail propagates xargs' nonzero status.
  printf '%s\0' "${JOBS[@]}" | xargs -0 -P "$NPROC" -I{} bash -c 'run_one "$@" || exit 255' _ '{}' "${FILES[@]}"
fi
