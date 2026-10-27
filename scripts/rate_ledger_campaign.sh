#!/bin/bash
# Transport-ledger campaign: G4 per-muon table and paired g4ref_v1 + g4ref_v2 fast arms.
# usage (tracker directory, analysis environment and site.env sourced):
#   scripts/rate_ledger_campaign.sh OUT CORE_MAIN CORE_OLD [ARMSPEC...]
# CORE_MAIN / CORE_OLD: energy (2bcefde and later) / earlier momentum world conventions.
# Defaults: legacy, M, A, W0 (on CORE_OLD), AMT, AM, E; NPROC=1, NCHUNK=4, PY=python3.
# Every arm is passed to the runner, which validates and skips individual completed chunks.
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "usage: $0 OUT CORE_MAIN CORE_OLD [ARMSPEC...]" >&2
  exit 2
fi
OUT=$1; CM=$2; CO=$3; shift 3
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${CRVFS_DATA:?source site.env (CRVFS_DATA is required)}" "${M2FS_DATA:?source site.env (M2FS_DATA is required)}"
GDML=${GDML:-$CRVFS_DATA/geometry/mu2e_extracted_v04.gdml}
V1=$CRVFS_DATA/g4ref_v1_tables; V2=$CRVFS_DATA/g4ref_v2_tables
R1=$M2FS_DATA/trk_fastsim/g4ref_v1_tables; R2=$M2FS_DATA/trk_fastsim/g4ref_v2_tables
PY=${PY:-python3}; NPROC=${NPROC:-1}; NCHUNK=${NCHUNK:-4}
HOST=$(hostname)
case "$HOST" in
  mu2egpvm*)
    [ "$NPROC" = 1 ] || echo "# $HOST: forcing NPROC=1 (requested $NPROC)" >&2
    NPROC=1 ;;
esac
export PY NPROC NCHUNK
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
export ARROW_IO_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
export LC_ALL=C
ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
  ARMS=("legacy:legacy:$CM:history" "M:M:$CM:history" "A:A:$CM:history" "W0:legacy:$CO:history"
        "AMT:AMT:$CM:history" "AM:AM:$CM:history" "E:E:$CM:history")
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$HERE"
echo "# $(date -Is) start on $HOST, tracker $(git rev-parse HEAD 2>/dev/null || true) (+ working tree)"
if [ -e "$OUT/g4.parquet" ] || [ -L "$OUT/g4.parquet" ] || [ -e "$OUT/g4.parquet.json" ] || [ -L "$OUT/g4.parquet.json" ]; then
  if [ ! -f "$OUT/g4.parquet" ] || [ ! -f "$OUT/g4.parquet.json" ]; then
    echo "Partial G4 output: both $OUT/g4.parquet and its .json sidecar are required; refusing to overwrite/delete." >&2
    exit 1
  fi
  echo "# existing G4 parquet and .json kept (full validation occurs in ratearms check)"
else
  shopt -s nullglob
  G4_V1=("$R1"/sgsteps.*.001400_*.root); G4_V2=("$R2"/sgsteps.*.001400_*.root)
  if [ "${#G4_V1[@]}" -eq 0 ] || [ "${#G4_V2[@]}" -eq 0 ]; then
    echo "No G4 inputs in one or both directories: $R1, $R2" >&2
    exit 2
  fi
  if (set -o noclobber
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CM:." "$PY" -m trkfastsim.analyses.ratearms g4 \
        "${G4_V1[@]}" "${G4_V2[@]}" --out "$OUT/g4.parquet" > "$OUT/g4.log" 2>&1); then
    if [ ! -f "$OUT/g4.parquet" ] || [ ! -f "$OUT/g4.parquet.json" ]; then
      echo "G4 command returned success without parquet plus .json; refusing to continue." >&2
      exit 1
    fi
    echo "# $(date -Is) g4 table done"
  else
    rc=$?
    echo "G4 table failed (exit=$rc); existing outputs/logs retained. Inspect $OUT/g4.log or use a new output directory." >&2
    exit "$rc"
  fi
fi
bash scripts/rate_ledger_arms.sh "$OUT" "$GDML" "$V1" "$V2" "$OUT/g4.parquet" "${ARMS[@]}"
echo "# $(date -Is) arms done"
