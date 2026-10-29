#!/bin/bash
# F1 closure arms on the p109 3,000-event SaveAllFits dump, one CPU, sequential. Usage: run_f1_closure.sh OUTDIR ARM...
# own_* : own hit states, every single-seed fit (yield vs stock FitOK); fixed_* : stock final hit states on stock FitOK
# tracks (T-VR2 fidelity). Suffixes: _xdoca crossing activity from the DOCA test alone (diagnostic, not the p109 rule);
# _L<part> stock time model with one part reverted to legacy (kalman_jax.TIME_PARTS); _xstock stock final crossing
# activity flags (diagnostic); _pstock also stock crossing path lengths (diagnostic); _gasp109 the p109 per-material
# gas scattering fraction. Extra vr2f1 options via F1C_EXTRA.
set -e -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/exp/mu2e/app/users/ram2aq/analysis
DATA=/exp/mu2e/data/users/ram2aq/trk_fastreco
OUT=$1; shift
INPUT=${F1C_INPUT:-$DATA/p109_f1_candidates_3k_20261025/candidates.root}
COND=${F1C_COND:-$DATA/p109_prior_all_20261025/conditions.json}
export PYTHONPATH="$HERE:$ROOT/mu2e-fastsim/core:$ROOT/mu2e-fastsim/crv"
export MU2E_SEARCH_PATH=${F1C_ANN:-/exp/mu2e/app/users/ram2aq/fastreco_capture_p109_20261026}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export F1_JAX_CACHE=$DATA/jax_cache
PY=/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3
CORE=$(awk '/Cpus_allowed_list/ {split($2,a,/[-,]/); print a[1]}' /proc/self/status)
mkdir -p "$OUT"
for arm in "$@"; do
  envs=(); opt=""
  case $arm in
    own_*) opt="--own-states";;
    fixed_*) opt="";;
    *) echo "unknown arm $arm"; exit 2;;
  esac
  rest=${arm#*_}
  case $rest in
    legacy*) opt="$opt --time-model legacy";;
    stock*) opt="$opt --time-model stock";;
    *) echo "unknown arm $arm"; exit 2;;
  esac
  for sfx in ${rest//_/ }; do
    case $sfx in
      legacy|stock) ;;
      xdoca) opt="$opt --xing-activity doca";;
      xstock) envs+=(F1_XACT=stock);;
      pstock) envs+=(F1_XACT=stock F1_PATHS=stock);;
      gasp109) envs+=(F1_GASFRAC=p109);;
      refboundary) opt="$opt --reference-model boundary";;
      refsearch) opt="$opt --reference-model search";;
      panelgeom) opt="$opt --straw-geometry panel";;
      xconstructed) opt="$opt --crossing-time constructed";;
      seednative) opt="$opt --native-seed-check ${F1C_NATIVE_CHECK:?seed arm needs F1C_NATIVE_CHECK}";;
      native) opt="$opt --native-time-check ${F1C_NATIVE_CHECK:?native arm needs F1C_NATIVE_CHECK}";;
      L*) opt="$opt --time-legacy ${sfx#L}";;
      *) echo "unknown suffix $sfx in $arm"; exit 2;;
    esac
  done
  if [[ -e "$OUT/$arm" || -e "$OUT/$arm.log" ]]; then
    echo "refusing to overwrite existing arm: $OUT/$arm" >&2
    exit 2
  fi
  mkdir "$OUT/$arm"
  (cd "$HERE" && sha256sum trkfastsim/reco/{kalman_jax,vr_f1,cli,kkmaterial,nativetime,nativeseed,strawaxis,nativegeometry}.py scripts/run_f1_closure.sh) > "$OUT/$arm/source.sha256"
  sha256sum "$MU2E_SEARCH_PATH/Offline/Mu2eKinKal/data/"Train{BkgSeed,BkgFinal,Sign_Stage0,Cluster_Stage0,Sign_Stage1,Cluster_Stage1}.dat > "$OUT/$arm/ann.sha256"
  printf '%s\n' "host=$(hostname -s)" "input=$INPUT" "conditions=$COND" "options=$opt ${F1C_EXTRA:-}" "environment=${envs[*]}" > "$OUT/$arm/command.txt"
  echo "== $arm [$opt ${envs[*]}] $(date -Is)"
  env "${envs[@]}" /usr/bin/time -v taskset -c "$CORE" $PY -c "from trkfastsim.reco.cli import main; main()" vr2f1 "$INPUT" \
    --conditions "$COND" --straw-axis $DATA/align/TrkAlignStraw.csv --save-yield-ids $opt ${F1C_EXTRA:-} \
    --outdir "$OUT/$arm" > "$OUT/$arm.log" 2>&1
  echo "== $arm done $(date -Is)"
done
