#!/usr/bin/env bash
# Run full-census NumPy transport, then the shared current Offline chain in a fresh shell.
set -e
area=$1
out=$2
source_art=$3
code=$(cd "$(dirname "$0")/.." && pwd)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
pyenv ana
set -u
export PYTHONPATH="$code:$code/../core"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
base=$(dirname "$out")
cd "$code"
test ! -e "$out/fasttrk_field.root"
/usr/bin/time -v -o "$out/transport.time" python3 -m trkfastsim.validation.entry_fast "$base/prepared/entries.root" \
  --metadata "$base/prepared/entries.json" --gdml "$base/prepared/mu2e_run1a_v01.gdml" \
  --map /cvmfs/mu2e.opensciencegrid.org/DataFiles/BFieldMaps/Mau13/DSMap.header \
  --origin -3904 0 10170.999666398546 --seed 20261105 --outdir "$out" > "$out/transport.log" 2>&1
ln -s fasttrk_field.root "$out/transport.root"
deactivate
bash "$code/scripts/entry_chain.sh" "$area" "$out" "$source_art"
pyenv ana
python3 -m trkfastsim.validation.entry_benchmark --entries "$base/prepared/entries.root" \
  --fast "$out" --g4 "$base/g4-full" --output "$base/benchmark.json" > "$base/benchmark.log" 2>&1
