#!/bin/bash
#SBATCH -J trk_digijax
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o /scratch/%u/mu2efastsim/kpp/digijax-%j.out
# Synthetic-only A100 test; requires an explicit code snapshot and a new output directory.
set -euo pipefail
umask 077
CODE=${CODE:?directory containing core and tracker snapshots}
OUT=${OUT:?new statistics-only output directory}
SIF=${SIF:-/scratch/$USER/mu2efastsim/mu2efastsim.sif}
mkdir "$OUT"
module load apptainer
L=$(mktemp -d "${TMPDIR:-/tmp}/digijax_${SLURM_JOB_ID:-manual}_XXXXXX")
trap 'rm -rf "$L"; printf "%s removed %s\n" "$(hostname)" "$L" > "$OUT/cleanup.txt"' EXIT
trap 'exit 143' TERM INT
A() {
  apptainer exec --nv --cleanenv --env JAX_PLATFORMS=cuda --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH=/code/tracker:/code/core --env TMPDIR="$L" \
    -B "$CODE:/code:ro" -B "$OUT:/output" -B "$L" --pwd /code/tracker "$SIF" "$@"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/device.txt"
A python3 -c 'import jax; print(jax.__version__, jax.devices()); assert jax.default_backend() == "gpu"' >> "$OUT/device.txt" 2>&1
cp "$CODE/MANIFEST.sha256" "$OUT/source_manifest.sha256"
A python3 -m pytest -q -p no:cacheprovider tests/test_digi_jax.py tests/test_digi_clusters_jax.py > "$OUT/tests.txt" 2>&1
A python3 scripts/digi_bench.py 100 3 --backend jax --json /output/jax_100.json > "$OUT/jax_100.txt" 2>&1
A python3 scripts/digi_bench.py 100 3 --backend numpy --json /output/numpy_100.json > "$OUT/numpy_100.txt" 2>&1
printf 'complete\n' > "$OUT/completed.txt"
