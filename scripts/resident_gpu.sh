#!/bin/bash
# Run only within an allocated GPU job, against an explicit source snapshot and saved measured inputs.
set -euo pipefail
: "${RESIDENT_SOURCE:?set source snapshot root}"
: "${RESIDENT_INPUT:?set saved measured-input directory}"
: "${RESIDENT_OUTPUT:?set a new output directory}"
: "${RESIDENT_IMAGE:?set existing Apptainer image}"
: "${SLURM_JOB_ID:?allocate a GPU job before running this script}"
SRC=$(readlink -f "$RESIDENT_SOURCE")
INPUT=$(readlink -f "$RESIDENT_INPUT")
IMAGE=$(readlink -f "$RESIDENT_IMAGE")
if [ -e "$RESIDENT_OUTPUT" ]; then echo 'Output directory already exists' >&2; exit 2; fi
mkdir -p "$RESIDENT_OUTPUT"
OUT=$(readlink -f "$RESIDENT_OUTPUT")
exec > >(tee "$OUT/job.log") 2>&1
module load apptainer
printf 'host=%s job=%s source=%s input=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SRC" "$INPUT"
sha256sum "$IMAGE" > "$OUT/image.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.csv"
A() {
    apptainer exec --nv --cleanenv \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        --env JAX_PLATFORMS=cuda --env JAX_ENABLE_X64=true \
        --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
        --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH="$SRC" \
        --env JAX_COMPILATION_CACHE_DIR="$OUT/jax-cache" \
        -B "$SRC" -B "$INPUT" -B "$OUT" --pwd "$SRC" "$IMAGE" "$@"
}
A python3 -c 'import jax; print(jax.__version__, jax.devices()); assert jax.default_backend() == "gpu"'
A python3 -m pytest -q -p no:cacheprovider -W error \
    tests/test_helixseed_jax.py tests/test_helixfit_jax.py tests/test_resident.py tests/test_residentio.py \
    > "$OUT/tests.log" 2>&1
tail -5 "$OUT/tests.log"
A python3 -m trkfastsim.validation.residentcheck --require-gpu \
    --hits "$INPUT/numpy.npz" --conditions "$INPUT/conditions.json" --bz 1.0117 \
    --outdir "$OUT/development" --repeats 3 > "$OUT/development.log" 2>&1
cat "$OUT/development/report.json"
printf 'completed\n' > "$OUT/complete.txt"
