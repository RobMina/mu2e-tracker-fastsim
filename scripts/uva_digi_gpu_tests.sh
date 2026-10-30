#!/bin/bash
# Physics contracts on the GPU resource policy, in isolated test-file processes.
set -euo pipefail
CODE=${CODE:?core and tracker snapshots}
OUT=${OUT:?new test output directory}
SIF=${SIF:?project image}
mkdir "$OUT"
module load apptainer
rc=0
for testfile in test_digi_jax test_digi_bounded test_digi_gpu_batches test_digi_dense_limit; do
  apptainer exec --nv --cleanenv --env "JAX_PLATFORMS=cuda,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$CODE/tracker:$CODE/core,XLA_PYTHON_CLIENT_PREALLOCATE=false" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" -B "$CODE" -B "$OUT" --pwd "$CODE/tracker" "$SIF" \
    python3 -m pytest -q -p no:cacheprovider --runslow "tests/$testfile.py" > "$OUT/$testfile.log" 2>&1 || rc=1
done
exit "$rc"
