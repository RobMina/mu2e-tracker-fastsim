#!/usr/bin/env bash
set -euo pipefail
R=${1:?owned output root}
SHARED=${2:?new shared environment directory}
SIF=${3:?existing project image}
TOOLS=${4:?directory containing parquet_environment_check.py}
TRACKER=${5:-$R/code/tracker}
[[ ! -e "$SHARED" ]] || { echo "refusing existing shared environment: $SHARED" >&2; exit 1; }
WORK="$R/evidence/parquet_environment_${SLURM_JOB_ID:?run inside Slurm}"
mkdir "$WORK"
STAGE="$WORK/environment"
mkdir -p "$STAGE/site-packages" "$STAGE/wheels"
module load apptainer
sha256sum "$SIF" > "$WORK/image.before.sha256"
apptainer exec --cleanenv "$SIF" python3 -m pip freeze --all > "$WORK/packages.before.txt"
apptainer exec --cleanenv -B "$R" "$SIF" python3 -m pip download \
  --disable-pip-version-check --no-cache-dir --only-binary=:all: --no-deps \
  --dest "$STAGE/wheels" pyarrow==26.0.0
sha256sum "$STAGE"/wheels/*.whl > "$WORK/wheels.sha256"
apptainer exec --cleanenv -B "$R" "$SIF" python3 -m pip install \
  --disable-pip-version-check --no-cache-dir --no-deps --no-index --only-binary=:all: \
  --target "$STAGE/site-packages" "$STAGE"/wheels/*.whl
apptainer exec --cleanenv --env "PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$STAGE/site-packages" \
  -B "$R" "$SIF" python3 "$TOOLS/parquet_environment_check.py" \
  --expected-root "$STAGE/site-packages" --out "$WORK/environment_check.json"
(
  cd "$TRACKER"
  apptainer exec --cleanenv --env "JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,PYTHONPATH=$STAGE/site-packages:$TRACKER:$R/code/core:$R/code/crv,MU2E_SEARCH_PATH=$R/code/offline_search" \
    -B "$R" "$SIF" python3 -m pytest tests/test_ratearms_provenance.py -q -p no:cacheprovider
)
apptainer exec --cleanenv "$SIF" python3 -m pip freeze --all > "$WORK/packages.after.txt"
diff -u "$WORK/packages.before.txt" "$WORK/packages.after.txt"
sha256sum "$SIF" > "$WORK/image.after.sha256"
diff -u "$WORK/image.before.sha256" "$WORK/image.after.sha256"
cp "$WORK/environment_check.json" "$STAGE/installation-check.json"
cp "$WORK/image.before.sha256" "$STAGE/base-image.sha256"
cp "$WORK/packages.before.txt" "$STAGE/base-packages.txt"
cp "$TOOLS/parquet_environment_check.py" "$STAGE/verify.py"
cat > "$STAGE/requirements.txt" <<'EOF'
# The exact binary wheel and its SHA256 are retained alongside this file.
EOF
find "$STAGE/site-packages" -maxdepth 1 -name 'pyarrow-*.dist-info' -printf '%f\n' | \
  sed -E 's/^pyarrow-(.*)\.dist-info$/pyarrow==\1/' >> "$STAGE/requirements.txt"
cat > "$STAGE/env.sh" <<'EOF'
FASTSIM_PARQUET_ENV=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FASTSIM_PARQUET_ENV
export FASTSIM_PARQUET_SITE_PACKAGES="$FASTSIM_PARQUET_ENV/site-packages"
EOF
cat > "$STAGE/README.md" <<'EOF'
# Shared Python 3.13 Parquet add-on for mu2efastsim.sif

This versioned, additive environment is shared by all project jobs.
It contains only the PyArrow binary wheel, not NumPy, pandas, JAX, or a Python interpreter.
The existing SIF, host Python and obsolete project venv are unchanged.

In a Slurm job after `module load apptainer`:

```bash
source "$FASTSIMKIT_SCRATCH/environments/pyarrow-cp313-20261030-v1/env.sh"   # FASTSIMKIT_SCRATCH: the project work area
apptainer exec --cleanenv \
  --env "PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$FASTSIM_PARQUET_SITE_PACKAGES:YOUR_CODE_PATHS" \
  "$FASTSIMKIT_SCRATCH/mu2efastsim.sif" python3 YOUR_SCRIPT.py
```

The add-on is opt-in: merely sourcing env.sh does not alter the host's PYTHONPATH.
Use --nv as usual for GPU jobs; no GPU validation is claimed for this CPU-only IO add-on.
Do not modify this version in place. Exact requirements, wheel, package hashes, base image
hash and original package list are retained. Upgrade by installing a new version and validating it.
Scratch is subject to the site's purge policy; the wheel/checker/install script must be preserved
in the project's normal source/evidence workflow for recovery. A future user-built image should
include the exact requirement in both its Apptainer and Docker dependency lists.
EOF
cp "$0" "$STAGE/install_shared_parquet_uva.sh"
(
  cd "$STAGE"
  find site-packages wheels -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
)
mkdir -p "$(dirname "$SHARED")"
[[ ! -e "$SHARED" ]] || { echo "shared destination appeared during install" >&2; exit 1; }
mv -T -n "$STAGE" "$SHARED"
[[ ! -e "$STAGE" ]] || { echo "publication refused: destination appeared" >&2; exit 1; }
chmod -R a-w "$SHARED"
printf 'Published shared environment: %s\n' "$SHARED"
