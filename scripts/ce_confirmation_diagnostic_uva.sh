#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Read completed comparisons and actual saved stock-shadow decisions; never tune or rerun transport.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:20:00
#SBATCH -J r2_ce_diag_s34
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s34
P=$Y/out/ce_confirm_compare_s33
W=$Y/out/ce_confirm_diagnostic_s34
IMAGE=$B/mu2efastsim.sif.prev-20261104
[ -s "$P/COMPLETE" ]
mkdir "$W"
cd "$W"
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$P" && sha256sum -c compare.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
for mode in entry chain; do (cd "$Y/out/ce_confirm_${mode}_s31" && sha256sum -c transport.sha256); done
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
(cd "$S/tracker" && runpy -m pytest -q -p no:cacheprovider tests/test_cestockfilter.py tests/test_cerootcause.py)
runpy -m trkfastsim.validation.cerootcause --entry-events "$P/entry/events.csv.gz" --chain-events "$P/chain/events.csv.gz" \
  --entry-dir "$Y/out/ce_confirm_entry_s31" --chain-dir "$Y/out/ce_confirm_chain_s31" \
  --gdml "$B/hardlib_proto/data/geometry/mu2e_run1a_v01.gdml" --outdir "$W/rootcause"
runpy -m trkfastsim.validation.cestockfilter seed202621100="$W/rootcause/events.csv.gz" \
  --census-events "$Y/out/ce_closed_audit_s30/census/events.csv.gz" --expected-events 2000 --outdir "$W/stock_conditioning"
sha256sum rootcause/* stock_conditioning/* > diagnostic.sha256
date -u +%FT%TZ > COMPLETE
