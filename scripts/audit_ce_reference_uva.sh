#!/bin/bash
# Portable historical repository template; executed snapshot remains authoritative; must not be rerun during pause; submission must supply --output externally.
# Audit a closed inclusive reference without generation or transport.
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:25:00
#SBATCH -J r2_ce_closed_s30
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s30
D=$Y/out/ce_reference_s26
W=$Y/out/ce_closed_audit_s30
F=$Y/inputs/ce_source_s29/art/fcl/ce_reference_source.fcl
A=$B/trk_ledger_s3_20261028/evidence/tracker_runtime_stage_v30/Code_tv2.tar.bz2
R=$B/trk_field_y4_20261101/runtime/offline_uva_chain.sh
IMAGE=$B/mu2efastsim.sif.prev-20261104
POOL=$Y/inputs/ce_reference_s26/window/stop_pool.art
[ -s "$D/TRANSPORT_CLOSED" ] && [ -s "$D/COMPLETE" ]
mkdir "$W"
cd "$W"
cat > inputs.sha256 <<EOF
689e0801ab86dd527e24d3612f414fbb91c1cf526c1b37c076ab514c96895d52  $D/ce_reference.art
dcb2bd3bb0581fcf422ec081aee5ec603aa54602996042e751187fc829c09a19  $A
34311e1e6dc3668779b6f9a84ab988ca4ee293fd00a7faa6e25a0c11be55296a  $R
d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE
f507ff76e34569cc3211fd12562af04b2a55da90673d2a76e800701afed2a6b8  $F
ebb330dade519b028e823e7abdbd80317be3c1903d2aacbfe06df60db7e2b448  $POOL
EOF
sha256sum -c inputs.sha256
(cd "$D" && sha256sum -c transport.sha256 && sha256sum -c derived.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
runpy() { apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 "$@"; }
cd "$S/tracker"
runpy -m pytest -q -p no:cacheprovider tests/test_cecensus.py tests/test_cesource.py tests/test_cestopwindow.py tests/test_cecontract.py tests/test_ceinput.py tests/test_fieldce.py tests/test_cereco.py
cd "$W"
runpy -m trkfastsim.validation.cecensus --art "$D/ce_reference.art" --generation-log "$D/reference.log" --saved-log "$D/saved_read.log" --expected-events 2000 --run 1430 --subrun 6200 --fileindex-byteswap --outdir "$W/census"
runpy -m trkfastsim.validation.ceinput "$D/sgsteps.001430_00006200.root" --expected-events 2000 --census-events "$W/census/events.csv.gz" --outdir "$W/primary"
date -u +%FT%TZ > CENSUS_COMPLETE
cat > chain.sh <<EOF
set -eo pipefail
export TMPDIR=\$(dirname "\$OFFLINE_CACHE_PATH")/tmp
export SQLITE_TMPDIR=\$TMPDIR
mkdir -p "\$TMPDIR"
fhicl-dump -c "$F" > source.expanded.fcl
"\$OFFLINE_TIME" -v -o source.art.time mu2e -c source.expanded.fcl -s "$D/ce_reference.art" > source.log 2>&1
sha256sum source.log source.expanded.fcl > source.sha256
date -u +%FT%TZ > SOURCE_READ_COMPLETE
EOF
export CVMFSEXEC_DIST=$B/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$B:$B" OFFLINE_CHAIN="$W/chain.sh" OFFLINE_LOG_PREFIX=source
bash "$R" "$A" "$F" > runner.log 2>&1
runpy -m trkfastsim.validation.cesource --source-log "$W/source.log" --window-art "$POOL" --expected-events 2000 --fileindex-byteswap --out "$W/source.json"
sha256sum "$D/sgsteps.001430_00006200.root" census/* primary/* source.json source.log > audit.sha256
date -u +%FT%TZ > COMPLETE
