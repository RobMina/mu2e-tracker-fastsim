#!/bin/bash
# Candidate-only unchanged cosmic stock chain; retain raw products and all exact stage identities.
set -eo pipefail
module load apptainer
B=$(readlink -f "${FASTSIMKIT_SCRATCH:?set the existing scratch root}")
Y=$B/trk_field_y4s_20261103
S=$Y/code/r2s53
H=$Y/out/cosmic_wall_pilot_s52
W=$Y/out/cosmic_wall_reco_s53
A=$B/trk_ledger_s3_20261028/evidence/tracker_runtime_stage_v30/Code_tv2.tar.bz2
R=$B/trk_field_y4_20261101/runtime/offline_uva_chain.sh
IMAGE=$B/mu2efastsim.sif.prev-20261104
TAB=$H/transport/fasttrk_field.root
DTS=$Y/inputs/cosmiccryall_ac/dts.mu2e.CosmicCRYAll.MDC2025ac.001430_00000000.art
OLD=$Y/out/art3v2_fast_unbiased/fcl
[ -s "$H/COMPLETE" ]
mkdir "$W"
cd "$W"
echo "dcb2bd3bb0581fcf422ec081aee5ec603aa54602996042e751187fc829c09a19  $A" | sha256sum -c -
echo "34311e1e6dc3668779b6f9a84ab988ca4ee293fd00a7faa6e25a0c11be55296a  $R" | sha256sum -c -
echo "d19d3945b6dd8f397d8be850021e41a4aa37eb625c7f857359ab9fe8897f0182  $IMAGE" | sha256sum -c -
(cd "$H/transport" && sha256sum -c ../closed.sha256)
(cd "$Y/inputs/cosmiccryall_ac" && sha256sum -c MANIFEST.sha256)
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
echo "17920629546059732e854392907ff2ccfb211370b08bf7b08d3c89a6c4a5245f  $OLD/inject.fcl" | sha256sum -c -
echo "ea33792f00e51366514e8e62070a5166605a8ecc5efc3ad4f298165696ed7bb2  $OLD/digi.001430_00000000.fcl" | sha256sum -c -
echo "1478e298cd81d3e3e59523dcd514d1f344ea49e2192afab5bbe1fa4b9f7f18e3  $OLD/reco.fcl" | sha256sum -c -
echo "3124c9642201245736235b4265df13e9cb905516074bde830289eb6a2b7ca55c  $OLD/ntuple.fcl" | sha256sum -c -
sha256sum "$DTS" "$TAB" "$A" "$R" "$IMAGE" "$OLD/inject.fcl" "$OLD/digi.001430_00000000.fcl" "$OLD/reco.fcl" "$OLD/ntuple.fcl" > inputs.sha256
mkdir fcl
cp "$OLD/digi.001430_00000000.fcl" fcl/digi.fcl
cp "$OLD/reco.fcl" fcl/reco.fcl
cp "$OLD/ntuple.fcl" fcl/ntuple.fcl
[ "$(grep -o 'trkTableFile : "[^"]*"' "$OLD/inject.fcl" | wc -l)" -eq 1 ]
sed "s|trkTableFile : \"[^\"]*\"|trkTableFile : \"$TAB\"|" "$OLD/inject.fcl" > fcl/inject.fcl
[ "$(grep -o 'trkTableFile : "[^"]*"' fcl/inject.fcl | wc -l)" -eq 1 ]
grep -F "trkTableFile : \"$TAB\"" fcl/inject.fcl
grep -F 'services.SeedService.baseSeed : 20261102' fcl/digi.fcl
grep -F 'outputs.CentralHelixOutput.fileName : "mcs-CH.art"' fcl/reco.fcl
grep -F 'physics.analyzers.EventNtuple.FitType : CentralHelix' fcl/ntuple.fcl
cmp "$OLD/digi.001430_00000000.fcl" fcl/digi.fcl
cmp "$OLD/reco.fcl" fcl/reco.fcl
cmp "$OLD/ntuple.fcl" fcl/ntuple.fcl
sed "s|trkTableFile : \"[^\"]*\"|trkTableFile : \"TABLE\"|" "$OLD/inject.fcl" > fcl/old_inject.normalized
sed "s|trkTableFile : \"[^\"]*\"|trkTableFile : \"TABLE\"|" fcl/inject.fcl > fcl/new_inject.normalized
cmp fcl/old_inject.normalized fcl/new_inject.normalized
cat > chain.sh <<EOF
set -eo pipefail
export TMPDIR=\$(dirname "\$OFFLINE_CACHE_PATH")/tmp
export SQLITE_TMPDIR=\$TMPDIR
mkdir -p "\$TMPDIR"
for c in inject digi reco ntuple; do fhicl-dump -c fcl/\$c.fcl > fcl/\$c.expanded.fcl; done
grep -F '"$TAB"' fcl/inject.expanded.fcl
[ "\$(awk '/^[[:space:]]*baseSeed:/ { print \$2 }' fcl/digi.expanded.fcl | wc -l)" -eq 1 ]
awk '/^[[:space:]]*baseSeed:/ { count++; if (\$2+0 != 20261102) bad=1 } END { exit (count != 1 || bad) }' fcl/digi.expanded.fcl
grep -E 'FitType:?[[:space:]]+"?CentralHelix"?' fcl/ntuple.expanded.fcl
"\$OFFLINE_TIME" -v -o inject.time mu2e -c fcl/inject.expanded.fcl -s "$DTS" > inject.log 2>&1
date -u +%FT%TZ > INJECT_CLOSED
"\$OFFLINE_TIME" -v -o digi.time mu2e -c fcl/digi.expanded.fcl -s inj.art > digi.log 2>&1
date -u +%FT%TZ > DIGI_CLOSED
"\$OFFLINE_TIME" -v -o reco.time mu2e -c fcl/reco.expanded.fcl -s dig.art > reco.log 2>&1
date -u +%FT%TZ > RECO_CLOSED
"\$OFFLINE_TIME" -v -o ntuple.time mu2e -c fcl/ntuple.expanded.fcl -s mcs-CH.art -T nts.001430_00000000.root > ntuple.log 2>&1
date -u +%FT%TZ > NTUPLE_CLOSED
sha256sum inj.art dig.art mcs-CH.art nts.001430_00000000.root fcl/*.expanded.fcl > raw.sha256
EOF
export CVMFSEXEC_DIST=$B/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$B:$B" OFFLINE_CHAIN="$W/chain.sh" OFFLINE_LOG_PREFIX=reco
bash "$R" "$A" "$W/fcl/digi.fcl" > runner.log 2>&1
sha256sum -c inputs.sha256
sha256sum -c raw.sha256
for stage in inject digi reco ntuple; do
  [ -s "$stage.log" ] && [ -s "$stage.time" ] && [ -s "fcl/$stage.expanded.fcl" ]
  grep -E 'Exit status:[[:space:]]+0$' "$stage.time"
done
export APPTAINERENV_PYTHONPATH=$S/core:$S/tracker
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1
apptainer exec --cleanenv -B "$B:$B" "$IMAGE" python3 -m trkfastsim.validation.fieldwallhistory \
  --source-table "$Y/out/tables/cosmiccryall_ac/sgsteps.001430_00000000.root" --candidate-table "$TAB" \
  --stage-dir "$W" --legacy-dir "$Y/out" --outdir "$W/history"
(cd "$S/tracker" && sha256sum -c CENSUS.sha256)
(cd "$S/core" && sha256sum -c CENSUS.sha256)
sha256sum -c inputs.sha256
sha256sum -c raw.sha256
sha256sum inj.art dig.art mcs-CH.art nts.001430_00000000.root history/* fcl/*.fcl inject.log digi.log reco.log ntuple.log runner.log inject.time digi.time reco.time ntuple.time reco.total.time chain.sh inputs.sha256 raw.sha256 > closed.sha256
date -u +%FT%TZ > COMPLETE
