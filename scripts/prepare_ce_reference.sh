#!/bin/bash
# Freeze a historical inclusive reference with stock shadow selection and fresh seeds.
set -eo pipefail
TEMPLATE=${1:?historical expanded reference}
POOL=${2:?planned disjoint stop window}
OBSERVATION=${3:?observation epilog}
OUT=${4:?fresh configuration directory}
mkdir "$OUT"
fhicl-dump -c Production/JobConfig/primary/CeEndpoint.fcl > "$OUT/stock.expanded.fcl"
cat "$TEMPLATE" > "$OUT/reference.fcl"
for label in GenFilter PrimaryFilter; do
  printf '\nphysics.filters.Stock%s : ' "$label" >> "$OUT/reference.fcl"
  sed -n "/^      $label: {/,/^      }/p" "$OUT/stock.expanded.fcl" | sed "1s/^      $label: //" >> "$OUT/reference.fcl"
done
cat >> "$OUT/reference.fcl" <<EOF

physics.filters.TargetStopResampler.fileNames : [ "$POOL" ]
physics.filters.TargetStopResampler.wrapFiles : false
services.SeedService.baseSeed : 202621100
source.firstRun : 1430
source.firstSubRun : 6200
source.maxEvents : 2000
outputs.PrimaryOutput.fileName : "ce_reference.art"
services.TFileService.fileName : "reference.hist.root"
EOF
cat "$OBSERVATION" >> "$OUT/reference.fcl"
fhicl-dump -c "$OUT/reference.fcl" > "$OUT/reference.expanded.fcl"
mu2e --print-description GenFilter > "$OUT/GenFilter.description.txt" 2>&1
mu2e --print-description DetectorStepFilter > "$OUT/DetectorStepFilter.description.txt" 2>&1
mu2e --print-description PrintModule > "$OUT/PrintModule.description.txt" 2>&1
mu2e --print-description RunSubrunEvent > "$OUT/RunSubrunEvent.description.txt" 2>&1
mu2e --version > "$OUT/runtime.version.txt"
sha256sum "$TEMPLATE" "$OBSERVATION" "$OUT/reference.fcl" "$OUT/reference.expanded.fcl" "$OUT/stock.expanded.fcl" > "$OUT/CONFIG.sha256"
printf 'seed=202621100\nfirst_run=1430\nfirst_subrun=6200\ngenerated_event_cap=2000\nsource_record_window_zero_based=[2000,4000)\nsource_expected_events=2000\npositive_event_transport_performed=false\n' > "$OUT/receipt.txt"
date -u +%FT%TZ > "$OUT/PREFLIGHT_CONFIG_COMPLETE"
