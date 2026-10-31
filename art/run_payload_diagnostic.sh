#!/bin/bash
# Inspect only saved byte payloads using ROOT's interpreter, without compiling or changing products.
set -eo pipefail
[[ -n ${MUSE_WORK_DIR:-} && $(readlink -f "$MUSE_WORK_DIR") == "$(readlink -f "$(dirname "$0")")" ]] || exit 2
[[ $# == 3 ]] || exit 2
ROOT=$(dirname "$(dirname "$2")")
for mode in mc kpp; do
  root -l -b -q "$MUSE_WORK_DIR/TrkFastSim/macros/payload_diagnostic.C(\"$ROOT/${mode}_payload_smoke_v3/original_reco.root\",\"$ROOT/${mode}_payload_smoke_v3/replay_reco.root\")"
done
