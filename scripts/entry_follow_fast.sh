#!/usr/bin/env bash
# Serialize the fast arm after a specific G4 driver, without occupying a second analysis core.
set -eu
previous_pid=$1
area=$2
base=$3
source_art=$4
code=$(cd "$(dirname "$0")/.." && pwd)
while kill -0 "$previous_pid" 2>/dev/null; do sleep 30; done
if ! test -f "$base/g4-full/chain.done"; then
  echo "G4 chain did not finish; fast arm not launched" >&2
  exit 1
fi
bash "$code/scripts/entry_fast_chain.sh" "$area" "$base/fast-full" "$source_art"
