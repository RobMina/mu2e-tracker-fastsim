#!/bin/bash
# Run from the laptop on the UVA network/VPN (it reaches both a Mu2e GPVM and UVA; neither reaches the other).
# Submits uva_steps_gpu.sh (sbatch from stdin), streams a tarball of the code (this tracker tree and a core tree, laptop)
# and the Mu2e-internal inputs (GPVM: gen shards, the GDML, the CPU JAX tables of jax_steps_cpu.sh) into the job's tmpfs
# (srun --overlap; never on UVA disk), waits for the job, copies the outputs (stats only) to the GPVM and removes them
# on UVA.
# usage: uva_steps_stream.sh GPVM CORE_DIR CPU_OUT DEST [SHARD ...]
#   CPU_OUT: GPVM output dir of jax_steps_cpu.sh (jc_<shard>/fasttrkCjx.*.root); DEST: GPVM dir for the outputs
# env: EXPORTS (extra sbatch --export variables, comma separated), SBATCH_OPTS, UVA_HOST
set -euo pipefail
G=$1; CORE=$2; CPU=$3; DEST=$4; shift 4
SH=("${@:-1001 1002 1003 1004 1005}"); SH=(${SH[*]})
TRK=$(cd "$(dirname "$0")/.." && pwd)
U=${UVA_HOST:-uva-hpc}; F=/scratch/ram2aq/mu2efastsim
D=/exp/mu2e/data/users/ram2aq
TMP=$(mktemp -d -t trkgpu_code); TGZ=$TMP/code.tgz
{ echo "tracker $(git -C "$TRK" rev-parse --short HEAD) + diff md5 $(cd "$TRK" && { git diff; git ls-files -o --exclude-standard | xargs cat; } | md5 -q)"
  echo "core $(git -C "$CORE" rev-parse --short HEAD) + diff md5 $(git -C "$CORE" diff | md5 -q)"; } > "$TMP/VERSION.txt"
cat "$TMP/VERSION.txt"
tar -czf "$TGZ" --exclude .git --exclude __pycache__ --exclude plots --exclude .pytest_cache --exclude '*.root' \
  -s ",^$(basename "$TRK"),tracker," -s ",^$(basename "$CORE"),core," \
  -C "$(dirname "$TRK")" "$(basename "$TRK")" -C "$(dirname "$CORE")" "$(basename "$CORE")" -C "$TMP" VERSION.txt
J=$(ssh "$U" "mkdir -p $F/kpp && sbatch --parsable ${SBATCH_OPTS:-} --export='ALL,SHARDS=${SH[*]}${EXPORTS:+,$EXPORTS}'" \
  < "$TRK/scripts/uva_steps_gpu.sh")
JID=${J%%;*}; echo "job $JID"
until INBOX=$(ssh "$U" "cat $F/kpp/inbox_$JID 2>/dev/null" </dev/null) && [ -n "$INBOX" ]; do
  st=$(ssh "$U" "squeue -h -j $JID -o %T" </dev/null || true)
  [ -z "$st" ] && { echo "job $JID left the queue before asking for input"; exit 3; }
  sleep 20
done
read -r HOST L O <<<"$INBOX"; echo "node $HOST tmpfs $L out $O"
put() {  # stdin -> $L/$1 on the job node, checked against md5 $2
  ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST sh -c 'cat > $L/$1 && md5sum $L/$1'" \
    | tee /dev/stderr | grep -q "$2" || { echo "checksum mismatch $1"; ssh "$U" "scancel $JID" </dev/null; exit 4; }
}
put code.tgz "$(md5 -q "$TGZ")" < "$TGZ"; rm -rf "$TMP"
SRCS=("$D/crv_fastsim/geometry/mu2e_extracted_v04.gdml")
for s in "${SH[@]}"; do
  SRCS+=("$D/trk_fastsim/gen_as_v1/gen.001400_0000$s.root" "?$CPU/jc_$s/fasttrkCjx.001400_0000$s.root")
done
for SRC in "${SRCS[@]}"; do   # a leading ? marks an optional input (CPU tables not made yet: no --cross for that shard)
  opt=; [ "${SRC:0:1}" = "?" ] && { opt=1; SRC=${SRC:1}; }
  N=$(basename "$SRC"); SUM=$(ssh "$G" "md5sum $SRC 2>/dev/null || true" </dev/null | cut -c1-32)
  [ -z "$SUM" ] && [ -n "$opt" ] && { echo "skipped optional $SRC"; continue; }
  [ -n "$SUM" ] || { echo "missing $SRC on $G"; ssh "$U" "scancel $JID" </dev/null; exit 4; }
  ssh "$G" "cat $SRC" </dev/null | put "$N" "$SUM"; echo "streamed $N ($SUM)"
done
ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST touch $L/ready" </dev/null
while [ -n "$(ssh "$U" "squeue -h -j $JID -o %T" </dev/null)" ]; do sleep 30; done
ssh "$U" "cp $F/kpp/slurm-$JID.out $O/slurm.out 2>/dev/null; rm -f $F/kpp/slurm-$JID.out" </dev/null
ssh "$U" "sacct -j $JID -X -o JobID,Elapsed,State,ExitCode,NodeList; tail -n 60 $O/run.log | cut -c1-400" </dev/null
ssh "$U" "tar -C $O -cf - ." </dev/null | ssh "$G" "mkdir -p $DEST/$JID && tar -C $DEST/$JID -xf - && ls $DEST/$JID"
ssh "$U" "rm -rf $O" </dev/null; echo "outputs in $G:$DEST/$JID; removed $O"
