#!/bin/bash
# Profile stage C steps on one gen_as_v1 shard (cProfile, cumulative top 40) in several configurations.
# usage (umbrella root, site.env sourced, pyenv ana): tracker/scripts/profile_stagec.sh OUTDIR [SHARD, default 1001] [CONFIG ...]
# CONFIG: name=flags, e.g. default= nosplit=--kink-sup-xx0=0 nocalo=--no-calo
set -u
O=${1:?outdir}; S=${2:-1001}; shift 2 || shift $#
D=$M2FS_DATA/trk_fastsim
G=$D/gen_as_v1/gen.001400_0000$S.root
mkdir -p "$O"
[ $# -gt 0 ] || set -- "default=" "nosplit=--kink-sup-xx0=0" "nocalo=--no-calo" "neither=--kink-sup-xx0=0 --no-calo"
for c in "$@"; do
  n=${c%%=*}; f=${c#*=}
  /usr/bin/time -f "%e s %M kB" -o "$O/$n.time" python3 -m cProfile -o "$O/$n.prof" -m trkfastsim steps "$G" \
    --outdir "$O/$n" --seed 1 --stage C --sliced-msc $f > "$O/$n.log" 2>&1
  python3 -c "import pstats,sys; pstats.Stats(sys.argv[1]).sort_stats('cumulative').print_stats(45)" "$O/$n.prof" > "$O/$n.top.txt"
  echo "$n: $(cat $O/$n.time) | $(grep interior: $O/$n.log | cut -c1-160)"
done
