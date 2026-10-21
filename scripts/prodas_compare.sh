#!/bin/bash
# Larger-statistics validation against the MDC2025as production (unpaired: different CRY primaries, equal live time).
# usage (umbrella root, site.env sourced, pyenv ana): tracker/scripts/prodas_compare.sh STEP [STEP ...]
#   check  gen_as_v1 tables: files, subruns, events and live time against prod_as20
#   g4g4   G4 against G4: prod_as20 (Musing MDC2025as) vs g4ref_v1 (MDC2025ay) with v1compare, msccheck, stepcheck,
#          to see whether the two Musings differ on the shared distributions before using as20 as the reference
#   steps  fast StrawGasSteps (current defaults, seed 1) on the gen-only primaries gen_as_v1 (grid/: submit_gen_only.sh, then dump_gen_only.sh)
#   cmp    fast_as_v1 against prod_as20 with v1compare, msccheck, stepcheck
# Inputs under $M2FS_DATA/trk_fastsim: prod_as20/sgsteps.*.root (20 production dts, 500 k generated, 155.8 s),
# gen_as_v1/gen.*.root (20 x 500 k generated, ~156 s). Comparisons that need the same primaries (worldcheck, the
# same-straw diagnostic in v1compare) are not meaningful here; rates are per second of live time.
set -u
D=$M2FS_DATA/trk_fastsim
T=${FAST_TAG:-as_v1}    # fast run tag: fast_$T, plots/*_$T (FAST_TAG=as_v2 tracker/scripts/prodas_compare.sh steps cmp)
AS="$D/prod_as20/sgsteps.*.root"
REF="$D/g4ref_v1_tables/sgsteps.*.root"
for step in "$@"; do
  case $step in
    g4g4)
      python3 -m trkfastsim v1compare "$AS" --compare "$REF" --outdir $D/plots/v1compare_as20_vs_g4ref > $D/prodas_g4g4.v1.log 2>&1
      python3 -m trkfastsim msccheck "$AS" --compare "$REF" --outdir $D/plots/msccheck_as20_vs_g4ref > $D/prodas_g4g4.msc.log 2>&1
      python3 -m trkfastsim stepcheck "$AS" --compare "$REF" --outdir $D/plots/stepcheck_as20_vs_g4ref > $D/prodas_g4g4.step.log 2>&1
      echo "g4g4 done rc=$?" ;;
    check)
      # gen_as_v1 tables: file count, subruns, events and live time against prod_as20
      python3 - "$D" <<'PY'
import glob, sys
from mu2efastsim import tables as T
d = sys.argv[1]
fs = sorted(glob.glob(d + "/gen_as_v1/gen.*.root"))
lt, ev, sub = 0.0, 0, []
for f in fs:
    s = T.read(f, "subruns")
    lt += s.livetime.sum()
    ev += T.read(f, "gen", ["run", "subrun", "event"]).event.nunique()
    sub += s.subrun.tolist()
p = T.livetime(sorted(glob.glob(d + "/prod_as20/sgsteps.*.root")))
print(f"gen_as_v1: {len(fs)} files, subruns {min(sub)}-{max(sub)} ({len(set(sub))} unique), {ev:,} events, "
      f"live time {lt:.3f} s; prod_as20 {p:.3f} s; ratio {lt / p:.4f}")
PY
      ;;
    steps)
      O=$D/fast_$T; mkdir -p $O; rm -f $O/*.root
      /usr/bin/time -f "%e" -o $O/time.log python3 -m trkfastsim steps "$D/gen_as_v1/gen.*.root" --outdir $O --seed 1 \
        --stage C --sliced-msc > $O/steps.log 2>&1 || { echo "steps failed"; tail -20 $O/steps.log; exit 1; }
      echo "steps done wall $(cat $O/time.log) s, $(ls $O/fasttrkC.*.root | wc -l) files" ;;
    cmp)
      O=$D/fast_$T
      python3 -m trkfastsim v1compare "$O/fasttrkC.*.root" --compare "$AS" --outdir $D/plots/v1compare_$T > $O/v1.log 2>&1
      python3 -m trkfastsim msccheck "$O/fasttrkC.*.root" --compare "$AS" --outdir $D/plots/msccheck_$T > $O/msc.log 2>&1
      python3 -m trkfastsim stepcheck "$O/fasttrkC.*.root" --compare "$AS" --outdir $D/plots/stepcheck_$T > $O/step.log 2>&1
      echo "cmp done rc=$?" ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
