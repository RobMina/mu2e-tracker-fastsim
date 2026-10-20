"""Summarise stage C A/B arms from their v1compare and worldcheck JSONs: python3 stagec_msc_ab_summary.py D TAG [TAG ...]."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
for tag in sys.argv[2:]:
    v = json.loads((D / f"plots/v1compare_{tag}/v1compare.json").read_text())
    r = v["rate_per_s"]
    fs = v["first_straw"]
    print(f"== {tag}: rate {r['ratio_minus_1']:+.3f}+-{r['stat_err_rel']:.3f}  events f/g4 {v['events_selected']['fast']}/{v['events_selected']['g4']}"
          f"  straws/mu {v['straws_per_muon_mean']['ratio_minus_1']:+.4f}  len {v['len_mean']['ratio_minus_1']:+.4f}"
          f"  eion mean {v['eion_keV']['mean_ratio_minus_1']:+.4f}  plane-share maxdev {v['per_plane_share_ratio_minus_1_maxabs']:.3f}")
    print("   pass:", {k: x for k, x in v["thresholds_pass"].items()})
    for k in ("ux", "uz", "x", "z"):
        b = fs[k]["band"]
        print(f"   first straw {k}: q10/q90 fast {[round(x, 4) for x in fs[k]['q10_q90_fast']]} g4 {[round(x, 4) for x in fs[k]['q10_q90_g4']]}"
              f"  band-2s {b.get('max_abs_dev_minus_2sigma')}")
    print("   deflection first->last straw [fast, g4]:")
    for pb, x in v.get("deflection_first_to_last", {}).items():
        print(f"     |p| {pb:>11s} n {x['n']}  med {[round(a, 2) for a in x['median_mrad']]}  q90 {[round(a, 2) for a in x['q90_mrad']]}"
              f"  q99 {[round(a, 1) for a in x['q99_mrad']]}  >10x med {[round(a, 4) for a in x['share_above_10x_g4_median']]}")
    w = D / f"plots/worldcheck_{tag}/worldcheck.json"
    if w.exists():
        wj = json.loads(w.read_text())
        keys = [k for k in wj if "term" in k or "summary" in k][:3]
        for k in keys:
            print(f"   worldcheck {k}: {wj[k]}")
