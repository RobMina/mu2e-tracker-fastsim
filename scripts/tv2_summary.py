"""Side-by-side summary of tv2compare.json outputs (T-V2/T-V3 headline metrics).

    python3 tracker/scripts/tv2_summary.py LABEL=DIR/tv2compare.json [...]
"""
import json
import sys

runs = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in sys.argv[1:]]


def f(d, k):
    v = d.get(k)
    if v is None:
        return "n/a (needs --hits)"
    if k == "yield_per_s":
        return f"{v['fast']:.1f} vs {v['g4']:.1f} ({v['ratio_minus_1']:+.3f} +- {v['err']:.3f})"
    if k.startswith("resid_"):
        return f"width {v['width'][0]:.2f} vs {v['width'][1]:.2f} ({v['width'][2]:+.3f}); bias {v['bias'][0]:+.2f} vs {v['bias'][1]:+.2f}"
    if k in ("nactive_mean", "digis_per_event"):
        return f"{v[0]:.2f} vs {v[1]:.2f} ({v[2]:+.4f})"
    if k == "tracks_per_selected_event":
        return f"{v[0]:.4f} vs {v[1]:.4f}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return json.dumps(v)[:100]


keys = ["selected_events", "tracks", "yield_per_s", "tracks_per_selected_event", "nactive_mean", "chi2ndf_band",
        "fitcon_band", "digis_per_event", "tdc0_band", "tot0_band", "panel_occupancy_chi2ndf",
        "resid_dx_mm", "resid_dy_mm", "resid_dang_mrad", "resid_dt_ns"]
for k in keys:
    for lab, d in runs:
        print(f"{k:26s} {lab:4s} {f(d, k)}")
