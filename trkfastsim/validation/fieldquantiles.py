"""Paired bootstrap errors for saved cosmic reco relative cores, without changing the truth proxy.

Truth remains each arm's own first-straw momentum; reco surfaces must agree. These event-bootstrap errors quantify
one file's descriptive cores, not a multi-file physics gate or a same-surface CE resolution measurement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trkfastsim.validation import ceaudit, cereco


EDGES = (300.0, 1000.0, 2000.0, 5000.0, np.inf)


def report(d):
    d = ceaudit.unique(d, "cosmic reco events")
    ia, ir = (d[f"{arm}_ntrk"].fillna(0).to_numpy() > 0 for arm in ("fast", "g4"))
    same = ia & ir & (d.fast_sid == d.g4_sid)
    ra = (d.fast_p_reco - d.fast_p_first_own) / d.fast_p_first_own
    rr = (d.g4_p_reco - d.g4_p_first_own) / d.g4_p_first_own
    valid = same & np.isfinite(ra) & np.isfinite(rr)
    out = {"events": len(d), "common_reco_surface_events": int(valid.sum()),
           "truth": "own first-straw momentum proxy; not same location as reco",
           "uncertainty": "shared-index event bootstrap within one file; no multi-file or family-wise gate", "bins": []}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = valid & (d.p_first >= lo) & (d.p_first < hi)
        core = cereco.paired_core(ra[sel], rr[sel])
        core["metrics"] = ["relative_median", "relative_IQR_sigma"]
        out["bins"].append({"reference_first_straw_p_lo": lo, "reference_first_straw_p_hi": hi, **core})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events", help="fieldreco saved events.csv.gz")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = report(pd.read_csv(a.events))
    (out / "fieldquantiles.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
