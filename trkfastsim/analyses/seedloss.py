"""Read saved seedcheck tables into a lost-muon inventory; no transport or geometric exit inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import tables as T
from .ledgercheck import sha256


def select(frames, variant="plane"):
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[(data.variant == variant) & data.crv_pass].copy()
    if data.duplicated(T.KEY).any():
        raise ValueError("duplicate event identities across saved seedcheck inputs")
    data["lost"] = data.g4_pass & ~data.seed_pass
    data["gained"] = ~data.g4_pass & data.seed_pass
    data["straight_recovers"] = data.lost & data.straight_pass
    return data


def summarize(data):
    rows = []
    for lo, hi in zip([0, 300, 500, 1000, 2000, 5000], [300, 500, 1000, 2000, 5000, 1e9]):
        x = data[(data.p_seed >= lo) & (data.p_seed < hi)]
        lost = x[x.lost]
        rows.append({"p_seed_MeV_c": [lo, hi], "seeds": len(x), "g4_pass": int(x.g4_pass.sum()),
                     "fast_pass": int(x.seed_pass.sum()), "lost": len(lost), "gained": int(x.gained.sum()),
                     "lost_straight_recovers": int(lost.straight_pass.sum()),
                     "lost_fast_zero": int((lost.n_seed == 0).sum()),
                     "lost_fast_eleven": int((lost.n_seed == 11).sum()),
                     "lost_pmin_lt75": int((lost.fast_pmin < 75).sum())})
    return rows


def compare(base, candidate):
    if not base.set_index(T.KEY).index.sort_values().equals(candidate.set_index(T.KEY).index.sort_values()):
        raise ValueError("baseline and candidate seed identities differ")
    joined = base.merge(candidate, on=T.KEY, suffixes=("_base", "_candidate"), validate="one_to_one")
    if not np.array_equal(joined.g4_pass_base, joined.g4_pass_candidate) or not np.allclose(
            joined.p_seed_base, joined.p_seed_candidate, rtol=0, atol=1e-9):
        raise ValueError("saved arms have different reference selection or seed momenta")
    return {"matched": len(joined),
            "baseline_lost_recovered": int((joined.lost_base & joined.seed_pass_candidate).sum()),
            "baseline_selected_lost": int((joined.seed_pass_base & ~joined.seed_pass_candidate).sum()),
            "candidate_selected_gained": int((~joined.seed_pass_base & joined.seed_pass_candidate).sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--candidate", nargs="+")
    parser.add_argument("--variant", default="plane", choices=["plane", "envelope", "top"])
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    paths = T.expand(args.inputs)
    base = select([pd.read_csv(p) for p in paths], args.variant)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope": "saved event-level primary seedcheck, CRV subset; not unconditional all-muon efficiency",
              "variant": args.variant, "baseline": summarize(base),
              "inputs": [{"path": p, "sha256": sha256(p)} for p in paths],
              "limitations": "No saved last-straw or exit geometry. Zero fast count is not proof of range-out. Historical code/seed provenance is not certified."}
    if args.candidate:
        paths = T.expand(args.candidate)
        candidate = select([pd.read_csv(p) for p in paths], args.variant)
        report.update(candidate=summarize(candidate), paired=compare(base, candidate),
                      candidate_inputs=[{"path": p, "sha256": sha256(p)} for p in paths])
    lost = base.loc[base.lost & (base.p_seed < 500)].sort_values(T.KEY)
    lost.to_csv(out / "lost_below500.csv", index=False)
    report["lost_below500"] = len(lost)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs", "candidate_inputs")}, indent=2))


if __name__ == "__main__":
    main()
