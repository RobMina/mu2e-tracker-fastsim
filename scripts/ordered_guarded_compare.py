"""Saved whole-event guarded-vs-stored forward-handoff sensitivity, not precision rate closure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_rate_smoke_compare import primary_flags
from scripts.transport_capture_audit import digest
from trkfastsim import tables
from trkfastsim.analyses.ratearms import _paired


def run(candidate, control, out):
    candidate, control, out = map(Path, (candidate, control, out))
    inputs = [directory / name for directory in (candidate, control) for name in (
        "complete.json", "ordered_steps.root", "handoff_primaries.csv", "muons.csv")]
    hashes = {str(p): digest(p) for p in inputs}
    reports = [json.loads((directory / "complete.json").read_text()) for directory in (candidate, control)]
    if any(r["status"] != "bounded_full_chain_complete" for r in reports):
        raise ValueError("completed whole-event producer arms required")
    if reports[0]["handoff"] != reports[1]["handoff"]:
        raise ValueError("candidate/control handoff policy differs")
    gen = [tables.read(directory / "ordered_steps.root", "gen") for directory in (candidate, control)]
    pd.testing.assert_frame_equal(*gen, check_exact=True)
    handoff = [pd.read_csv(directory / "handoff_primaries.csv") for directory in (candidate, control)]
    pd.testing.assert_frame_equal(*handoff, check_exact=True)
    events = pd.MultiIndex.from_frame(gen[0][tables.KEY].drop_duplicates())
    flags, counts = [], []
    for directory in (candidate, control):
        sgs = tables.read(directory / "ordered_steps.root", "sgsteps")
        particles = tables.read(directory / "ordered_steps.root", "particles")
        selected = primary_flags(sgs, particles)
        flags.append(events.isin(pd.MultiIndex.from_frame(selected)))
        counts.append(dict(straw_rows=len(sgs), selected_events=len(selected), eion_MeV=float(sgs.eion.sum())))
    mu = [pd.read_csv(directory / "muons.csv") for directory in (candidate, control)]
    key = tables.KEY + ["sp"]
    merged = mu[0].merge(mu[1], on=key, suffixes=("_candidate", "_control"), validate="one_to_one")
    if len(merged) != len(mu[0]) or len(merged) != len(mu[1]):
        raise ValueError("primary denominator changed")
    live = merged.path_interior_mm_candidate.notna()
    difference = merged.loc[live, key].copy()
    for name in ("ke_final_MeV", "lost_support_MeV", "lost_thin_MeV", "path_interior_mm"):
        difference["delta_" + name] = (merged.loc[live, name + "_candidate"] - merged.loc[live, name + "_control"]).to_numpy()
    result = dict(scope="same-input/identical-upstream whole-event sensitivity, not causal G4 allocation or statistical equivalence",
        generated_events=len(events), primary_muons=len(merged), delivered=len(difference), paired=_paired(*flags),
        counts=dict(candidate=counts[0], control=counts[1]), identical_handoff=True,
        active_primary_differences=difference.to_dict("records"),
        warnings=["shared seed does not preserve variates after changed navigation", "same-policy control is not original lower-plane baseline",
                  "single bounded sample cannot close historical rate or finite-bank uncertainty"], input_sha256=hashes,
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_production=False)
    if hashes != {str(p): digest(p) for p in inputs}:
        raise ValueError("saved producer arms changed during comparison")
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sha256", "active_primary_differences")}, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("--control", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.candidate, args.control, args.out)
