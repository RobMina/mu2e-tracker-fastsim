"""Paired CE reconstruction at TT_Front against each arm's own VD13 momentum.

The reference momentum is NOT the first-straw momentum. Each arm's input hand-off state is at VD13, adjacent to
TT_Front in vacuum; reco is evaluated on the first in-bounds TT_Front segment of its first good track. Track-MC
information remains associated within that track, never matched across tracks by distance or time. Output labels
the input selection and stage boundary, with paired mean/tail errors and event-bootstrap median/IQR core errors.
"""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from trkfastsim.validation import cecontract
from trkfastsim.validation.fieldreco import KEY, TREE, _me, _vec


def read(path):
    """One vectorized pass per file; highest ntuple cycle is selected by uproot."""
    with uproot.open(path) as f:
        t = f[TREE]
        a = t.arrays([f"evtinfo/{k}" for k in KEY] + ["trk/trk.status", "trk/trk.nactive"], library="ak")
        good = a["trk/trk.status"] > 0
        has = ak.to_numpy(ak.any(good, axis=1))
        first = ak.argmax(good, axis=1, keepdims=True)
        seg = ak.fill_none(ak.firsts(t["trksegs"].array()[first], axis=1), [], axis=0)
        s = seg[(seg.sid == 0) & seg.inbounds]
        s0 = ak.firsts(s[ak.argmin(s.time, axis=1, keepdims=True)])
        valid = has & ak.to_numpy(~ak.is_none(s0))
        momentum = np.full(len(has), np.nan)
        if valid.any():
            pv = _vec(ak.fill_none(s0.mom, {"fCoordinates": {"fX": np.nan, "fY": np.nan, "fZ": np.nan}}))
            momentum[valid] = np.linalg.norm(pv[valid], axis=1)
        df = pd.DataFrame({k: ak.to_numpy(a[f"evtinfo/{k}"]) for k in KEY})
        df["track"] = has
        df["surface"] = valid
        df["p_reco"] = momentum
        df["nactive"] = np.where(has, ak.to_numpy(ak.fill_none(ak.firsts(a["trk/trk.nactive"][first]), 0)), np.nan)
        hits = ak.fill_none(ak.firsts(t["trkhits"].array()[first], axis=1), [], axis=0)
        hits = hits[hits.state > 0]
        straw = hits.plane * 768 + hits.panel * 128 + hits.straw
        event_index = np.repeat(np.arange(len(df)), ak.to_numpy(ak.num(straw)))
        pair = np.stack([event_index, ak.to_numpy(ak.flatten(straw))], axis=1)
        unique = np.unique(pair, axis=0)
        nstraw = np.bincount(unique[:, 0].astype(int), minlength=len(df))
        df["nstraw"] = np.where(has, nstraw, np.nan)
        if df.duplicated(KEY).any():
            raise ValueError(f"duplicate event keys in {path}")
        if len(df) != t.num_entries:
            raise ValueError(f"event table/tree count mismatch in {path}")
        return df, int(t.num_entries)


def core(x):
    """Median and robust sigma of the residual in MeV/c."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.array([np.nan, np.nan])
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return np.array([q[1], (q[2] - q[0]) / 1.349])


def paired_core(a, b, seed=20261103, draws=500):
    """Shared bootstrap indices preserve event pairing; resampling is vectorized, bounded to 500 replicates."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return {"events": len(a)}
    idx = np.random.default_rng(seed).integers(len(a), size=(draws, len(a)))
    qa, qb = (np.quantile(x[idx], [0.25, 0.5, 0.75], axis=1) for x in (a, b))
    ba = np.stack([qa[1], (qa[2] - qa[0]) / 1.349], axis=1)
    bb = np.stack([qb[1], (qb[2] - qb[0]) / 1.349], axis=1)
    return dict(events=len(a), candidate=core(a).tolist(), reference=core(b).tolist(),
                difference=(core(a) - core(b)).tolist(),
                paired_error=np.std(ba - bb, axis=0, ddof=1).tolist(),
                metrics=["median_MeV", "IQR_sigma_MeV"], bootstrap_draws=draws)


def compare(d, name, ref):
    ia, ir = (d[f"{n}_track"].fillna(False).to_numpy(bool) for n in (name, ref))
    sa, sr = (d[f"{n}_surface"].fillna(False).to_numpy(bool) for n in (name, ref))
    ra, rr = (d[f"{n}_p_reco"] - d[f"{n}_p_truth"] for n in (name, ref))
    both = sa & sr & np.isfinite(ra) & np.isfinite(rr)
    out = {"yield": {name: _me(ia), ref: _me(ir), "paired_difference": _me(ia.astype(float) - ir),
                     "gained": int((ia & ~ir).sum()), "lost": int((~ia & ir).sum())},
           "at_front": {name: int(sa.sum()), ref: int(sr.sum()), "paired": int(both.sum())},
           "active_hits_both_tracks": {name: _me(d.loc[ia & ir, f"{name}_nactive"]),
                                        ref: _me(d.loc[ia & ir, f"{ref}_nactive"]),
                                        "paired_difference": _me(d.loc[ia & ir, f"{name}_nactive"]
                                                                 - d.loc[ia & ir, f"{ref}_nactive"])},
           "momentum_core_paired": paired_core(ra[both], rr[both]),
           "momentum_core_all_tracks": {name: core(ra).tolist(), ref: core(rr).tolist()}, "tails": {}}
    if f"{name}_nstraw" in d and f"{ref}_nstraw" in d:
        out["unique_active_straws_both_tracks"] = {
            name: _me(d.loc[ia & ir, f"{name}_nstraw"]), ref: _me(d.loc[ia & ir, f"{ref}_nstraw"]),
            "paired_difference": _me(d.loc[ia & ir, f"{name}_nstraw"] - d.loc[ia & ir, f"{ref}_nstraw"])}
    for cut in (0.5, 1.0, 2.0):
        ta, tr = np.abs(ra[both]) > cut, np.abs(rr[both]) > cut
        out["tails"][str(cut)] = {name: _me(ta), ref: _me(tr), "paired_difference": _me(ta.astype(float) - tr)}
    return out


def assemble(denominator, entries, reference_entries, frames, ref):
    """Left-join every arm to all births, rejecting foreign/duplicate identities rather than silently dropping them."""
    d = cecontract.unique(denominator[KEY], "birth denominator")
    ent = cecontract.subset(entries, d, "candidate VD13")
    gent = cecontract.subset(reference_entries, d, "reference VD13")
    for name, frame in frames.items():
        f = cecontract.subset(frame, d, f"{name} reconstruction")
        e = gent if name == ref else ent
        p = e[KEY].assign(p_truth=np.sqrt(e.px**2 + e.py**2 + e.pz**2))
        f = f.merge(p, on=KEY, how="left", validate="one_to_one")
        d = d.merge(f.rename(columns={k: f"{name}_{k}" for k in f if k not in KEY}), on=KEY, how="left",
                    validate="one_to_one")
    return d


def unchanged_physics(report, baseline):
    """Compare saved numerical metrics recursively, treating paired NaNs as equal and rejecting any finite change."""
    def equal(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b or (np.isnan(a) and np.isnan(b))
        return a == b

    for key in ("events", "ntuple_entries", "truth", "compare"):
        if not equal(report[key], baseline[key]):
            raise ValueError(f"saved physics changed: {key}")
    return {"unchanged": True, "checked": ["events", "ntuple_entries", "truth", "compare"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="NAME=NTUPLE, one common input dts file per comparison")
    ap.add_argument("--states", required=True, help="fieldce states.root: gen gives the full denominator")
    ap.add_argument("--ref", default="g4")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--baseline-report", help="optional saved cereco.json whose event counts and physics metrics must be unchanged")
    ap.add_argument("--claim", choices=("diagnostic", "complete-birth"), default="diagnostic",
                    help="complete-birth requires an audited unfiltered chain receipt; neither option asserts closure")
    a = ap.parse_args(argv)
    specs = {}
    for spec in a.inputs:
        name, path = spec.split("=", 1)
        if not name or name in specs:
            ap.error("arm labels must be nonempty and unique")
        specs[name] = path
    if a.ref not in specs or len(specs) < 2:
        ap.error("reference and at least one candidate arm are required")
    den = cecontract.unique(tables.read(a.states, "gen", KEY), "birth denominator")
    receipt = Path(a.states).parent / "handoff.json"
    provenance = json.loads(receipt.read_text()) if receipt.is_file() else {}
    scope = cecontract.efficiency_scope(den, provenance, a.claim)
    ent = tables.read(a.states, "entries")
    gent = tables.read(a.states, "g4entries")
    counts, frames = {}, {}
    for name, path in specs.items():
        frames[name], counts[name] = read(path)
    d = assemble(den, ent, gent, frames, a.ref)
    report = dict(events=len(den), ntuple_entries=counts, efficiency_scope=scope,
                  conditional_on_dts_selection=scope["conditional_on_dts_selection"],
                  reference_selection=scope["reference_selection"],
                  truth="own VD13 entry momentum; reco at first in-bounds TT_Front, sid 0",
                  compare={name: compare(d, name, a.ref) for name in specs if name != a.ref})
    if a.baseline_report:
        report["saved_physics_check"] = unchanged_physics(report, json.loads(Path(a.baseline_report).read_text()))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cereco.json").write_text(json.dumps(report, indent=2) + "\n")
    d.to_csv(out / "events.csv.gz", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
