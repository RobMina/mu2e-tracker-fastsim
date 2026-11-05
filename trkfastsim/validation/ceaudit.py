"""Attribute a selected-dts CE pilot using saved entry, chain and common Geant4 arms.

All state-conditioned summaries are truth diagnostics, never correction models or generated-electron efficiencies.
The optional Geant4 time cut uses the entry-control fast exit clock, not an independently measured G4 first passage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim.validation import cereco, fieldce
from trkfastsim.validation.cecontract import integer_keys, unique
from trkfastsim.validation.fieldreco import _me

KEY = tables.KEY


def join_events(entry, chain):
    """Require identical denominators and the identical saved Geant4 baseline in both comparisons."""
    entry, chain = unique(entry, "entry"), unique(chain, "chain")
    pd.testing.assert_frame_equal(entry[KEY], chain[KEY], check_dtype=False, check_exact=True)
    baseline = [k for k in entry if k.startswith("g4_")]
    pd.testing.assert_frame_equal(entry[baseline], chain[baseline])
    e = entry.rename(columns={k: k.replace("fast_", "entry_", 1) for k in entry if k.startswith("fast_")})
    c = chain[KEY + [k for k in chain if k.startswith("fast_")]]
    c = c.rename(columns={k: k.replace("fast_", "chain_", 1) for k in c if k.startswith("fast_")})
    d = e.merge(c, on=KEY, validate="one_to_one")
    for arm in ("g4", "entry", "chain"):
        for flag in ("track", "surface"):
            col = f"{arm}_{flag}"
            d[col] = d[col].fillna(False).astype(bool)
    return d


def step_summary(s):
    """Vectorized event counts, distinct physical straws and deposited energy, without per-event loops."""
    counts = s.groupby(KEY).agg(steps=("sid", "size"), planes=("plane", "nunique"), eion=("eion", "sum"))
    distinct = s.drop_duplicates(KEY + ["plane", "panel", "straw"]).groupby(KEY).size().rename("straws")
    return counts.join(distinct).reset_index()


def time_cut(g4, final, margin_ns=1.0):
    """Diagnostic G4 truncation by the paired fast exit clock; retain non-exiting/missing-final tracks."""
    g4 = integer_keys(g4, "G4 steps")
    final = unique(final, "entry final")
    stop = final[KEY + ["t", "status"]].rename(columns={"t": "fast_exit_t", "status": "fast_status"})
    s = g4.merge(stop, on=KEY, how="left", validate="many_to_one")
    keep = (s.fast_status != 0) | (s.t <= s.fast_exit_t + margin_ns)
    return s.loc[keep, g4.columns].copy()


def state_summary(s, prefix):
    s = unique(s, prefix)
    p = s[["px", "py", "pz"]].to_numpy(float)
    pmag = np.linalg.norm(p, axis=1)
    data = s[KEY].copy()
    data[f"{prefix}_p"] = pmag
    data[f"{prefix}_costheta"] = p[:, 2] / pmag
    data[f"{prefix}_r"] = np.hypot(s.x + 3904.0, s.y)
    data[f"{prefix}_t"] = s.t
    return data


def describe(d, mask, columns):
    """Fixed-size metric loop over vectorized event selections."""
    out = {"events": int(np.asarray(mask).sum())}
    for col in columns:
        x = d.loc[mask, col].to_numpy(float)
        x = x[np.isfinite(x)]
        out[col] = {"mean_error_n": _me(x), "q10_q50_q90": np.quantile(x, [0.1, 0.5, 0.9]).tolist() if len(x) else []}
    return out


def audit(d):
    """Three-arm decomposition plus explicit state/step-conditioned diagnostics."""
    out = {"events": len(d), "conditional_on_dts_selection": True,
           "state_conditioning": "truth diagnostic only; not a response/correction model",
           "chain_vs_entry": cereco.compare(d, "chain", "entry"), "yield_partition": {}, "chain_entry_categories": {}}
    delta = d.chain_track.astype(int) - d.g4_track.astype(int)
    total = 0
    for name, mask in (("no_chain_vd13", ~d.chain_has_vd13),
                       ("vd13_no_steps", d.chain_has_vd13 & (d.chain_steps == 0)),
                       ("has_steps", d.chain_steps > 0)):
        signed = int(delta[mask].sum())
        total += signed
        out["yield_partition"][name] = {"events": int(mask.sum()), "g4_tracks": int(d.g4_track[mask].sum()),
                                        "chain_tracks": int(d.chain_track[mask].sum()), "signed_track_difference": signed,
                                        "contribution_pp": 100.0 * signed / len(d)}
    if total != int(delta.sum()):
        raise ValueError("yield partition is not exhaustive")
    cols = [k for k in d if k.startswith("delta_")] + ["g4_steps", "g4_late_steps", "entry_steps", "chain_steps",
                                                                  "entry_planes", "chain_planes"]
    for label, mask in (("lost", d.entry_track & ~d.chain_track), ("gained", ~d.entry_track & d.chain_track),
                         ("both", d.entry_track & d.chain_track), ("neither", ~d.entry_track & ~d.chain_track)):
        out["chain_entry_categories"][label] = describe(d, mask, cols)
    out["return_clock_diagnostic"] = {"definition": "G4 steps later than entry-fast exit + 1 ns, only status 0; not a physical G4 passage boundary",
                                      "late_steps": int(d.g4_late_steps.sum()), "events": int((d.g4_late_steps > 0).sum()),
                                      "reco_comparison_by_late_steps": {}}
    for label, mask in (("none", d.g4_late_steps == 0), ("present", d.g4_late_steps > 0)):
        out["return_clock_diagnostic"]["reco_comparison_by_late_steps"][label] = cereco.compare(d.loc[mask], "entry", "g4")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry-events", required=True)
    ap.add_argument("--chain-events", required=True)
    ap.add_argument("--entry-dir", required=True, help="fieldce entry output directory")
    ap.add_argument("--chain-dir", required=True, help="fieldce chain output directory")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cut-events", help="optional cereco saved events with cut and identical G4 arms")
    a = ap.parse_args(argv)
    d = join_events(pd.read_csv(a.entry_events), pd.read_csv(a.chain_events))
    dirs = {"entry": Path(a.entry_dir), "chain": Path(a.chain_dir)}
    states = {arm: tables.read(path / "states.root", "entries") for arm, path in dirs.items()}
    for arm, path in dirs.items():
        gen = unique(tables.read(path / "states.root", "gen", KEY), f"{arm} denominator")
        pd.testing.assert_frame_equal(unique(d[KEY], "events"), gen, check_dtype=False, check_exact=True)
        st = state_summary(states[arm], arm)
        st[f"{arm}_has_vd13"] = True
        d = d.merge(st, on=KEY, how="left", validate="one_to_one")
        d[f"{arm}_has_vd13"] = d[f"{arm}_has_vd13"].fillna(False).astype(bool)
    reference = tables.read(dirs["entry"] / "g4prim_field.root", "sgsteps")
    final = tables.read(dirs["entry"] / "states.root", "final")
    cut = time_cut(reference, final)
    for arm, steps in (("g4", reference), ("g4cut", cut),
                        ("entry", tables.read(dirs["entry"] / "fasttrk_field.root", "sgsteps")),
                        ("chain", tables.read(dirs["chain"] / "fasttrk_field.root", "sgsteps"))):
        cols = unique(step_summary(steps), f"{arm} step summary")
        names = {k: f"{arm}_{k}" for k in cols if k not in KEY}
        d = d.merge(cols.rename(columns=names), on=KEY, how="left", validate="one_to_one")
        d[list(names.values())] = d[list(names.values())].fillna(0)
    d["g4_late_steps"] = d.g4_steps - d.g4cut_steps
    for col in ("p", "r", "costheta", "t"):
        d[f"delta_{col}"] = d[f"chain_{col}"] - d[f"entry_{col}"]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    report = audit(d)
    if a.cut_events:
        c = unique(pd.read_csv(a.cut_events), "time-cut events")
        pd.testing.assert_frame_equal(d[KEY], c[KEY], check_dtype=False, check_exact=True)
        baseline = [k for k in c if k.startswith("g4_")]
        b = c[baseline].copy()
        for col in ("g4_track", "g4_surface"):
            b[col] = b[col].fillna(False).astype(bool)
        pd.testing.assert_frame_equal(d[baseline], b)
        names = [k for k in c if k.startswith("cut_")]
        d = d.merge(c[KEY + names], on=KEY, how="left", validate="one_to_one")
        report["entry_vs_time_cut"] = cereco.compare(d, "entry", "cut")
        no_late = d.g4_late_steps == 0
        report["time_cut_vs_g4_unmodified_step_events"] = cereco.compare(d.loc[no_late], "cut", "g4")
    report["reference_integrity"] = "identical event keys and saved G4 columns in entry/chain comparisons"
    cut["nsgs"] = cut.groupby(KEY).sid.transform("size")
    fieldce.injector(out / "g4cut_field.root", cut, tables.read(dirs["entry"] / "states.root", "g4entries"))
    d.to_csv(out / "events.csv.gz", index=False)
    (out / "ceaudit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
