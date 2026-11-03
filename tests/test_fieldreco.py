"""validation.fieldreco: paired arm statistics on a synthetic per-event frame (no ntuples)."""
import numpy as np
import pandas as pd

from trkfastsim.validation import fieldreco as fr


def _frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"run": 1, "subrun": 0, "event": np.arange(n), "p_first": rng.uniform(200, 8000, n)})
    for arm, eff, dn in (("g4", 0.5, 0.0), ("fast", 0.45, -1.0)):
        has = rng.random(n) < eff
        d[f"{arm}_ntrk"] = np.where(has, 1, np.nan)
        d[f"{arm}_nactive"] = np.where(has, 40 + dn + rng.normal(0, 3, n), np.nan)
        d[f"{arm}_nhits"] = d[f"{arm}_nactive"] + 2
        d[f"{arm}_chi2ndf"] = np.where(has, 1.0, np.nan)
        d[f"{arm}_fitcon"] = np.where(has, rng.random(n), np.nan)
        d[f"{arm}_sid"] = np.where(has, 0, -1)
        d[f"{arm}_p_reco"] = np.where(has, d.p_first * 1.01, np.nan)
        d[f"{arm}_p_true"] = np.nan
    return d


def test_paired_yield_and_hits():
    d = _frame()
    c = fr.compare(d, "g4", "fast")
    ia, ir = d.fast_ntrk.notna(), d.g4_ntrk.notna()
    assert np.isclose(c["yield"]["diff"][0], ia.mean() - ir.mean())
    assert c["yield"]["gained"] == int((ia & ~ir).sum()) and c["yield"]["lost"] == int((~ia & ir).sum())
    both = ia & ir
    assert c["nactive"]["diff"][2] == int(both.sum())
    assert abs(c["nactive"]["diff"][0] + 1.0) < 5 * c["nactive"]["diff"][1]
    r = c["p_reco_same_surface"]
    assert r["truth"]["fast"] == "reference first straw" and abs(r["residual_MeV"]["diff"][0]) < 1e-9
    assert sum(b["events"] for b in c["by_true_p"]) <= len(d)


def test_self_pass_windows():
    from trkfastsim.validation import fieldcompare as fc
    n = 30
    d = pd.DataFrame({"run": 1, "subrun": 0, "event": np.repeat([1, 2, 3], n), "sp": 1, "plane": np.tile(np.arange(n), 3),
                      "px": 0.0, "py": 0.0, "pz": np.r_[np.full(n, 100.0), np.full(n, 5000.0), np.full(n, 500.0)]})
    s = fc.self_pass(d).set_index("event")
    assert s.loc[1, "signal"] and s.loc[2, "calib"] and not s.loc[3, "self"]
