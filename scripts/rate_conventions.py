"""Muon-counting conventions of the paired tracker-term rate: any primary muon (the PrimaryFilter) against one muon.

    python3 tracker/scripts/rate_conventions.py --g4 G4.parquet [--arm NAME=DIR ...] [--seed-csv EVENTS.csv.gz ...]
            [--variant plane] [--out REPORT.json]

G4.parquet and DIR/muons.*.parquet come from trkfastsim.analyses.ratearms (g4, fast). Per arm: events passing through
any primary muon, through muon key 1, and only through a key > 1 (CRY multi-muon events). G4 keys are mapped to the
fast muon key (rank among the generated muons) with ratearms.g4_fast_keys through the first --arm; without --arm the G4
SimParticle key (index among all generated particles + 1) is used as is. fatecheck (fast key 1 against G4 any muon) and seedcheck (one seeded muon against G4 any muon) count
G4 passes through a second muon that the fast arm cannot have; this script gives that offset.

--fatecheck: fatecheck_events.csv(.gz) of an earlier fatecheck run on the same primaries (fast key 1, earlier code):
its pass set against the first --arm (key 1 and any muon) and its G4 pass set against the G4 table.

--seed-csv: seedcheck_events.csv(.gz) of the seedcheck batches (variant `--variant`, CRV subset): G4 any-muon passes,
passes of the seeded muon (the lowest stored primary-muon key, seedcheck._primaries), fast seeded passes, and the
seeded relative difference against both, with the paired error sqrt(only_a + only_b) / n_ref.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["run", "subrun", "event"]
MIN = 12


def passes(mu):
    """Event-level pass sets from per-muon rows (KEY, sp, ngood)."""
    ok = mu[mu.ngood >= MIN]
    anyp = ok[KEY].drop_duplicates()
    sp1 = ok[ok.sp == 1][KEY].drop_duplicates()
    low = mu.sort_values(KEY + ["sp"]).drop_duplicates(KEY)                 # lowest stored key per event
    lowp = low[low.ngood >= MIN][KEY]
    n_any, n_sp1 = len(anyp), len(sp1)
    return {"any": n_any, "sp1": n_sp1, "only_sp_gt1": n_any - n_sp1, "lowest_stored_key": int(len(lowp)),
            "events": int(len(mu[KEY].drop_duplicates())), "muons": int(len(mu))}, anyp, lowp


def paired(a, b):
    na, nb, oa, ob = int(a.sum()), int(b.sum()), int((a & ~b).sum()), int((b & ~a).sum())
    return {"ref": nb, "arm": na, "arm_only": oa, "ref_only": ob, "rel": na / nb - 1 if nb else None,
            "err": np.sqrt(oa + ob) / nb if nb else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g4", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--seed-csv", nargs="*", default=[])
    ap.add_argument("--fatecheck", nargs="*", default=[])
    ap.add_argument("--variant", default="plane")
    ap.add_argument("--out", default=None)
    x = ap.parse_args(argv)
    g = pd.read_parquet(x.g4, columns=KEY + ["sp", "ngood"])
    rep = {"g4": str(x.g4)}
    if x.arm:
        from trkfastsim.analyses.ratearms import g4_fast_keys
        d0 = x.arm[0].split("=", 1)[1]
        fm = pd.concat([pd.read_parquet(f, columns=KEY + ["sp", "genidx"]) for f in sorted(glob.glob(f"{d0}/muons.*.parquet"))])
        g, rep["g4_unmapped"] = g4_fast_keys(g, fm)
        if rep["g4_unmapped"]:
            raise ValueError("G4 primary identities are missing from the fast arm; comparison is incomplete")
        g = g[KEY + ["sp", "ngood"]]
    rep["g4_counts"], g_any, g_low = passes(g)
    arm_sets = {}
    print(f"G4      {rep['g4_counts']}")
    for spec in x.arm:
        name, d = spec.split("=", 1)
        m = pd.concat([pd.read_parquet(f, columns=KEY + ["sp", "ngood"]) for f in sorted(glob.glob(f"{d}/muons.*.parquet"))])
        c, f_any, _ = passes(m)
        ev = m[KEY].drop_duplicates()
        flag = lambda s: ev.merge(s.assign(_x=True), on=KEY, how="left")._x.fillna(False).to_numpy(bool)
        f_sp1 = m[(m.ngood >= MIN) & (m.sp == 1)][KEY].drop_duplicates()
        c["any_vs_g4_any"] = paired(flag(f_any), flag(g_any))
        c["sp1_vs_g4_any (fatecheck)"] = paired(flag(f_sp1), flag(g_any))
        c["sp1_vs_g4_sp1"] = paired(flag(f_sp1), flag(g[(g.ngood >= MIN) & (g.sp == 1)][KEY].drop_duplicates()))
        rep[name] = c
        arm_sets.setdefault("first", (name, ev, f_any, f_sp1))
        print(f"{name:7s} {c}")
    if x.fatecheck:
        h = pd.concat([pd.read_csv(f, usecols=KEY + ["fast_pass", "g4_pass"]) for f in x.fatecheck], ignore_index=True)
        h = h.groupby(KEY, as_index=False)[["fast_pass", "g4_pass"]].max()
        name, ev, f_any, f_sp1 = arm_sets["first"]
        j = h.merge(ev, on=KEY, how="outer")
        fl = lambda s_: j[KEY].merge(s_.assign(_x=True), on=KEY, how="left")._x.fillna(False).to_numpy(bool)
        hf, hg = j.fast_pass.fillna(False).astype(bool).to_numpy(), j.g4_pass.fillna(False).astype(bool).to_numpy()
        rep["fatecheck"] = {"files": [str(f) for f in x.fatecheck], "events": int(len(h)),
                            "historical_fast_vs_g4_any": paired(hf, fl(g_any)),
                            "historical_g4_equals_g4_any": bool((hg == fl(g_any)).all()),
                            f"{name}_sp1_vs_historical_fast": paired(fl(f_sp1), hf),
                            f"{name}_any_vs_historical_fast": paired(fl(f_any), hf)}
        print(json.dumps(rep["fatecheck"], indent=1, default=str))
    if x.seed_csv:
        s = pd.concat([pd.read_csv(f) for f in x.seed_csv], ignore_index=True)
        s = s[(s.variant == x.variant) & s.crv_pass.astype(bool)].drop_duplicates(KEY)
        s = s.merge(g_low.assign(g4_seedmu=True), on=KEY, how="left")
        s["g4_seedmu"] = s.g4_seedmu.fillna(False).astype(bool)
        a, b, c0 = s.seed_pass.astype(bool).to_numpy(), s.g4_pass.astype(bool).to_numpy(), s.g4_seedmu.to_numpy()
        rep["seedcheck"] = {"variant": x.variant, "files": [str(f) for f in x.seed_csv], "seeds": int(len(s)),
                            "seeded_vs_g4_any": paired(a, b), "seeded_vs_g4_seeded_muon": paired(a, c0),
                            "g4_any_not_seeded_muon": int((b & ~c0).sum()), "g4_seeded_muon_not_any": int((c0 & ~b).sum())}
        edges = [0, 300, 500, 1000, 2000, 5000, 1e9]
        k = np.clip(np.digitize(s.p_seed.to_numpy(), edges) - 1, 0, len(edges) - 2)
        rep["seedcheck"]["by_p_seed"] = [{"bin": [edges[i], edges[i + 1]], "vs_g4_any": paired(a[k == i], b[k == i]),
                                          "vs_g4_seeded_muon": paired(a[k == i], c0[k == i])} for i in range(len(edges) - 1)]
        print(json.dumps(rep["seedcheck"], indent=1, default=str))
    if x.out:
        out = Path(x.out)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        json.dump(rep, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
