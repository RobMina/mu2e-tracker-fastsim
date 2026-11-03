"""field-reco -- stock CentralHelix reco (KKCHmu) on injected field-on steps: arms compared event by event.

Inputs: `NAME=DIR` per arm, DIR holding the arm's EventNtuple files `nts.<stem>.root`, one per dts file (stock
digitisation + recoMC/OffSpill + EventNtuple on the same dts events, each arm a different StrawGasStep source). The
denominator is the events of the dts files every arm has (from the per-file dumper tables, `--file-tables`), restricted
to the events of `--events` (a primary-only injector table) when given, so an event without CentralHelix output counts
as no track.

Per event, the first good track (status > 0): active hits, hits, chi2/ndof, fit consistency, and the reco momentum at
its earliest in-bounds tracker crossing (trksegs, SurfaceIds 0-4). Truth: |p| at the first straw of the `--events`
table (the reference G4 steps; injected events carry no SurfaceSteps), else trksegsmc at the same
surface and index nearest in time. Compared to the reference arm (`--ref`), paired over the denominator: track yield
(difference of per-event indicators), and over events with a track in both arms: hits, chi2/ndof, fit consistency, the
reco momentum (same surface) and the momentum residual reco - truth; per truth-momentum bin. Errors: standard error of
the mean of the paired per-event differences; distributions also as unpaired binned chi2 (Poisson errors).
"""
import glob
import json
import os
import re

import awkward as ak
import numpy as np
import pandas as pd
import uproot

TREE = "EventNtuple/ntuple"
KEY = ["run", "subrun", "event"]
TRK = ["status", "nactive", "nhits", "ndof", "chisq", "fitcon"]
SIDS = (0, 1, 2, 3, 4)
P_EDGES = np.array([0.0, 300.0, 1000.0, 2000.0, 5000.0, 1e9])


def _files(d):
    """{file stem (e.g. 001430_00000000): path} of the nts.<stem>.root files in d."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "nts.*.root"))):
        m = re.search(r"nts\.(\w+)\.root$", f)
        if m:
            out[m.group(1)] = f
    return out


def _vec(r):
    c = r.fCoordinates
    return np.stack([ak.to_numpy(c.fX), ak.to_numpy(c.fY), ak.to_numpy(c.fZ)], 1).astype(float)


def read_arm(paths):
    """Per event of the ntuples: KEY, ntrk and the first good track's quantities (NaN without a track)."""
    parts = []
    for path in paths:
        t = uproot.open(path)[TREE]
        a = t.arrays(["evtinfo/run", "evtinfo/subrun", "evtinfo/event"] + [f"trk/trk.{k}" for k in TRK], library="ak")
        good = a["trk/trk.status"] > 0
        ntrk = ak.to_numpy(ak.sum(good, axis=1))
        has = ntrk > 0
        first = ak.argmax(good, axis=1, keepdims=True)
        df = pd.DataFrame({"run": ak.to_numpy(a["evtinfo/run"]), "subrun": ak.to_numpy(a["evtinfo/subrun"]),
                           "event": ak.to_numpy(a["evtinfo/event"]), "ntrk": ntrk})
        for k in TRK[1:]:
            v = ak.to_numpy(ak.fill_none(ak.firsts(a[f"trk/trk.{k}"][first]), 0)).astype(float)
            df[k] = np.where(has, v, np.nan)
        df["chi2ndf"] = df.chisq / np.maximum(df.ndof, 1)
        segs = ak.firsts(t["trksegs"].array()[first], axis=1)
        mc = ak.firsts(t["trksegsmc"].array()[first], axis=1)
        segs = ak.fill_none(segs, [], axis=0)
        mc = ak.fill_none(mc, [], axis=0)
        ok = (segs.sid >= 0) & (segs.sid <= 4) & segs.inbounds
        s = segs[ok]
        i0 = ak.argmin(s.time, axis=1, keepdims=True)
        s0 = ak.firsts(s[i0])
        hit = ~ak.is_none(s0)
        sid = ak.to_numpy(ak.fill_none(s0.sid, -1))
        sindex = ak.fill_none(s0.sindex, -1)
        rt = ak.fill_none(s0.time, np.nan)
        pr = np.full(len(df), np.nan)
        if ak.any(hit):
            pv = _vec(ak.fill_none(s0.mom, {"fCoordinates": {"fX": np.nan, "fY": np.nan, "fZ": np.nan}}))
            pr = np.linalg.norm(pv, axis=1)
        cand = mc[(mc.sid == ak.Array(sid)) & (mc.sindex == sindex)]
        j = ak.argmin(abs(cand.time - rt), axis=1, keepdims=True)
        c0 = ak.firsts(cand[j])
        pt = np.full(len(df), np.nan)
        if ak.any(~ak.is_none(c0)):
            tv = _vec(ak.fill_none(c0.mom, {"fCoordinates": {"fX": np.nan, "fY": np.nan, "fZ": np.nan}}))
            pt = np.linalg.norm(tv, axis=1)
        df["sid"] = np.where(has, sid, -1)
        df["p_reco"] = np.where(has, pr, np.nan)
        df["p_true"] = np.where(has, pt, np.nan)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ["ntrk"])


HIT_FIELDS = ["plane", "panel", "straw", "state", "edep", "wdist", "tottdrift", "rdrift", "udoca", "rdresid", "utresid"]


def read_hits(paths):
    """trkhits of the first good track per event, flattened: KEY + HIT_FIELDS + etime0, tot0."""
    parts = []
    for path in paths:
        t = uproot.open(path)[TREE]
        a = t.arrays(["evtinfo/run", "evtinfo/subrun", "evtinfo/event", "trk/trk.status"], library="ak")
        good = a["trk/trk.status"] > 0
        has = ak.to_numpy(ak.any(good, axis=1))
        first = ak.argmax(good, axis=1, keepdims=True)
        h = ak.firsts(t["trkhits"].array()[first], axis=1)[has]
        n = ak.to_numpy(ak.num(h))
        cols = {k: ak.to_numpy(ak.flatten(h[k])) for k in HIT_FIELDS}
        cols["etime0"] = ak.to_numpy(ak.flatten(h.etime[:, :, 0]))
        cols["tot0"] = ak.to_numpy(ak.flatten(h.tot[:, :, 0]))
        df = pd.DataFrame(cols)
        for k in KEY:
            df[k] = np.repeat(ak.to_numpy(a[f"evtinfo/{k}"])[has], n)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + HIT_FIELDS)


def hit_compare(ha, hr, arm, ref, events):
    """Hits of the same straw on the first tracks of both arms (events with a track in both)."""
    k = KEY + ["plane", "panel", "straw"]
    ha = ha.merge(events, on=KEY).drop_duplicates(k)
    hr = hr.merge(events, on=KEY).drop_duplicates(k)
    m = ha.merge(hr, on=k, suffixes=("_a", "_r"))
    out = {"hits": {arm: int(len(ha)), ref: int(len(hr)), "matched": int(len(m))}}
    act_a, act_r = ha.state > 0, hr.state > 0
    out["active_fraction"] = {arm: _me(act_a), ref: _me(act_r)}
    for c in ("edep", "tot0", "etime0", "wdist", "tottdrift", "rdrift"):
        out[c] = {"diff": _robust(m[f"{c}_a"] - m[f"{c}_r"]), arm: _me(m[f"{c}_a"]), ref: _me(m[f"{c}_r"])}
    for c in ("rdresid", "utresid", "udoca"):
        out[c] = {arm: _robust(ha[c][act_a]), ref: _robust(hr[c][act_r])}
    return out


def _me(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan"), float("nan"), int(len(x))]
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def _chi2(a, b, edges):
    a, b = (np.asarray(v, float) for v in (a, b))
    ha, _ = np.histogram(a[np.isfinite(a)], edges)
    hb, _ = np.histogram(b[np.isfinite(b)], edges)
    na, nb = ha.sum(), hb.sum()
    if not na or not nb:
        return {"chi2": float("nan"), "ndf": 0}
    var = ha / na ** 2 + hb / nb ** 2
    ok = var > 0
    return {"chi2": float(((ha / na - hb / nb)[ok] ** 2 / var[ok]).sum()), "ndf": int(ok.sum())}


def _robust(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {}
    q = np.quantile(x, [0.16, 0.25, 0.5, 0.75, 0.84])
    return {"median": float(q[2]), "iqr_sigma": float((q[3] - q[1]) / 1.349), "q16_q84": [float(q[0]), float(q[4])]}


def compare(d, ref, arm):
    """Paired summary of `arm` against `ref` on the merged per-event frame d (columns <name>_<field>)."""
    out = {}
    ia, ir = (d[f"{n}_ntrk"].fillna(0).to_numpy() > 0 for n in (arm, ref))
    out["yield"] = {arm: _me(ia), ref: _me(ir), "diff": _me(ia.astype(float) - ir),
                    "ratio": float(ia.sum() / max(ir.sum(), 1)), "gained": int((ia & ~ir).sum()),
                    "lost": int((~ia & ir).sum())}
    both = d[ia & ir]
    for k in ("nactive", "nhits", "chi2ndf", "fitcon"):
        out[k] = {arm: _me(both[f"{arm}_{k}"]), ref: _me(both[f"{ref}_{k}"]),
                  "diff": _me(both[f"{arm}_{k}"] - both[f"{ref}_{k}"])}
    out["nactive"].update(_chi2(both[f"{arm}_nactive"], both[f"{ref}_nactive"], np.arange(-0.5, 120.5, 2.0)))
    out["fitcon"].update(_chi2(both[f"{arm}_fitcon"], both[f"{ref}_fitcon"], np.linspace(0, 1, 21)))
    out["fitcon_below_1e-3"] = {arm: _me(both[f"{arm}_fitcon"] < 1e-3), ref: _me(both[f"{ref}_fitcon"] < 1e-3)}
    same = both[both[f"{arm}_sid"] == both[f"{ref}_sid"]]
    # truth: the reference |p| at the first straw of the injector table (same for every arm), else trksegsmc per arm
    # truth per arm: that arm's own |p| at its first straw (--arm-tables), else the reference table's, else trksegsmc
    def truth(n):
        for c in (f"{n}_p_first_own", "p_first", f"{n}_p_true"):
            if c in same:
                return same[c]
    ta, tr = truth(arm), truth(ref)
    ra = same[f"{arm}_p_reco"] - ta
    rr = same[f"{ref}_p_reco"] - tr
    out["p_reco_same_surface"] = {"events": int(len(same)), "diff_MeV": _me(same[f"{arm}_p_reco"] - same[f"{ref}_p_reco"]),
                                  "residual_MeV": {arm: _me(ra), ref: _me(rr), "diff": _me(ra - rr)},
                                  "residual_robust": {arm: _robust(ra), ref: _robust(rr)},
                                  "rel_residual_robust": {arm: _robust(ra / ta), ref: _robust(rr / tr),
                                                          "diff": _robust(ra / ta - rr / tr)},
                                  "rel_residual_within_10pc": {arm: _me(np.abs(ra / ta) < 0.1),
                                                               ref: _me(np.abs(rr / tr) < 0.1),
                                                               "diff": _me((np.abs(ra / ta) < 0.1).astype(float)
                                                                           - (np.abs(rr / tr) < 0.1))},
                                  "truth": {n: ("own first straw" if f"{n}_p_first_own" in same else
                                                "reference first straw" if "p_first" in same else "trksegsmc")
                                            for n in (arm, ref)}}
    pt = d["p_first"].to_numpy() if "p_first" in d else d[f"{ref}_p_true"].where(ir, d[f"{arm}_p_true"]).to_numpy()
    bins = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        k = (pt >= lo) & (pt < hi)
        if (k & (ia | ir)).sum() < 5:
            continue
        kb = k[ia & ir]
        sel = np.isin(same.index, both.index[kb])
        bins.append({"p_lo": float(lo), "p_hi": float(hi), "events": int(k.sum()), "events_with_a_track": int((k & (ia | ir)).sum()),
                     "yield_diff": _me((ia.astype(float) - ir)[k]),
                     "nactive_diff": _me((both[f"{arm}_nactive"] - both[f"{ref}_nactive"])[kb]),
                     "residual_diff_MeV": _me((ra - rr)[sel]),
                     "rel_residual_robust": {arm: _robust((ra / ta)[sel]), ref: _robust((rr / tr)[sel])}})
    out["by_true_p"] = bins
    if "first_plane" in d:
        fp = d.first_plane.to_numpy()
        cls = {"end_planes": (fp <= 1) | (fp >= 34), "inner_planes": (fp > 1) & (fp < 34)}
        out["by_first_plane"] = {}
        for name, k in cls.items():
            sel = np.isin(same.index, d.index[k])
            out["by_first_plane"][name] = {"events": int(k.sum()), "yield_diff": _me((ia.astype(float) - ir)[k]),
                                           "rel_residual_robust": {arm: _robust((ra / ta)[sel]), ref: _robust((rr / tr)[sel])},
                                           "p_reco_rel_diff_robust": _robust((ra / ta - rr / tr)[sel])}
    return out


def run(inputs, outdir, *, events=None, file_tables=None, ref="g4", self_pass=False, unbiased=False, arm_tables=None,
        **kw):
    """inputs: NAME=DIR per arm. Denominator: the events of the per-file dumper tables `file_tables`/sgsteps.<stem>.root
    (dump/particles keys) of the stems every arm has (a dts file spans several subruns), intersected with the events of
    the injector table `events` (dump/sgsteps keys) when given."""
    os.makedirs(outdir, exist_ok=True)
    arms = dict(s.split("=", 1) for s in inputs)
    if ref not in arms:
        raise ValueError(f"reference arm {ref!r} not among {sorted(arms)}")
    if not file_tables:
        raise ValueError("file_tables is required (the per-file event lists)")
    files = {n: _files(d) for n, d in arms.items()}
    common = sorted(set.intersection(*(set(f) for f in files.values())))
    if not common:
        raise ValueError("no file common to every arm")
    den = pd.concat([uproot.open(os.path.join(file_tables, f"sgsteps.{s}.root"))["dump/particles"]
                     .arrays(KEY, library="pd").drop_duplicates() for s in common]).drop_duplicates()
    n_file_events = int(len(den))
    if events:
        inj = uproot.open(events)["dump/sgsteps"].arrays(KEY + ["t", "px", "py", "pz", "plane"], library="pd")
        inj = inj.sort_values(KEY + ["t"]).groupby(KEY, sort=False).first().reset_index()
        inj["p_first"] = np.sqrt(inj.px ** 2 + inj.py ** 2 + inj.pz ** 2)
        # entry class from the first straw's plane: an end plane pair (0-1, 34-35) or the side (through the support)
        inj["first_plane"] = inj.plane
        den = den.merge(inj[KEY + ["p_first", "first_plane"]], on=KEY)
        if self_pass:
            from trkfastsim.validation import fieldcompare as fc
            st_ = uproot.open(events)["dump/sgsteps"].arrays(KEY + ["sp", "plane", "px", "py", "pz"], library="pd")
            sp_ = fc.self_pass(st_)
            ok = sp_.groupby(KEY).self.any().reset_index()
            den = den.merge(ok[ok.self][KEY], on=KEY)
    if unbiased:
        from trkfastsim.validation import fieldcompare as fc
        paths = [os.path.join(file_tables, f"sgsteps.{s}.root") for s in common]
        # primaries of the reference sample: the dumper's MC primary and its same-species continuation
        part, _, _, _, _ = fc.primaries(paths, 13)
        bad = fc.secondary_selected(paths, part)
        den = den.merge(bad.assign(_b=1), on=KEY, how="left")
        den = den[den._b.isna()].drop(columns="_b")
    den = den.sort_values(KEY).reset_index(drop=True)
    for spec in arm_tables or []:
        n, path = spec.split("=", 1)
        a = uproot.open(path)["dump/sgsteps"].arrays(KEY + ["t", "px", "py", "pz"], library="pd")
        a = a.sort_values(KEY + ["t"]).groupby(KEY, sort=False).first().reset_index()
        a[f"{n}_p_first_own"] = np.sqrt(a.px ** 2 + a.py ** 2 + a.pz ** 2)
        den = den.merge(a[KEY + [f"{n}_p_first_own"]], on=KEY, how="left")
    frames = {n: read_arm([files[n][s] for s in common]) for n in arms}
    d = den.copy()
    for n, f in frames.items():
        f = f.drop_duplicates(KEY)
        d = d.merge(f.rename(columns={c: f"{n}_{c}" for c in f.columns if c not in KEY}), on=KEY, how="left")
    res = {"arms": {n: os.path.basename(os.path.normpath(p)) for n, p in arms.items()}, "ref": ref,
           "files": len(common), "file_events": n_file_events, "events": int(len(d)), "self_pass": bool(self_pass), "unbiased": bool(unbiased),
           "ntuple_events": {n: int(len(f)) for n, f in frames.items()},
           "ntuple_events_outside_denominator": {n: int(len(f.merge(den, on=KEY, how="left", indicator=True)
                                                           .query("_merge == 'left_only'"))) for n, f in frames.items()},
           "compare": {n: compare(d, ref, n) for n in arms if n != ref}}
    if kw.get("hits"):
        hits = {n: read_hits([files[n][s] for s in common]) for n in arms}
        res["hits"] = {}
        for n in arms:
            if n == ref:
                continue
            ev = d[(d[f"{n}_ntrk"].fillna(0) > 0) & (d[f"{ref}_ntrk"].fillna(0) > 0)][KEY]
            res["hits"][n] = hit_compare(hits[n], hits[ref], n, ref, ev)
    if kw.get("dump_events"):
        d.to_csv(os.path.join(outdir, "events.csv.gz"), index=False)
    with open(os.path.join(outdir, "fieldreco.json"), "w") as fo:
        json.dump(res, fo, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res
