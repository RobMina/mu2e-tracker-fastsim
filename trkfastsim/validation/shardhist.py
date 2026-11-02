"""shardhist -- per-shard histograms of one arm for the paired per-shard statistic (`shardstats`).

A shard is a subrun: arms made from the same generated primaries share their subruns, and subruns are independent, so
per-shard differences carry the event clustering and the pairing of the arms. --batches K splits each subrun into K
shards by event number modulo K (for samples with few subruns; events are independent, the pairing is kept). Every histogram keeps its underflow and
overflow bins. Output: <outdir>/<arm>.npz with h|<arm>|<key> (shards x (bins + 2), int64), edges|<key> and
shards|<arm>; a small <arm>.json with counts and settings.

Kinds:
- steps: StrawGasStep tables (fast or Geant4). Events pass the PrimaryFilter tracker term for any primary muon
  (isprim == 1, |pdg| == 13, >= 12 good steps); every step of a selected event enters. Keys: sel/events; ev/* per event
  (primary-muon steps, other steps, eIon, occupied straws); mu/* per primary-muon step (eIon, path, |p|, time, plane,
  panel, straw); sec/* per other step (eIon, path, species, place relative to the muon, plane distance); straw/* per
  occupied straw (summed eIon of all, muon and secondary-only straws, steps per straw, straw occupancy).
- digis: StrawHitTableDumper or fast digi tables, restricted to the events selected on --select-steps (events absent
  from the digi tables count as events with no digi). Keys: the digicmp histograms (digi/*; hit/* from the fast hit
  reco on these digis) and sel/events. --hit-cond defaults to the conditions of --drift-source or inputs[0]. --stock-hits adds an arm <arm>@stockreco with the stored stock straw and combo
  hits under the same hit/* keys (stock dumper tables only). --f2 adds fit/* from the fast reco chain per shard
  (fits per event, active hits, chi2/ndf of good fits); --kkline NTUPLES adds an arm <arm>@kkline with fit/* from the
  stock KKLine fits of those events (status > 0).
"""
import json
import os

import numpy as np
import pandas as pd

from trkfastsim import tables as T

KEV_STEP = np.linspace(0.0, 10.0, 101)
KEV_STRAW = np.linspace(0.0, 10.0, 101)
LEN = np.linspace(0.0, 20.0, 81)
PMOM = np.logspace(np.log10(50.0), 5.0, 61)
TIME = np.linspace(0.0, 100000.0, 201)
EDGES = {
    "sel/events": np.array([-0.5, 0.5]),
    "ev/mu_steps": np.arange(0, 101) - 0.5, "ev/sec_steps": np.arange(0, 201) - 0.5,
    "ev/eion_keV": np.linspace(0.0, 300.0, 151), "ev/straws": np.arange(0, 151) - 0.5,
    "mu/eion_keV": KEV_STEP, "mu/len_mm": LEN, "mu/p_MeV": PMOM, "mu/t_ns": TIME,
    "mu/plane": np.arange(37) - 0.5, "mu/upanel": np.arange(217) - 0.5, "mu/straw": np.arange(97) - 0.5,
    "sec/eion_keV": KEV_STEP, "sec/len_mm": LEN, "sec/species": np.arange(5) - 0.5,
    "sec/place": np.arange(6) - 0.5, "sec/dplane": np.arange(37) - 0.5,
    "straw/eion_keV": KEV_STRAW, "straw/eion_mu_keV": KEV_STRAW, "straw/eion_seconly_keV": KEV_STRAW,
    "straw/eion_muonly_keV": KEV_STRAW, "straw/nsteps": np.arange(1, 22) - 0.5,
    "straw/occupancy": np.arange(216 * 96 + 1) - 0.5,
    "fit/fits_per_event": np.arange(0, 7) - 0.5, "fit/nactive": np.arange(0, 81) - 0.5,
    "fit/chi2ndf": np.linspace(0.0, 10.0, 101),
}
SPECIES = {11: 0, -11: 1, 13: 2, -13: 2}   # e-, e+, non-primary muon; 3 = other
PLACE = ("in muon straw", "same plane", "1-2 planes", ">= 3 planes", "no primary-muon step")


class Hists:
    """Per-shard histograms with flows: fill(key, shard index, values) bins with the key's edges."""

    def __init__(self, shards):
        self.shards = np.asarray(shards, np.int64)
        self.h = {}

    def fill(self, key, sidx, x, edges=None):
        e = EDGES[key] if edges is None else edges
        nb = len(e) + 1                                    # bins + underflow + overflow
        x = np.asarray(x, np.float64)
        sidx = np.asarray(sidx, np.int64)
        ok = np.isfinite(x)
        b = np.searchsorted(e, x[ok], side="right")       # 0 underflow, len(e) overflow
        c = np.bincount(sidx[ok] * nb + b, minlength=len(self.shards) * nb).reshape(len(self.shards), nb)
        self.h[key] = self.h.get(key, 0) + c

    def save(self, outdir, arm, extra=None):
        os.makedirs(outdir, exist_ok=True)
        z = {f"h|{arm}|{k}": v.astype(np.int64) for k, v in self.h.items()}
        z.update({f"edges|{k}": EDGES[k] for k in self.h})
        z[f"shards|{arm}"] = self.shards
        np.savez_compressed(os.path.join(outdir, f"{arm}.npz"), **z)
        info = {"arm": arm, "shards": int(len(self.shards)), "totals": {k: int(v.sum()) for k, v in self.h.items()}}
        info.update(extra or {})
        with open(os.path.join(outdir, f"{arm}.json"), "w") as f:
            json.dump(info, f, indent=1)
        return info


def _shards_of(paths, batches=1):
    """Sorted shard ids of the step tables' subruns: subrun * batches + j, j < batches."""
    sr = T.read_many(paths, "subruns", ["run", "subrun"])
    if sr.duplicated().any():
        raise ValueError("a subrun appears twice in the step tables")
    s = np.sort(sr.subrun.to_numpy(np.int64))
    return (s[:, None] * batches + np.arange(batches)[None, :]).ravel()


def _sid(subrun, event, batches):
    """Shard id of events: subrun alone, or subrun * batches + event % batches (events are independent)."""
    return np.asarray(subrun, np.int64) * batches + np.asarray(event, np.int64) % batches


def _sidx(shards, sid):
    i = np.searchsorted(shards, sid)
    if np.any(i >= len(shards)) or np.any(shards[np.minimum(i, len(shards) - 1)] != sid):
        raise ValueError("entries outside the shard list")
    return i


def load_selected_steps(path):
    """All steps of the events passing the tracker term, with pdg, isprim and a primary-muon mask."""
    s = T.read(path, "sgsteps")
    p = T.read(path, "particles", T.KEY + ["key", "pdg", "isprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="left", validate="many_to_one")
    if s.pdg.isna().any():
        raise ValueError(f"{path}: steps without particle coverage")
    s["mu"] = (s.isprim == 1) & (s.pdg.abs() == 13)
    f = T.trk_filter(s[s.mu])
    keys = f[f.pass_trk].reset_index()[T.KEY]
    return s.merge(keys, on=T.KEY, how="inner"), keys


def _place(s):
    """Place of every non-primary-muon step: 0 in a straw the muon crossed, 1 same plane, 2 one or two planes away,
    3 farther, 4 event without primary-muon steps; and the minimum plane distance (-1 for place 0)."""
    sec = s[~s.mu]
    mu = s[s.mu]
    ev = pd.MultiIndex.from_frame(s[T.KEY])
    evid, _ = pd.factorize(ev)
    s_ev = pd.Series(evid, index=s.index)
    ms = s_ev[mu.index].to_numpy() * 65536 + mu.sid.to_numpy()
    inmu = np.isin(s_ev[sec.index].to_numpy() * 65536 + sec.sid.to_numpy(), ms)
    # minimum |plane - muon plane| per event: muon planes per event as a bit set over 36 planes
    bits = np.zeros(evid.max() + 1 if len(evid) else 0, np.int64)
    np.bitwise_or.at(bits, s_ev[mu.index].to_numpy(), np.left_shift(1, mu.plane.to_numpy(np.int64)))
    sb = bits[s_ev[sec.index].to_numpy()]
    pl = sec.plane.to_numpy(np.int64)
    dmin = np.full(len(sec), 99, np.int64)
    for d in range(36):                                     # at most 36 vectorised passes
        hit = (((sb >> np.clip(pl - d, 0, 35)) & 1) & (pl - d >= 0)) | (((sb >> np.clip(pl + d, 0, 35)) & 1) & (pl + d <= 35))
        dmin = np.where((dmin == 99) & (hit == 1), d, dmin)
    place = np.where(inmu, 0, np.where(dmin == 0, 1, np.where(dmin <= 2, 2, np.where(dmin < 99, 3, 4))))
    return place, np.where(inmu, -1, dmin)


def steps_arm(paths, arm, outdir, batches=1):
    paths = T.expand(paths)
    shards = _shards_of(paths, batches)
    H = Hists(shards)
    nsel = 0
    for path in paths:
        s, keys = load_selected_steps(path)
        nsel += len(keys)
        H.fill("sel/events", _sidx(shards, _sid(keys.subrun, keys.event, batches)), np.zeros(len(keys)))
        if not len(s):
            continue
        si = _sidx(shards, _sid(s.subrun, s.event, batches))
        s = s.assign(si=si)
        e = 1e3 * s.eion.to_numpy()
        g = s.groupby(T.KEY, sort=False)
        evs = g.agg(si=("si", "first"), nmu=("mu", "sum"), n=("mu", "size"), e=("eion", "sum"))
        evs["straws"] = s.drop_duplicates(T.KEY + ["sid"]).groupby(T.KEY, sort=False).size()
        # selected events without any step cannot occur (the selection needs steps)
        H.fill("ev/mu_steps", evs.si, evs.nmu)
        H.fill("ev/sec_steps", evs.si, evs.n - evs.nmu)
        H.fill("ev/eion_keV", evs.si, 1e3 * evs.e)
        H.fill("ev/straws", evs.si, evs.straws)
        m = s.mu.to_numpy()
        mu = s[m]
        H.fill("mu/eion_keV", mu.si, e[m])
        H.fill("mu/len_mm", mu.si, mu.len)
        H.fill("mu/p_MeV", mu.si, np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2))
        H.fill("mu/t_ns", mu.si, mu.t)
        H.fill("mu/plane", mu.si, mu.plane)
        H.fill("mu/upanel", mu.si, mu.upanel)
        H.fill("mu/straw", mu.si, mu.straw)
        sec = s[~m]
        place, dmin = _place(s)
        H.fill("sec/eion_keV", sec.si, e[~m])
        H.fill("sec/len_mm", sec.si, sec.len)
        H.fill("sec/species", sec.si, pd.Series(sec.pdg.to_numpy()).map(SPECIES).fillna(3).to_numpy())
        H.fill("sec/place", sec.si, place)
        H.fill("sec/dplane", sec.si, np.where(dmin >= 0, np.minimum(dmin, 35), np.nan))
        st = s.groupby(T.KEY + ["sid"], sort=False).agg(si=("si", "first"), e=("eion", "sum"), n=("mu", "size"),
                                                        nmu=("mu", "sum"), upanel=("upanel", "first"),
                                                        straw=("straw", "first"))
        ek = 1e3 * st.e.to_numpy()
        hasmu = st.nmu.to_numpy() > 0
        H.fill("straw/eion_keV", st.si, ek)
        H.fill("straw/eion_mu_keV", st.si[hasmu], ek[hasmu])
        H.fill("straw/eion_seconly_keV", st.si[~hasmu], ek[~hasmu])
        only = hasmu & (st.n.to_numpy() == st.nmu.to_numpy())
        H.fill("straw/eion_muonly_keV", st.si[only], ek[only])
        H.fill("straw/nsteps", st.si, st.n)
        H.fill("straw/occupancy", st.si, st.upanel.to_numpy() * 96 + st.straw.to_numpy())
    return H.save(outdir, arm, {"kind": "steps", "inputs": [os.path.basename(p) for p in paths], "selected_events": nsel,
                                "batches": batches})


def _restrict_events(d, keep):
    from trkfastsim.validation.digicmp import _restrict
    return _restrict(d, keep)


def _selected_order(sel_keys, d):
    """Digi arrays on exactly the selected events in sel_keys order; selected events absent from the digi tables are
    added with no digis. Returns (digis, number of selected events absent)."""
    from trkfastsim.validation.digicmp import _keyid
    kd = _keyid(d["evkey"])
    if len(np.unique(kd)) != len(kd):
        raise ValueError("duplicate events in the digi tables")
    ks = _keyid(sel_keys)
    o = np.argsort(kd)
    pos = np.searchsorted(kd[o], ks)
    found = (pos < len(kd)) & (kd[o][np.minimum(pos, len(kd) - 1)] == ks)
    have = o[pos[found]]
    r = _restrict_events(d, have)
    # renumber to the full selected list, absent events keep zero digis
    full = np.flatnonzero(found)
    r["ev"] = full[r["ev"]]
    r["evkey"] = np.asarray(sel_keys, np.int64)
    pbt = np.zeros(len(ks), d["pbt"].dtype)
    pbt[full] = d["pbt"][have]
    r["pbt"] = pbt
    return r, int((~found).sum())


DIGI_KEYS = {  # shardhist key: digicmp HIST name
    "digi/panel_occupancy": "panel_occupancy", "digi/straw_occupancy": "straw_occupancy",
    "digi/tdc0_rel_ns": "tdc0_rel_ns", "digi/tdc1_rel_ns": "tdc1_rel_ns", "digi/tdc0_abs_ns": "tdc0_abs_ns",
    "digi/tdc1_abs_ns": "tdc1_abs_ns", "digi/tot0": "tot0", "digi/tot1": "tot1", "digi/pmp_adc": "pmp_adc",
    "digi/adc_sample": "adc_sample", "digi/dt_hv_minus_cal_ns": "dt_hv_minus_cal_ns", "digi/per_event": "digis_per_event",
    "hit/wdist_mm": "wdist_mm", "hit/edep_keV": "edep_keV", "hit/per_event": "strawhits_per_event",
    "hit/panelhits_per_event": "panelhits_per_event",
}


def _digi_edges():
    from trkfastsim.validation import digicmp as DC
    return {key: DC.HIST[name][1] for key, name in DIGI_KEYS.items()}


EDGES.update(_digi_edges())


def _digi_fill(H, obs, ev_si):
    from trkfastsim.validation import digicmp as DC
    for key, name in DIGI_KEYS.items():
        k, bins = DC.HIST[name]
        x = np.asarray(obs[k], float)
        if k in DC.PER_EVENT:
            si = ev_si
        elif k == "adc":
            si = ev_si[obs["adc_ev"]]
        else:
            si = ev_si[obs["ev"]]
        if k.startswith("tdc"):
            x = x * DC.LSB_TDC
        if k == "edep":
            x = x * 1000.0
        H.fill(key, si, x, bins)


def _stock_hits(paths, sel_keys, ev_si, H):
    """Stored stock straw hits (s_*) and combo hits (p_*) of the selected events under the hit/* keys."""
    import awkward as ak
    import uproot
    from trkfastsim.validation import digicmp as DC
    from trkfastsim.validation.digicmp import _keyid
    ks = _keyid(sel_keys)
    o = np.argsort(ks)
    nsh = np.zeros(len(ks), np.int64)
    nph = np.zeros(len(ks), np.int64)
    for p in paths:
        a = uproot.open(p)["trkdump/events"].arrays(["run", "subrun", "event", "s_wdist", "s_edep", "p_sid"])
        k = _keyid(np.stack([ak.to_numpy(a.run), ak.to_numpy(a.subrun), ak.to_numpy(a.event)], 1))
        pos = np.searchsorted(ks[o], k)
        ok = (pos < len(ks)) & (ks[o][np.minimum(pos, len(ks) - 1)] == k)
        idx = o[pos[ok]]
        a = a[ok]
        n_s = ak.to_numpy(ak.num(a.s_wdist))
        nsh[idx] = n_s
        nph[idx] = ak.to_numpy(ak.num(a.p_sid))
        si = np.repeat(ev_si[idx], n_s)
        H.fill("hit/wdist_mm", si, ak.to_numpy(ak.flatten(a.s_wdist)), DC.HIST["wdist_mm"][1])
        H.fill("hit/edep_keV", si, 1000.0 * ak.to_numpy(ak.flatten(a.s_edep)), DC.HIST["edep_keV"][1])
    H.fill("hit/per_event", ev_si, nsh, DC.HIST["strawhits_per_event"][1])
    H.fill("hit/panelhits_per_event", ev_si, nph, DC.HIST["panelhits_per_event"][1])


def _f2_fill(H, d, cond, drift, ev_si, nshard):
    """fit/* from the fast reco chain, one batch per shard (fixed shapes)."""
    from trkfastsim.validation.digicmp import _f2_fixed_batch
    import time
    t0 = time.perf_counter()
    seeds = 0
    for i in range(nshard):
        keep = np.flatnonzero(ev_si == i)
        if not len(keep):
            continue
        b = _restrict_events(d, keep)
        y = _f2_fixed_batch(b, cond, drift)
        seeds += int(y["seeds"])
        H.fill("fit/fits_per_event", np.full(len(keep), i), y["fits_per_event"])
        H.fill("fit/nactive", np.full(len(y["nactive"]), i), y["nactive"])
        H.fill("fit/chi2ndf", np.full(len(y["chi2ndf"]), i), y["chi2ndf"])
    return {"f2_seeds": seeds, "f2_wall_s": time.perf_counter() - t0}


def _kkline(paths, sel_keys, ev_si, H):
    import awkward as ak
    import uproot
    from trkfastsim.validation.digicmp import _keyid
    ks = _keyid(sel_keys)
    o = np.argsort(ks)
    nfit = np.zeros(len(ks), np.int64)
    seen = np.zeros(len(ks), bool)
    for p in T.expand(paths):
        a = uproot.open(p)["EventNtuple/ntuple"].arrays(
            ["evtinfo/run", "evtinfo/subrun", "evtinfo/event", "trk/trk.status", "trk/trk.nactive", "trk/trk.chisq",
             "trk/trk.ndof"], library="ak")
        k = _keyid(np.stack([ak.to_numpy(a[f"evtinfo/{c}"]) for c in ("run", "subrun", "event")], 1))
        pos = np.searchsorted(ks[o], k)
        ok = (pos < len(ks)) & (ks[o][np.minimum(pos, len(ks) - 1)] == k)
        idx = o[pos[ok]]
        if seen[idx].any():
            raise ValueError("duplicate EventNtuple rows for a selected event")
        seen[idx] = True
        st = a["trk/trk.status"][ok]
        good = st > 0
        nfit[idx] = ak.to_numpy(ak.sum(good, axis=1))
        si = np.repeat(ev_si[idx], ak.to_numpy(ak.sum(good, axis=1)))
        H.fill("fit/nactive", si, ak.to_numpy(ak.flatten(a["trk/trk.nactive"][ok][good])))
        H.fill("fit/chi2ndf", si, ak.to_numpy(ak.flatten((a["trk/trk.chisq"] / np.maximum(a["trk/trk.ndof"], 1))[ok][good])))
    H.fill("fit/fits_per_event", ev_si, nfit)
    return int((~seen).sum())


def digis_arm(paths, arm, outdir, select_steps, hit_cond, stock_hits=False, f2=False, drift_source=None, kkline=None,
              batches=1):
    from trkfastsim.reco import hitreco
    from trkfastsim.validation import digicmp as DC
    paths, select_steps = T.expand(paths), T.expand(select_steps)
    shards = _shards_of(select_steps, batches)
    sel = []
    for p in select_steps:                                  # one file at a time keeps memory low
        _, keys = load_selected_steps(p)
        sel.append(keys.to_numpy(np.int64))
    sel = np.concatenate(sel)
    sel = sel[np.lexsort((sel[:, 2], sel[:, 1], sel[:, 0]))]
    ev_si = _sidx(shards, _sid(sel[:, 1], sel[:, 2], batches))
    if hit_cond is None:
        hit_cond = os.path.join(outdir, f"{arm}.hitcond.json")
        os.makedirs(outdir, exist_ok=True)
        hitreco.extract_conditions(drift_source or paths[0], hit_cond)
    cond = hitreco.load_conditions(hit_cond)
    d, absent = _selected_order(sel, hitreco.read_digis(paths))
    obs = DC.observables(d, cond)
    H = Hists(shards)
    H.fill("sel/events", ev_si, np.zeros(len(sel)))
    _digi_fill(H, obs, ev_si)
    extra = {"kind": "digis", "inputs": [os.path.basename(p) for p in paths], "selected_events": int(len(sel)),
             "selected_absent_from_digis": absent, "digis": int(len(d["sid"])), "batches": batches}
    if f2:
        from trkfastsim.reco.cli import _drift_table
        extra.update(_f2_fill(H, d, cond, _drift_table(drift_source or paths[0]), ev_si, len(shards)))
    info = H.save(outdir, arm, extra)
    if stock_hits:
        S = Hists(shards)
        S.fill("sel/events", ev_si, np.zeros(len(sel)))
        _stock_hits(paths, sel, ev_si, S)
        S.save(outdir, f"{arm}@stockreco", {"kind": "stock hits", "inputs": extra["inputs"]})
    if kkline:
        K = Hists(shards)
        K.fill("sel/events", ev_si, np.zeros(len(sel)))
        miss = _kkline(kkline, sel, ev_si, K)
        K.save(outdir, f"{arm}@kkline", {"kind": "stock KKLine", "inputs": [os.path.basename(p) for p in T.expand(kkline)],
                                          "selected_absent_from_ntuple": miss})
    return info


def run(inputs, outdir, arm=None, kind="steps", select_steps=None, hit_cond=None, stock_hits=False, f2=False,
        drift_source=None, kkline=None, batches=1, **kw):
    if not arm:
        raise ValueError("--arm is required")
    if kind == "steps":
        info = steps_arm(inputs, arm, outdir, batches)
    else:
        if not select_steps:
            raise ValueError("digis need --select-steps")
        info = digis_arm(inputs, arm, outdir, select_steps, hit_cond, stock_hits, f2, drift_source, kkline, batches)
    print(json.dumps({k: v for k, v in info.items() if k != "totals"}))
    return info
