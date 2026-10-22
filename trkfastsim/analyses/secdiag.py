"""secdiag -- why non-muon particles make fewer KKLine track hits in the fast arm than in G4 (T-V2 follow-up diagnostic).

Same inputs and selection as tv2compare (events whose primary muon passes the PrimaryFilter tracker term in the step
tables). Three parts, each fast vs G4:

  steps   truth StrawGasSteps: per muon straw crossing, the share with a non-muon step in the same straw; non-muon
          steps by place (muon straw / same panel within ADJ straws of a muon straw / same panel further / elsewhere);
          in shared straws the delay of the earliest secondary after the muon and which step comes closer to the wire
          (the proxy for which StrawCluster crosses threshold first, so for the StrawDigiMC attribution); step shapes.
  hits    EventNtuple track hits (trkhits + trkhitsmc), joined by (event, StrawId) to the step tables: hit classes by
          straw content (muon + secondary, muon only, secondary only by adjacency), share of rel > 0 per class (so the
          rel > 0 share decomposes into class share x win probability), rel codes, pdg, startCode, edep, TOT.

    secdiag FAST_NTS [...] --compare G4_NTS [...] --steps FAST_TABLES [...] --steps-ref G4_SGSTEPS [...] --outdir DIR
"""
import json
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .. import geometry, tables as T
from .tv2compare import TREE, _keyid_rows, _selected

MIN_STEP_E = 2.0e-6     # MeV, StrawDigisFromStrawGasSteps minstepE
ADJ = 2                 # straws: "adjacent" = same panel, |dstraw| <= ADJ from a muon straw
POINT_LEN = 0.3         # mm, steps shorter than this make one cluster at the start point (proxy for meanFreePath)
FAST_OFFAXIS0 = 100000  # fast SimParticle keys: 1 muon, >= 1000 in-straw library, >= 100000 off-axis
REL_NAMES = {-1: "none", 0: "same", 1: "daughter", 2: "mother", 3: "sibling", 4: "udaughter", 5: "umother", 6: "usibling"}


def _steps(paths, sel, geom):
    """Selected events' StrawGasSteps with the event key, muon flag, source, pdg (and G4 creation code), wire distances."""
    s = T.read_many(paths, "sgsteps", T.KEY + ["sid", "upanel", "straw", "sp", "eion", "len", "shape", "t",
                                               "x0", "y0", "z0", "x1", "y1", "z1"])
    s["ev"] = _keyid_rows(s[T.KEY].to_numpy())
    s = s[np.isin(s.ev.to_numpy(), sel)].reset_index(drop=True)
    pcols = T.KEY + ["key", "pdg", "ismcprim"]
    with uproot.open(T.expand(paths)[0]) as f:
        if "ccode" in f["dump/particles"].keys():
            pcols.append("ccode")
    p = T.read_many(paths, "particles", pcols)
    p["ev"] = _keyid_rows(p[T.KEY].to_numpy())
    p = p[np.isin(p.ev.to_numpy(), sel)]
    mu = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)].drop_duplicates("ev")[["ev", "key"]].rename(columns={"key": "mu"})
    s = s.merge(mu, on="ev", how="inner")
    s = s.merge(p[["ev", "key", "pdg"] + (["ccode"] if "ccode" in p else [])].rename(columns={"key": "sp"}),
                on=["ev", "sp"], how="left")
    s["is_mu"] = (s.sp == s.mu).to_numpy()
    s["src"] = np.where(s.is_mu, "mu", np.where("ccode" in p, "g4", np.where(s.sp >= FAST_OFFAXIS0, "off", "ins")))
    # distance of the step to the wire: segment minimum, and of its start point (single-cluster steps)
    js = (s.upanel.to_numpy() * geometry.NSTRAWS + s.straw.to_numpy()).astype(np.int64)
    a = geometry.to_mu2e(s[["x0", "y0", "z0"]].to_numpy())
    b = geometry.to_mu2e(s[["x1", "y1", "z1"]].to_numpy())
    d, m = geom.dir[js], geom.mid[js]
    w0 = a - m
    w0 -= (w0 * d).sum(1)[:, None] * d
    w1 = b - m
    w1 -= (w1 * d).sum(1)[:, None] * d
    dw = w1 - w0
    tt = np.clip(-(w0 * dw).sum(1) / np.maximum((dw * dw).sum(1), 1e-12), 0, 1)
    s["dmin"] = np.linalg.norm(w0 + tt[:, None] * dw, axis=1)
    s["r0"] = np.linalg.norm(w0, axis=1)
    single = (s["shape"].to_numpy() == 3) | (s.len.to_numpy() < POINT_LEN)
    s["rclu"] = np.where(single, s.r0, s.dmin)
    return s


def _dsmin(df, mx):
    """Per row of df (ev, upanel, straw): min |straw - muon straw| over the event's muon straws in that panel (NaN: none)."""
    q = df[["ev", "upanel", "straw"]].reset_index().merge(
        mx[["ev", "upanel", "straw"]].rename(columns={"straw": "ms"}), on=["ev", "upanel"])
    q["ds"] = (q.straw - q.ms).abs()
    return q.groupby("index").ds.min().reindex(df.index).to_numpy()


def _place(inmu, ds):
    return np.where(inmu, "muon straw", np.where(ds <= ADJ, f"adjacent (<= {ADJ})",
                                                 np.where(np.isfinite(ds), "same panel", "elsewhere")))


def _q(x, qs=(0.1, 0.5, 0.9)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, qs)] if len(x) else []


def _top(x, n=8):
    u, c = np.unique(np.asarray(x), return_counts=True)
    o = np.argsort(c)[::-1][:n]
    return {str(int(u[i])): round(float(c[i] / c.sum()), 4) for i in o}


def _step_part(s):
    """Muon straw table, straw-content table, and the step-level metrics."""
    nev = int(s.ev.nunique())
    mu, sec = s[s.is_mu], s[~s.is_mu & (s.eion > MIN_STEP_E)].reset_index(drop=True)
    mx = mu.groupby(["ev", "sid"]).agg(t_mu=("t", "min"), r_mu=("rclu", "min"), upanel=("upanel", "first"),
                                       straw=("straw", "first")).reset_index()
    sec = sec.merge(mx[["ev", "sid", "t_mu", "r_mu"]], on=["ev", "sid"], how="left")
    sec["inmu"] = sec.t_mu.notna()
    sec["ds"] = _dsmin(sec, mx)
    sec["place"] = _place(sec.inmu.to_numpy(), sec.ds.to_numpy())
    out = {"events": nev, "muon_crossings": int(len(mx)), "muon_crossings_per_event": len(mx) / nev,
           "sec_steps_per_event": len(sec) / nev}
    # per muon crossing: secondary in the same straw
    sh = sec[sec.inmu].groupby(["ev", "sid"]).agg(t_sec=("t", "min"), r_sec=("rclu", "min"), n=("t", "size"),
                                                   e_sec=("eion", "sum")).reset_index()
    sh = sh.merge(mx, on=["ev", "sid"])
    out["muon_crossings_with_sec_same_straw"] = len(sh) / len(mx)
    out["shared"] = {"sec_steps_per_shared_straw": float(sh.n.mean()), "sec_eion_keV_q": _q(sh.e_sec * 1e3),
                     "dt_earliest_sec_minus_mu_ns_q": _q(sh.t_sec - sh.t_mu, (0.1, 0.5, 0.9, 0.99)),
                     "share_dt_gt_5ns": float(((sh.t_sec - sh.t_mu) > 5).mean()),
                     "share_sec_closer_to_wire": float((sh.r_sec < sh.r_mu).mean()),
                     "r_sec_mm_q": _q(sh.r_sec), "r_mu_mm_q": _q(sh.r_mu)}
    ins = sec[sec.inmu]
    out["shared"]["sec_step_shape_share"] = _top(ins["shape"])
    out["shared"]["sec_step_len_mm_q"] = _q(ins.len)
    out["shared"]["sec_pdg_share"] = _top(ins.pdg)
    if "ccode" in ins:
        out["shared"]["sec_ccode_share"] = _top(ins.ccode.fillna(-1))
    # where the secondary steps are, per selected event (and by fast source)
    pl = {}
    for place, g in sec.groupby("place"):
        pl[place] = {"steps_per_event": len(g) / nev, "eion_keV_per_event": float(g.eion.sum() * 1e3 / nev),
                     "by_src": {k: len(v) / nev for k, v in g.groupby("src")}}
    out["sec_place"] = pl
    # straws holding secondaries per event, by place (a hit needs a straw, not a step)
    st = sec.drop_duplicates(["ev", "sid"])
    out["sec_straws_per_event_by_place"] = {k: len(g) / nev for k, g in st.groupby("place")}
    # straw content for the hit join
    cont = s[s.eion > MIN_STEP_E].assign(m=lambda x: x.is_mu, i=lambda x: x.src == "ins", o=lambda x: x.src == "off",
                                          g=lambda x: x.src == "g4")
    cont = cont.groupby(["ev", "sid"]).agg(has_mu=("m", "any"), has_ins=("i", "any"), has_off=("o", "any"),
                                           has_g4=("g", "any"), nstep=("m", "size")).reset_index()
    cont["has_sec"] = cont.nstep > cont.has_mu.astype(int)
    return mx, cont, out


def _hits(paths, sel):
    """Track hits of selected events (all tracks, as tv2compare._trkhits) with event key, StrawId and truth."""
    cols = {k: [] for k in ("ev", "sid", "upanel", "straw", "state", "edep", "tot", "tstat", "rel", "rem", "pdg",
                            "startCode", "mcedep", "doca", "wlen", "rdrift")}
    for path in T.expand(paths):
        t = uproot.open(path)[TREE]
        ev = _keyid_rows(np.stack([t[f"evtinfo/{k}"].array(library="np").astype(np.int64)
                                   for k in ("run", "subrun", "event")], 1))
        m = np.isin(ev, sel)
        h, hm = t["trkhits"].array()[m], t["trkhitsmc"].array()[m]
        st = t["trk/trk.status"].array()[m]
        same = ak.all(ak.num(h, axis=2) == ak.num(hm, axis=2), axis=1)
        h, hm, st, evm = h[same], hm[same], st[same], ev[m][same]
        flat = lambda a: ak.to_numpy(ak.flatten(a, axis=None))
        cols["ev"].append(flat(ak.broadcast_arrays(ak.Array(evm), h.plane)[0]))
        cols["tstat"].append(flat(ak.broadcast_arrays(st, h.plane)[0]))
        pl, pa, sw = flat(h.plane).astype(np.int64), flat(h.panel).astype(np.int64), flat(h.straw).astype(np.int64)
        cols["sid"].append(geometry.straw_id(pl, pa, sw))
        cols["upanel"].append(pl * geometry.NPANELS + pa)
        cols["straw"].append(sw)
        cols["tot"].append(ak.to_numpy(ak.flatten(ak.flatten(h.tot, axis=1), axis=1)[:, 0]))
        for k, a in (("state", h.state), ("edep", h.edep), ("rel", hm.rel["_rel"]), ("rem", hm.rel["_rem"]),
                     ("pdg", hm.pdg), ("startCode", hm.startCode), ("mcedep", hm.edep), ("doca", hm.doca),
                     ("wlen", hm.len), ("rdrift", hm.rdrift)):
            cols[k].append(flat(a))
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})


def _hit_part(h, mx, cont):
    h = h.merge(cont, on=["ev", "sid"], how="left")
    for c in ("has_mu", "has_sec", "has_ins", "has_off", "has_g4"):
        h[c] = h[c].fillna(False).astype(bool)
    h["ds"] = _dsmin(h, mx)
    ds = h.ds.to_numpy()
    h["cls"] = np.where(h.has_mu & h.has_sec, "muon + sec", np.where(h.has_mu, "muon only", np.where(
        h.has_sec, np.where(ds <= ADJ, f"sec only, adjacent (<= {ADJ})", np.where(np.isfinite(ds), "sec only, same panel",
                                                                                  "sec only, elsewhere")), "no step")))
    oth = h.rel.to_numpy() > 0
    out = {"track_hits": int(len(h)), "rel_gt0_share": float(oth.mean()),
           "rel_codes": {REL_NAMES.get(int(k), str(k)): v for k, v in _top(h.rel, 10).items()},
           "rel_gt0": {"rel_codes": {REL_NAMES.get(int(k), str(k)): v for k, v in _top(h.rel[oth], 10).items()},
                       "rem": _top(h.rem[oth]), "pdg": _top(h.pdg[oth]), "startCode": _top(h.startCode[oth]),
                       "edep_keV_q": _q(h.edep[oth] * 1e3), "tot_ns_q": _q(h.tot[oth]),
                       "active_share": float((h.state[oth] > 0).mean()),
                       "straw_has_muon_step": float(h.has_mu[oth].mean())},
           "muon_hits": {"edep_keV_q": _q(h.edep[~oth] * 1e3), "tot_ns_q": _q(h.tot[~oth])}}
    cl = {}
    for c, g in h.groupby("cls"):
        o = g.rel.to_numpy() > 0
        d = {"hit_share": len(g) / len(h), "rel_gt0_in_class": float(o.mean()),
             "rel_gt0_per_track_hit": o.sum() / len(h)}
        if o.sum() > 20:
            d.update({"rel_gt0_tot_ns_q": _q(g.tot[o]), "rel_gt0_edep_keV_q": _q(g.edep[o] * 1e3),
                      "rel_gt0_abs_wire_pos_mm_q": _q(np.abs(g.wlen[o])), "rel_gt0_rdrift_mm_q": _q(g.rdrift[o]),
                      "rel_gt0_active": float((g.state[o] > 0).mean())})
        if (~o).sum() > 20:
            d.update({"rel0_tot_ns_q": _q(g.tot[~o]), "rel0_edep_keV_q": _q(g.edep[~o] * 1e3)})
        if c == "muon + sec":
            d["by_fast_src_rel_gt0"] = {"ins": float(o[g.has_ins.to_numpy()].mean()) if g.has_ins.any() else None,
                                        "off": float(o[g.has_off.to_numpy()].mean()) if g.has_off.any() else None}
        cl[c] = d
    out["by_class"] = cl
    act = h[h.state > 0]
    out["active_only"] = {"track_hits": int(len(act)), "rel_gt0_share": float((act.rel > 0).mean()),
                          "by_class_rel_gt0_per_track_hit": {c: float((g.rel > 0).sum() / len(act))
                                                             for c, g in act.groupby("cls")}}
    return out


def run(inputs, outdir, compare=None, steps=None, steps_ref=None, **kw):
    if not (compare and steps and steps_ref):
        raise SystemExit("secdiag FAST_NTS --compare G4_NTS --steps FAST_TABLES --steps-ref G4_SGSTEPS")
    from .steps import default_gdml
    geom = geometry.from_gdml(default_gdml())
    outdir = outdir or "plots/secdiag"
    os.makedirs(outdir, exist_ok=True)
    res = {}
    for arm, nts, tabs in (("fast", inputs, steps), ("g4", compare, steps_ref)):
        sel, _ = _selected(tabs)
        s = _steps(tabs, sel, geom)
        mx, cont, sp = _step_part(s)
        h = _hits(nts, sel)
        h = h[np.isin(h.ev.to_numpy(), s.ev.unique())]          # only events covered by the step tables read
        res[arm] = {"steps": sp, "hits": _hit_part(h, mx, cont)}
    with open(os.path.join(outdir, "secdiag.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
