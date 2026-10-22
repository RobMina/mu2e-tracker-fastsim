"""tv2compare -- T-V2/T-V3: stock digitisation + reco + EventNtuple on fast steps (path A) against the same chain on G4 steps.

Inputs: the EventNtuples of grid/grid_tv2.sh (fast and G4 arms), the step tables they came from (fast: fasttrk tables;
G4: StrawGasStepTableDumper sgsteps tables of the same dts), and optionally the StrawHitTableDumper tables (DUMP=1) for
the digi level. The arms have different CRY primaries (unpaired): every rate is per second of live time, and every
distribution is compared among events whose primary muon passes the PrimaryFilter tracker term in the step tables (the
G4 dts also keeps CRV-only events, which this drops).

T-V3 (EventNtuple): KKLine tracks per selected event and per second (yield), active hits per track, chi2/ndf, fit
consistency, track-to-truth residuals at TT_Outer (sid 4; identity-matched crossings as crvextrap: position perpendicular to the
track in two directions, space angle, time), against the agreed thresholds (yield within 1 %, hits within 1 %, widths within 3 %, chi2 binned ratio +-5 %).
T-V2 (hit tables, optional): StrawDigis per selected event, TDC/TOT spectra, per-panel occupancy chi2/ndf.

    tv2compare FAST_NTS [...] --compare G4_NTS [...] --steps FAST_TABLES [...] --steps-ref G4_SGSTEPS [...]
               [--hits FAST_HITS [...] --hits-ref G4_HITS [...]] --outdir DIR
"""
import json
import os

import awkward as ak
import numpy as np
import uproot

from mu2efastsim import stats
from .. import tables as T

TREE = "EventNtuple/ntuple"
SID_OUTER = 4


def _selected(step_paths):
    """Events (run, subrun, event) passing the tracker term, by the MC primary muon's steps; and the live time."""
    s = T.read_many(step_paths, "sgsteps")
    p = T.read_many(step_paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p[(p.ismcprim == 1) & (p.pdg.abs() == 13)][T.KEY + ["sp"]], on=T.KEY + ["sp"])
    f = T.trk_filter(s)
    sel = f[f.pass_trk].reset_index()[T.KEY]
    return _keyid(sel.to_numpy().astype(np.int64)), T.livetime(step_paths)


def _keyid(k):
    """(N,3) run/subrun/event -> one int64 key per event (subrun < 2^24, event < 2^24)."""
    k = np.asarray(k, np.int64)
    return np.unique((k[:, 0] << 48) | (k[:, 1] << 24) | k[:, 2])


def _keyid_rows(k):
    k = np.asarray(k, np.int64)
    return (k[:, 0] << 48) | (k[:, 1] << 24) | k[:, 2]


def _ntuple(paths, sel):
    """Per selected event: track quality arrays and the TT_Outer entry crossing of reco and truth (first track)."""
    br = ["evtinfo/run", "evtinfo/subrun", "evtinfo/event", "trk/trk.status", "trk/trk.nactive", "trk/trk.chisq",
          "trk/trk.ndof", "trk/trk.fitcon", "trk/trk.nhits"]
    out = {k: [] for k in ("ntrk", "nactive", "chi2ndf", "fitcon", "nhits", "dx", "dy", "dt", "dang", "pt")}
    for path in T.expand(paths):
        t = uproot.open(path)[TREE]
        a = t.arrays(br, library="ak")
        keys = np.stack([ak.to_numpy(a[f"evtinfo/{k}"]).astype(np.int64) for k in ("run", "subrun", "event")], 1)
        m = np.isin(_keyid_rows(keys), sel)
        st = a["trk/trk.status"][m]
        good = st > 0
        out["ntrk"].append(ak.to_numpy(ak.sum(good, axis=1)))
        for k, b in (("nactive", "nactive"), ("fitcon", "fitcon"), ("nhits", "nhits")):
            out[k].append(ak.to_numpy(ak.flatten(a[f"trk/trk.{b}"][m][good])))
        out["chi2ndf"].append(ak.to_numpy(ak.flatten((a["trk/trk.chisq"] / np.maximum(a["trk/trk.ndof"], 1))[m][good])))
        reco, mc = t["trksegs"].array()[m], t["trksegsmc"].array()[m]
        rp, rm, rt, tp, tm, tt = _pairs_at(reco, mc, SID_OUTER)
        if len(rp):
            u = tm / np.linalg.norm(tm, axis=1)[:, None]
            d = rp - tp
            dperp = d - (d * u).sum(1)[:, None] * u
            # signed perpendicular residuals: along the horizontal direction normal to the track, and its complement
            h = np.cross(u, np.array([0.0, 1.0, 0.0]))
            h /= np.maximum(np.linalg.norm(h, axis=1), 1e-12)[:, None]
            v = np.cross(h, u)
            out["dx"].append((dperp * h).sum(1))
            out["dy"].append((dperp * v).sum(1))
            out["dt"].append(rt - tt)
            ur = rm / np.linalg.norm(rm, axis=1)[:, None]
            out["dang"].append(np.arccos(np.clip((ur * u).sum(1), -1, 1)) * 1000.0)   # space angle [mrad]
            out["pt"].append(np.linalg.norm(tm, axis=1))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def _pairs_at(reco_segs, mc_segs, sid):
    """Reco <-> truth crossings of `sid` on the same track, matched as crvextrap.matching.match_segments (identity):
    same (sid, sindex), same multiplicity on both sides, paired in time order within each side (entry -> exit), and the
    same travel direction (mom . mom > 0). Vectorised over all tracks. Returns (reco pos, reco mom, reco time,
    truth pos, truth mom, truth time) arrays of the matched pairs."""
    from crvextrap import io as XIO
    _, ev_r, loc_r = XIO.flatten_tracks(reco_segs)
    _, ev_t, loc_t = XIO.flatten_tracks(mc_segs)
    fr = XIO.flat_xings(reco_segs, inbounds_only=True)
    ft = XIO.flat_xings(mc_segs, inbounds_only=False)

    def keyed(fx, ev, loc):
        trk = fx.track_of()
        m = fx.sid == sid
        k = (ev[trk][m] << 32) | (loc[trk][m] << 16) | fx.sindex[m]          # (event, track, sindex)
        o = np.lexsort((fx.time[m], k))
        idx = np.flatnonzero(m)[o]
        k = k[o]
        first = np.r_[True, k[1:] != k[:-1]]
        start = np.flatnonzero(first)
        rank = np.arange(len(k)) - np.repeat(start, np.diff(np.r_[start, len(k)]))
        cnt = np.repeat(np.diff(np.r_[start, len(k)]), np.diff(np.r_[start, len(k)]))
        return k, rank, cnt, idx

    kr, rr, cr, ir = keyed(fr, ev_r, loc_r)
    kt, rt, ct, it_ = keyed(ft, ev_t, loc_t)
    full_r = kr * 64 + rr
    full_t = kt * 64 + rt
    common, jr, jt = np.intersect1d(full_r, full_t, return_indices=True)
    ok = cr[jr] == ct[jt]                                                 # equal multiplicity only (identity case)
    jr, jt = ir[jr[ok]], it_[jt[ok]]
    same_dir = (fr.mom[jr] * ft.mom[jt]).sum(1) > 0
    jr, jt = jr[same_dir], jt[same_dir]
    return fr.pos[jr], fr.mom[jr], fr.time[jr], ft.pos[jt], ft.mom[jt], ft.time[jt]


def _width(x):
    """Core width: half the 16-84 % interval; bias: median."""
    if len(x) < 10:
        return float("nan"), float("nan")
    q16, q50, q84 = np.quantile(x, [0.16, 0.5, 0.84])
    return float(0.5 * (q84 - q16)), float(q50)


def _trkhits(paths, sel):
    """Per track hit among selected events (EventNtuple trkhits + trkhitsmc, aligned): TOT (cal end), edep, state, the
    true relationship to the track's particle (rel 0 = same particle) and trkhitsmc.len (the true cluster distance along
    the wire from the straw centre, not a path length)."""
    cols = {"tot": [], "edep": [], "state": [], "rel": [], "len": [], "plane": []}
    for path in T.expand(paths):
        t = uproot.open(path)[TREE]
        keys = np.stack([t[f"evtinfo/{k}"].array(library="np").astype(np.int64) for k in ("run", "subrun", "event")], 1)
        m = np.isin(_keyid_rows(keys), sel)
        h = t["trkhits"].array()[m]
        hm = t["trkhitsmc"].array()[m]
        same = ak.all(ak.num(h, axis=2) == ak.num(hm, axis=2), axis=1)   # same hit count per track, per event
        h, hm = h[same], hm[same]
        tot = ak.flatten(ak.flatten(h.tot, axis=1), axis=1)       # [hit][2]: TOT per straw end
        cols["tot"].append(ak.to_numpy(tot[:, 0]))                 # cal end, as the T-V2 digi comparison
        for k, a in (("edep", h.edep), ("state", h.state), ("plane", h.plane), ("rel", hm.rel["_rel"]), ("len", hm.len)):
            cols[k].append(ak.to_numpy(ak.flatten(a, axis=None)))
    return {k: np.concatenate(v) for k, v in cols.items()}


def _hits(paths, sel):
    """Digi-level arrays among selected events from StrawHitTableDumper tables."""
    out = {"ndigi": [], "tdc0": [], "tot0": [], "panel": []}
    for path in T.expand(paths):
        a = uproot.open(path)["trkdump/events"].arrays(["run", "subrun", "event", "d_sid", "d_tdc0", "d_tot0"], library="ak")
        keys = np.stack([ak.to_numpy(a[k]).astype(np.int64) for k in ("run", "subrun", "event")], 1)
        m = np.isin(_keyid_rows(keys), sel)
        out["ndigi"].append(ak.to_numpy(ak.num(a.d_sid[m])))
        sid = ak.to_numpy(ak.flatten(a.d_sid[m])).astype(np.int64)
        out["panel"].append((sid >> 7) & 0x3FF)
        out["tdc0"].append(ak.to_numpy(ak.flatten(a.d_tdc0[m])))
        out["tot0"].append(ak.to_numpy(ak.flatten(a.d_tot0[m])))
    return {k: np.concatenate(v) for k, v in out.items()}


def run(inputs, outdir, compare=None, steps=None, steps_ref=None, hits=None, hits_ref=None, **kw):
    if not (compare and steps and steps_ref):
        raise SystemExit("tv2compare FAST_NTS --compare G4_NTS --steps FAST_TABLES --steps-ref G4_SGSTEPS")
    outdir = outdir or "plots/tv2compare"
    os.makedirs(outdir, exist_ok=True)
    sel_f, lt_f = _selected(steps)
    sel_r, lt_r = _selected(steps_ref)
    nf, nr = _ntuple(inputs, sel_f), _ntuple(compare, sel_r)
    res = {"livetime_s": [lt_f, lt_r], "selected_events": [int(len(sel_f)), int(len(sel_r))],
           "selected_in_ntuple": [int(len(nf["ntrk"])), int(len(nr["ntrk"]))]}
    tf, tr = int(nf["ntrk"].sum()), int(nr["ntrk"].sum())
    rf, rr = tf / lt_f, tr / lt_r
    res["tracks"] = [tf, tr]
    res["yield_per_s"] = {"fast": rf, "g4": rr, "ratio_minus_1": rf / rr - 1, "err": rf / rr * np.sqrt(1 / max(tf, 1) + 1 / max(tr, 1))}
    res["tracks_per_selected_event"] = [tf / max(len(nf["ntrk"]), 1), tr / max(len(nr["ntrk"]), 1)]
    res["nactive_mean"] = [float(nf["nactive"].mean()), float(nr["nactive"].mean()), float(nf["nactive"].mean() / nr["nactive"].mean() - 1)]
    res["chi2ndf_band"] = stats.ratio_band(nf["chi2ndf"], nr["chi2ndf"], min_ref=100)["max_abs_dev_minus_2sigma"]
    res["fitcon_band"] = stats.ratio_band(nf["fitcon"], nr["fitcon"], min_ref=100)["max_abs_dev_minus_2sigma"]
    for k, unit in (("dx", "mm"), ("dy", "mm"), ("dang", "mrad"), ("dt", "ns")):
        wf, bf = _width(nf[k])
        wr, br = _width(nr[k])
        res[f"resid_{k}_{unit}"] = {"width": [wf, wr, wf / wr - 1 if wr else float("nan")], "bias": [bf, br],
                                    "bias_over_width": (bf - br) / wr if wr else float("nan"), "n": [int(len(nf[k])), int(len(nr[k]))]}
    if hits and hits_ref:
        hf, hr = _hits(hits, sel_f), _hits(hits_ref, sel_r)
        res["digis_per_event"] = [float(hf["ndigi"].mean()), float(hr["ndigi"].mean()), float(hf["ndigi"].mean() / hr["ndigi"].mean() - 1)]
        for k in ("tdc0", "tot0"):
            res[f"{k}_band"] = stats.ratio_band(hf[k], hr[k], min_ref=100)["max_abs_dev_minus_2sigma"]
        npan = 216
        cf, cr = np.bincount(hf["panel"], minlength=npan)[:npan], np.bincount(hr["panel"], minlength=npan)[:npan]
        ok = cr >= 20
        sf, sr = cf[ok] / cf.sum(), cr[ok] / cr.sum()
        chi = ((sf - sr) ** 2 / (sf / cf.sum() + sr / cr.sum())).sum()
        res["panel_occupancy_chi2ndf"] = float(chi / max(ok.sum() - 1, 1))
    if kw.get("tot_breakdown"):
        tf_, tr_ = _trkhits(inputs, sel_f), _trkhits(compare, sel_r)
        bd = {}
        for name, mf, mr in (("all", np.ones(len(tf_["tot"]), bool), np.ones(len(tr_["tot"]), bool)),
                             ("muon (rel 0)", tf_["rel"] == 0, tr_["rel"] == 0),
                             ("other (rel > 0)", tf_["rel"] > 0, tr_["rel"] > 0)):
            d = {"n": [int(mf.sum()), int(mr.sum())], "share": [float(mf.mean()), float(mr.mean())]}
            if mf.sum() > 100 and mr.sum() > 100:
                for k in ("tot", "edep"):
                    d[f"{k}_median"] = [float(np.median(tf_[k][mf])), float(np.median(tr_[k][mr]))]
                    d[f"{k}_band"] = stats.ratio_band(tf_[k][mf], tr_[k][mr], min_ref=100)["max_abs_dev_minus_2sigma"]
            bd[name] = d
        res["tot_breakdown"] = {"by_relation": bd}
    with open(os.path.join(outdir, "tv2compare.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
