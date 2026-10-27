"""tv2compare -- T-V2/T-V3: stock digitisation + reco + EventNtuple on fast steps (path A) against the same chain on G4 steps.

Inputs: the EventNtuples of grid/grid_tv2.sh (fast and G4 arms), the step tables they came from (fast: fasttrk tables;
G4: StrawGasStepTableDumper sgsteps tables of the same dts), and optionally the StrawHitTableDumper tables (DUMP=1) for
the digi level. The arms have different CRY primaries (unpaired): every rate is per second of live time, and every
distribution is compared among events where any primary muon passes the PrimaryFilter tracker term in the step tables (the
G4 dts also keeps CRV-only events, which this drops).

T-V3 (EventNtuple): KKLine tracks per selected event and per second (yield), active hits per track, chi2/ndf, fit
consistency, track-to-truth residuals at TT_Outer (sid 4; identity-matched crossings as crvextrap: position perpendicular to the
track in two directions, space angle, time), against the agreed thresholds (yield within 1 %, hits within 1 %, widths within 3 %, chi2 binned ratio +-5 %).
T-V2 (hit tables, optional): StrawDigis per selected event, TDC/TOT spectra, per-panel occupancy over the 216
uniquePanels with a full-covariance statistic and diagnostic Poisson/diagonal-clustered scores (occupancy.py; digis come in per-muon clusters, so Poisson errors
overstate the significance), and G4-against-G4 floors (even against odd G4 hit files) for the occupancy and the bands.
Residual widths carry jackknife errors over input files and a split by crossing (entry = first in time, exit).
--paired (both arms on the same CRY primaries, e.g. g4ref_v2 and the fast sim on its gen stream): the yield difference
split into (a) events selected in one arm only and (b) the track-count difference of events selected in both.
Paired mode checks equal subrun exposure and livetime, but these metadata do not prove generated event identities.
Paired occupancy uses an independent-arm covariance approximation, not a certification statistic.
The thresholds block lists every row against its agreed threshold.

    tv2compare FAST_NTS [...] --compare G4_NTS [...] --steps FAST_TABLES [...] --steps-ref G4_SGSTEPS [...]
               [--hits FAST_HITS [...] --hits-ref G4_HITS [...]] [--paired] --outdir DIR
"""
import json
import os

import awkward as ak
import numpy as np
import uproot

from mu2efastsim import stats
from .. import tables as T
from . import occupancy as OCC

TREE = "EventNtuple/ntuple"
SID_OUTER = 4


def _selected(step_paths, with_nmax=False):
    """Events (run, subrun, event) passing the tracker term for any primary muon; and the live time. With
    with_nmax also the maximum good-step count of any primary muon of each selected event (same order as the sorted keys)."""
    s = T.read_many(step_paths, "sgsteps")
    p = T.read_many(step_paths, "particles", T.KEY + ["key", "pdg", "isprim"]).rename(columns={"key": "sp"})
    if p.duplicated(T.KEY + ["sp"]).any():
        raise ValueError("duplicate particle identity in step tables")
    s = s.merge(p, on=T.KEY + ["sp"], how="left", validate="many_to_one", indicator=True)
    if (s["_merge"] != "both").any():
        raise ValueError("step tables contain steps without particle coverage")
    s = s[(s.isprim == 1) & (s.pdg.abs() == 13)]
    f = T.trk_filter(s)
    sel = f[f.pass_trk].reset_index()
    k = _keyid_rows(sel[T.KEY].to_numpy().astype(np.int64))
    o = np.argsort(k)
    if with_nmax:
        return k[o], T.livetime(step_paths), sel.nmax.to_numpy()[o]
    return k[o], T.livetime(step_paths)


def _keyid(k):
    """(N,3) run/subrun/event -> one int64 key per event (subrun < 2^24, event < 2^24)."""
    k = np.asarray(k, np.int64)
    return np.unique((k[:, 0] << 48) | (k[:, 1] << 24) | k[:, 2])


def _keyid_rows(k):
    k = np.asarray(k, np.int64)
    return (k[:, 0] << 48) | (k[:, 1] << 24) | k[:, 2]


def _validate_coverage(keys, selected, label):
    """Event rows must be unique, and selected events require explicit rows (including zero counts)."""
    keys, selected = np.asarray(keys, np.int64), np.asarray(selected, np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError(f"{label}: duplicate selected event identity")
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"{label}: duplicate event rows")
    missing = np.setdiff1d(selected, keys)
    if len(missing):
        raise ValueError(f"{label}: missing {len(missing)} selected event rows; absent rows are not zero counts")


def _paired_exposure(steps, steps_ref, lt_f, lt_r):
    """Check subrun exposure, not identity of the underlying CRY primaries.

    The current dumper stores CosmicLivetime.primaries(), not nEvents; newer schemas
    may provide nEvents as well. Neither metadata field proves event identities.
    """
    frames = [T.read_many(paths, "subruns") for paths in (steps, steps_ref)]
    for arm, frame in zip(("fast", "G4"), frames):
        if not {"run", "subrun", "livetime"}.issubset(frame.columns) or frame.empty:
            raise ValueError(f"paired {arm}: missing subrun exposure metadata")
        if frame.duplicated(["run", "subrun"]).any():
            raise ValueError(f"paired {arm}: duplicate subrun exposure rows")
        if not np.all(np.isfinite(frame.livetime) & (frame.livetime > 0)):
            raise ValueError(f"paired {arm}: invalid subrun livetime")
    a, b = [f.set_index(["run", "subrun"]).sort_index() for f in frames]
    if not a.index.equals(b.index):
        raise ValueError("paired arms have different generated (run, subrun) exposure")
    counts = [c for c in ("nEvents", "primaries") if c in a and c in b]
    if not counts:
        raise ValueError("paired exposure requires matching nEvents or primaries metadata")
    for col in counts:
        for f in (a, b):
            v = f[col].to_numpy()
            if not np.all(np.isfinite(v) & (v > 0) & (v == np.floor(v))):
                raise ValueError(f"paired exposure: invalid {col}")
        if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
            raise ValueError(f"paired arms have different generated exposure ({col})")
    if (not np.allclose(a.livetime, b.livetime, rtol=1e-10, atol=0)
            or not np.isclose(lt_f, lt_r, rtol=1e-10, atol=0)
            or not np.isclose(a.livetime.sum(), lt_f, rtol=1e-10, atol=0)
            or not np.isclose(b.livetime.sum(), lt_r, rtol=1e-10, atol=0)):
        raise ValueError("paired arms require equal subrun and total livetime")
    limitation = "Equal exposure metadata does not verify generated event identities or primary content."
    if "nEvents" not in counts:
        limitation += " nEvents is unavailable; primaries is CosmicLivetime exposure, not a generated-event count."
    return {"subruns": len(a), "count_fields_checked": counts,
            "generated_event_counts_verified": "nEvents" in counts,
            "generated_identities_verified": False, "limitation": limitation,
            "livetime_verified": True}


def _ntuple(paths, sel):
    """Per selected event: track quality arrays and the TT_Outer entry crossing of reco and truth (first track)."""
    br = ["evtinfo/run", "evtinfo/subrun", "evtinfo/event", "trk/trk.status", "trk/trk.nactive", "trk/trk.chisq",
          "trk/trk.ndof", "trk/trk.fitcon", "trk/trk.nhits"]
    out = {k: [] for k in ("key", "ntrk", "nactive", "chi2ndf", "fitcon", "nhits", "dx", "dy", "dt", "dang", "pt", "xrank",
                           "xfile", "tfile")}
    seen = []
    for ifile, path in enumerate(T.expand(paths)):
        t = uproot.open(path)[TREE]
        a = t.arrays(br, library="ak")
        keys = np.stack([ak.to_numpy(a[f"evtinfo/{k}"]).astype(np.int64) for k in ("run", "subrun", "event")], 1)
        seen.append(_keyid_rows(keys))
        m = np.isin(seen[-1], sel)
        st = a["trk/trk.status"][m]
        good = st > 0
        out["key"].append(_keyid_rows(keys)[m])
        out["ntrk"].append(ak.to_numpy(ak.sum(good, axis=1)))
        for k, b in (("nactive", "nactive"), ("fitcon", "fitcon"), ("nhits", "nhits")):
            out[k].append(ak.to_numpy(ak.flatten(a[f"trk/trk.{b}"][m][good])))
        out["tfile"].append(np.full(len(out["nactive"][-1]), ifile))
        out["chi2ndf"].append(ak.to_numpy(ak.flatten((a["trk/trk.chisq"] / np.maximum(a["trk/trk.ndof"], 1))[m][good])))
        reco, mc = t["trksegs"].array()[m], t["trksegsmc"].array()[m]
        rp, rm, rt, tp, tm, tt, xr = _pairs_at(reco, mc, SID_OUTER)
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
            out["xrank"].append(xr)
            out["xfile"].append(np.full(len(xr), ifile))
    _validate_coverage(np.concatenate(seen) if seen else [], sel, "EventNtuple")
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def _pairs_at(reco_segs, mc_segs, sid):
    """Reco <-> truth crossings of `sid` on the same track, matched as crvextrap.matching.match_segments (identity):
    same (sid, sindex), same multiplicity on both sides, paired in time order within each side (entry -> exit), and the
    same travel direction (mom . mom > 0). Vectorised over all tracks. Returns (reco pos, reco mom, reco time,
    truth pos, truth mom, truth time, crossing rank in time order: 0 = entry, 1 = exit) arrays of the matched pairs."""
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
    rank = rr[jr[ok]]
    jr, jt = ir[jr[ok]], it_[jt[ok]]
    same_dir = (fr.mom[jr] * ft.mom[jt]).sum(1) > 0
    jr, jt, rank = jr[same_dir], jt[same_dir], rank[same_dir]
    return fr.pos[jr], fr.mom[jr], fr.time[jr], ft.pos[jt], ft.mom[jt], ft.time[jt], rank


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
    """Digi-level arrays among selected events from StrawHitTableDumper tables: digis per event, TDC/TOT (cal end), and
    per digi its uniquePanel, event key and input-file index."""
    out = {"ndigi": [], "tdc0": [], "tot0": [], "panel": [], "ev": [], "file": [], "key": []}
    seen = []
    for ifile, path in enumerate(T.expand(paths)):
        a = uproot.open(path)["trkdump/events"].arrays(["run", "subrun", "event", "d_sid", "d_tdc0", "d_tot0"], library="ak")
        keys = np.stack([ak.to_numpy(a[k]).astype(np.int64) for k in ("run", "subrun", "event")], 1)
        seen.append(_keyid_rows(keys))
        m = np.isin(seen[-1], sel)
        nd = ak.to_numpy(ak.num(a.d_sid[m]))
        out["ndigi"].append(nd)
        out["key"].append(_keyid_rows(keys)[m])
        sid = ak.to_numpy(ak.flatten(a.d_sid[m])).astype(np.int64)
        out["panel"].append(OCC.unique_panel(sid))
        out["ev"].append(np.repeat(_keyid_rows(keys)[m], nd))
        out["file"].append(np.full(len(sid), ifile))
        out["tdc0"].append(ak.to_numpy(ak.flatten(a.d_tdc0[m])))
        out["tot0"].append(ak.to_numpy(ak.flatten(a.d_tot0[m])))
    _validate_coverage(np.concatenate(seen) if seen else [], sel, "hit tables")
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def _band_or_none(fast, ref, min_ref=100):
    """An empty comparison arm makes the diagnostic unavailable, not a zero discrepancy."""
    if not len(fast) or not len(ref):
        return None
    return stats.ratio_band(fast, ref, min_ref=min_ref)["max_abs_dev_minus_2sigma"]


def _jack_ratio(fn, xf, gf, xr, gr, ng):
    """fn(x_fast) / fn(x_ref) - 1 with a delete-one-group jackknife error (groups = input-file index mod ng)."""
    a, b = fn(xf), fn(xr)
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float("nan"), float("nan")
    rk = []
    for k in range(ng):
        fa, fb = fn(xf[gf % ng != k]), fn(xr[gr % ng != k])
        if np.isfinite(fa) and np.isfinite(fb) and fb:
            rk.append(fa / fb)
    rk = np.array(rk)
    g = len(rk)
    return float(a / b - 1), (float(np.sqrt((g - 1) / g * ((rk - rk.mean()) ** 2).sum())) if g > 1 else float("nan"))


def _per_key(keys, vals, q):
    """Sum of vals per key, looked up for the query keys q (0 where a key is absent)."""
    u, inv = np.unique(np.asarray(keys, np.int64), return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(vals, float), minlength=len(u))
    q = np.asarray(q, np.int64)
    if not len(u):
        return np.zeros(len(q))
    i = np.clip(np.searchsorted(u, q), 0, len(u) - 1)
    return np.where(u[i] == q, s[i], 0.0)


def _paired_yield(nf, nr, sel_f, sel_r, nmax_f, nmax_r):
    """Paired attribution of the track-yield difference (same CRY primaries in both arms): T = tracks summed over each
    arm's selected events; (T_f - T_g) / T_g = (a) events selected in one arm only + (b) the track-count difference in
    events selected in both. Ratio errors use event influence residuals, including
    the fluctuating G4 denominator and covariance between the two contributions."""
    tf, tg = (nf["key"], nf["ntrk"]), (nr["key"], nr["ntrk"])
    kf, kg = np.asarray(sel_f), np.asarray(sel_r)
    for arm, data, selected, nmax in (("fast", nf, kf, nmax_f), ("G4", nr, kg, nmax_r)):
        _validate_coverage(data["key"], selected, f"paired {arm} EventNtuple")
        if len(data["key"]) != len(data["ntrk"]) or len(selected) != len(nmax):
            raise ValueError(f"paired {arm}: misaligned event arrays")
    both = np.intersect1d(kf, kg)
    only_f, only_g = np.setdiff1d(kf, kg), np.setdiff1d(kg, kf)
    t = lambda d, k: _per_key(d[0], d[1], k)
    Tg = t(tg, kg).sum()
    if Tg <= 0:
        raise ValueError("zero reference track yield: paired ratios and errors are undefined")
    a_f, a_g = t(tf, only_f), t(tg, only_g)
    bf, bg = t(tf, both), t(tg, both)
    d = bf - bg
    a = (a_f.sum() - a_g.sum()) / Tg
    b = d.sum() / Tg
    union = np.union1d(kf, kg)
    in_f, in_g = np.isin(union, kf), np.isin(union, kg)
    fe = np.where(in_f, t(tf, union), 0.0)
    ge = np.where(in_g, t(tg, union), 0.0)
    ae = np.where(in_f & ~in_g, fe, np.where(in_g & ~in_f, -ge, 0.0))
    be = np.where(in_f & in_g, fe - ge, 0.0)
    ia, ib = (ae - a * ge) / Tg, (be - b * ge) / Tg
    ea, eb = np.linalg.norm(ia), np.linalg.norm(ib)
    covariance = float(ia @ ib)
    total_err = float(np.linalg.norm(ia + ib))
    assert np.isclose(a + b, fe.sum() / Tg - 1)
    assert np.isclose(total_err ** 2, ea ** 2 + eb ** 2 + 2 * covariance)
    selected_ratio = len(kf) / len(kg) - 1
    selected_influence = (in_f.astype(float) - in_g - selected_ratio * in_g) / len(kg)
    # Conditional-content factor: (Tf/Nf)/(Tg/Ng), with all four fluctuations paired.
    if len(kf):
        conditional_ratio = fe.sum() * len(kg) / (Tg * len(kf))
        conditional_influence = (len(kg) / (Tg * len(kf)) * (fe - fe.sum() / len(kf) * in_f)
                                 - conditional_ratio / Tg * (ge - Tg / len(kg) * in_g))
        conditional_err = float(np.linalg.norm(conditional_influence))
    else:
        conditional_err = float("nan")
    sf = _per_key(kf, nmax_f, both)
    sg = _per_key(kg, nmax_r, both)
    lost, gained = (bg > 0) & (bf == 0), (bf > 0) & (bg == 0)
    edges = [12, 14, 16, 20, 25, 30, 40, 1000]
    by = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sf >= lo) & (sf < hi)
        by.append({"fast_muon_straws": [lo, hi], "events": int(m.sum()), "lost": int((lost & m).sum()),
                   "gained": int((gained & m).sum()), "track_diff": float(d[m].sum()),
                   "mean_straws_fast_minus_g4": float((sf[m] - sg[m]).mean()) if m.any() else float("nan")})
    dsel = sf - sg
    byd = []
    for lo, hi in ((-1000, -3), (-3, -1), (-1, 0), (0, 1), (1, 3), (3, 1000)):
        m = (dsel >= lo) & (dsel < hi)
        byd.append({"straws_fast_minus_g4": [lo, hi], "events": int(m.sum()), "lost": int((lost & m).sum()),
                    "gained": int((gained & m).sum()), "track_diff": float(d[m].sum())})
    return {"tracks": [float(t(tf, kf).sum()), float(Tg)], "total_ratio_minus_1": float(a + b),
            "total_err": total_err, "selection_content_covariance": covariance,
            "error_method": "event influence residuals including fluctuating reference denominator",
            "tracks_per_selected_event_err": conditional_err,
            "a_selection_only": {"value": float(a), "err": float(ea), "events_fast_only": int(len(only_f)),
                                 "events_g4_only": int(len(only_g)), "tracks_fast_only": float(a_f.sum()),
                                 "tracks_g4_only": float(a_g.sum())},
            "b_both_selected_content": {"value": float(b), "err": float(eb), "events": int(len(both)),
                                        "tracks": [float(bf.sum()), float(bg.sum())], "events_lost_track": int(lost.sum()),
                                        "events_gained_track": int(gained.sum()),
                                        "by_fast_muon_straws": by, "by_straw_difference": byd},
            "selected_rate_ratio_minus_1": float(selected_ratio),
            "selected_rate_err": float(np.linalg.norm(selected_influence))}


def _yield_summary(nf, nr, sel_f, sel_r, lt_f, lt_r):
    """Use exactly the step-selected denominator, never the number of surviving rows."""
    counts = []
    for label, data, selected, lt in (("fast", nf, sel_f, lt_f), ("G4", nr, sel_r, lt_r)):
        _validate_coverage(data["key"], selected, f"{label} EventNtuple")
        if len(data["key"]) != len(data["ntrk"]):
            raise ValueError(f"{label}: misaligned event arrays")
        if not len(selected) or not np.isfinite(lt) or lt <= 0:
            raise ValueError(f"{label}: selected-event denominator and livetime must be positive")
        counts.append(_per_key(data["key"], data["ntrk"], selected))
    cf, cr = counts
    tf, tr = cf.sum(), cr.sum()
    if tr <= 0:
        raise ValueError("zero reference track yield: yield ratios and factorization are undefined")
    mf, mr = cf.mean(), cr.mean()
    ratio = (tf / lt_f) / (tr / lt_r)
    selected_ratio = (len(sel_f) / lt_f) / (len(sel_r) / lt_r)
    conditional_ratio = mf / mr
    product = selected_ratio * conditional_ratio
    assert np.isclose(ratio, product, rtol=1e-12, atol=1e-14), "yield factorization failed"
    # Independent-arm compound-event uncertainty (paired mode replaces these errors).
    yield_err = lt_r / lt_f / tr * np.sqrt(cf @ cf + (tf / tr) ** 2 * (cr @ cr))
    conditional_err = np.sqrt(cf.var() / len(cf) + conditional_ratio ** 2 * cr.var() / len(cr)) / mr
    return {"tracks": [int(tf), int(tr)],
            "yield_per_s": {"fast": float(tf / lt_f), "g4": float(tr / lt_r),
                            "ratio_minus_1": float(ratio - 1), "err": float(yield_err)},
            "tracks_per_selected_event": [float(mf), float(mr)],
            "yield_factorised": {"selected_rate_ratio_minus_1": float(selected_ratio - 1),
                                 "selected_rate_err": float(selected_ratio * np.sqrt(1 / len(sel_f) + 1 / len(sel_r))),
                                 "tracks_per_selected_event_ratio_minus_1": float(conditional_ratio - 1),
                                 "tracks_per_selected_event_err": float(conditional_err),
                                 "product_ratio_minus_1": float(product - 1), "factorization_verified": True}}


def _thresholds(res):
    """Every T-V2/T-V3 row against its agreed threshold (value, error, threshold, pass)."""
    rows = {}
    band_ok = lambda value, threshold: value is not None and np.isfinite(value) and value <= threshold
    y = res["yield_per_s"]
    rows["T-V3 KKLine yield per s"] = [y["ratio_minus_1"], y["err"], 0.01, abs(y["ratio_minus_1"]) <= 0.01]
    n = res["nactive_mean"]
    rows["T-V3 active hits per track"] = [n[2], n[3], 0.01, abs(n[2]) <= 0.01]
    for k in ("dx_mm", "dy_mm", "dang_mrad", "dt_ns"):
        w = res[f"resid_{k}"]
        rows[f"T-V3 TT_Outer width {k}"] = [w["width"][2], w["width_err"], 0.03, abs(w["width"][2]) <= 0.03]
    rows["T-V3 chi2/ndf binned ratio (minus 2 sigma)"] = [res["chi2ndf_band"], None, 0.05, band_ok(res["chi2ndf_band"], 0.05)]
    if "digis_per_event" in res:
        d = res["digis_per_event"]
        rows["T-V2 digis per selected event"] = [d[2], d[3], 0.01, abs(d[2]) <= 0.01]
        for k in ("tdc0", "tot0"):
            rows[f"T-V2 {k} binned ratio (minus 2 sigma)"] = [res[f"{k}_band"], None, 0.03, band_ok(res[f"{k}_band"], 0.03)]
        po = res["panel_occupancy"]
        cov_score = po.get("chi2ndf_covariance", float("nan"))
        cov_available = po.get("covariance_rank", 0) > 0 and not np.isnan(cov_score)
        occupancy_rows = {
            "T-V2 panel occupancy full covariance": [cov_score, None, 1.5,
                 cov_available and cov_score <= 1.5 and po["panels_beyond_5sigma"] == 0],
            "T-V2 panel occupancy diagonal clustered score (diagnostic)": [po["chi2ndf_clustered"], None, 1.5,
                 po["chi2ndf_clustered"] <= 1.5],
            "T-V2 panel occupancy Poisson score (diagnostic)": [po["chi2ndf_poisson"], None, 1.5,
                 po["chi2ndf_poisson"] <= 1.5]}
        rows.update(occupancy_rows)
    out = {k: {"value": v[0], "err": v[1], "threshold": v[2], "pass": bool(v[3])} for k, v in rows.items()}
    if "digis_per_event" in res:
        for key in occupancy_rows:
            certifying = "full covariance" in key and cov_available and not res.get("paired", False)
            out[key]["certifying"] = certifying
            if not certifying:
                out[key]["within_threshold_diagnostic"] = out[key]["pass"]
                out[key]["pass"] = None
            out[key]["interpretation"] = (
                "independent-arm approximation; paired cross-arm covariance missing; not certification"
                if res.get("paired", False) else
                "full panel covariance" if certifying else "diagnostic only; no covariance-based certification")
    return out


def run(inputs, outdir, compare=None, steps=None, steps_ref=None, hits=None, hits_ref=None, paired=False, **kw):
    if not (compare and steps and steps_ref):
        raise SystemExit("tv2compare FAST_NTS --compare G4_NTS --steps FAST_TABLES --steps-ref G4_SGSTEPS")
    if bool(hits) != bool(hits_ref):
        raise ValueError("hit-table comparison requires both --hits and --hits-ref")
    outdir = outdir or "plots/tv2compare"
    os.makedirs(outdir, exist_ok=True)
    sel_f, lt_f, nmax_f = _selected(steps, with_nmax=True)
    sel_r, lt_r, nmax_r = _selected(steps_ref, with_nmax=True)
    exposure = _paired_exposure(steps, steps_ref, lt_f, lt_r) if paired else None
    nf, nr = _ntuple(inputs, sel_f), _ntuple(compare, sel_r)
    ng = max(2, min(len(T.expand(inputs)), len(T.expand(compare))))
    res = {"paired": bool(paired), "livetime_s": [lt_f, lt_r], "selected_events": [int(len(sel_f)), int(len(sel_r))],
           "selected_in_ntuple": [int(len(nf["ntrk"])), int(len(nr["ntrk"]))],
           "input_files": {k: T.expand(v) if v else [] for k, v in
                           (("ntuple_fast", inputs), ("ntuple_g4", compare), ("steps_fast", steps),
                            ("steps_g4", steps_ref), ("hits_fast", hits), ("hits_g4", hits_ref))},
           "selection_convention": "any primary muon (isprim == 1, abs(pdg) == 13)",
           "input_coverage": "unique event rows; every selected event explicitly present; highest ROOT tree cycle"}
    res.update(_yield_summary(nf, nr, sel_f, sel_r, lt_f, lt_r))
    if paired:
        res["paired_exposure"] = exposure
        yp = res["yield_paired"] = _paired_yield(nf, nr, sel_f, sel_r, nmax_f, nmax_r)
        assert np.isclose(res["yield_per_s"]["ratio_minus_1"], yp["total_ratio_minus_1"])
        res["yield_per_s"]["err"] = yp["total_err"]
        for key in ("selected_rate_err", "tracks_per_selected_event_err"):
            res["yield_factorised"][key] = yp[key]
    na = [float(nf["nactive"].mean()), float(nr["nactive"].mean())]
    res["nactive_mean"] = na + [na[0] / na[1] - 1, float(na[0] / na[1] * np.hypot(nf["nactive"].std() / na[0] / np.sqrt(len(nf["nactive"])),
                                                                              nr["nactive"].std() / na[1] / np.sqrt(len(nr["nactive"]))))]
    res["chi2ndf_band"] = stats.ratio_band(nf["chi2ndf"], nr["chi2ndf"], min_ref=100)["max_abs_dev_minus_2sigma"]
    res["fitcon_band"] = stats.ratio_band(nf["fitcon"], nr["fitcon"], min_ref=100)["max_abs_dev_minus_2sigma"]
    wfn = lambda x: _width(x)[0]
    for k, unit in (("dx", "mm"), ("dy", "mm"), ("dang", "mrad"), ("dt", "ns")):
        wf, bf = _width(nf[k])
        wr, br = _width(nr[k])
        r, e = _jack_ratio(wfn, nf[k], nf["xfile"], nr[k], nr["xfile"], ng)
        d = {"width": [wf, wr, wf / wr - 1 if wr else float("nan")], "width_err": e, "bias": [bf, br],
             "bias_over_width": (bf - br) / wr if wr else float("nan"), "n": [int(len(nf[k])), int(len(nr[k]))]}
        for rank, lab in ((0, "entry"), (1, "exit")):   # time order along the track: upper (entry) and lower (exit) crossing
            mf, mr = nf["xrank"] == rank, nr["xrank"] == rank
            r, e = _jack_ratio(wfn, nf[k][mf], nf["xfile"][mf], nr[k][mr], nr["xfile"][mr], ng)
            d[lab] = {"width": [_width(nf[k][mf])[0], _width(nr[k][mr])[0], r], "width_err": e, "n": [int(mf.sum()), int(mr.sum())]}
        res[f"resid_{k}_{unit}"] = d
    if hits and hits_ref:
        hf, hr = _hits(hits, sel_f), _hits(hits_ref, sel_r)
        mdf, mdr = hf["ndigi"].mean(), hr["ndigi"].mean()
        if mdr <= 0:
            raise ValueError("zero reference digi yield: digi ratios are undefined")
        digi_ratio = mdf / mdr
        digi_err = np.sqrt(hf["ndigi"].var() / len(sel_f)
                           + digi_ratio ** 2 * hr["ndigi"].var() / len(sel_r)) / mdr
        if paired:
            # Same selected-event ratio estimator as tracks, now with digi counts.
            dp = _paired_yield({"key": hf["key"], "ntrk": hf["ndigi"]},
                               {"key": hr["key"], "ntrk": hr["ndigi"]}, sel_f, sel_r, nmax_f, nmax_r)
            digi_err = dp["tracks_per_selected_event_err"]
        res["digis_per_event"] = [float(mdf), float(mdr), float(digi_ratio - 1), float(digi_err)]
        even, odd = hr["file"] % 2 == 0, hr["file"] % 2 == 1
        for k in ("tdc0", "tot0"):
            res[f"{k}_band"] = stats.ratio_band(hf[k], hr[k], min_ref=100)["max_abs_dev_minus_2sigma"]
            res[f"{k}_band_g4_even_vs_odd_files"] = _band_or_none(hr[k][even], hr[k][odd])
        af, ar = OCC.accumulate(hf["ev"], hf["panel"]), OCC.accumulate(hr["ev"], hr["panel"])
        res["panel_occupancy"] = OCC.compare(af, ar)
        res["panel_occupancy"]["arm_covariance_assumption"] = (
            "independent-arm approximation; paired cross-arm covariance not included; not certification"
            if paired else "independent arms")
        res["panel_occupancy_g4_even_vs_odd_files"] = OCC.compare(OCC.accumulate(hr["ev"][even], hr["panel"][even]),
                                                                  OCC.accumulate(hr["ev"][odd], hr["panel"][odd]))
        # the pre-fix metric: Poisson errors on (sid >> 7) & 0x3FF = plane * 8 + panel, truncated to 216 bins (planes 0-26)
        leg = lambda up: (up // 6) * 8 + up % 6
        cf = np.bincount(leg(hf["panel"]), minlength=300)[:216]
        cr = np.bincount(leg(hr["panel"]), minlength=300)[:216]
        ok = cr >= 20
        sf, sr = cf[ok] / cf.sum(), cr[ok] / cr.sum()
        res["panel_occupancy_chi2ndf_legacy_binning"] = float(((sf - sr) ** 2 / (sf / cf.sum() + sr / cr.sum())).sum() / max(ok.sum() - 1, 1))
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
    res["thresholds"] = _thresholds(res)
    with open(os.path.join(outdir, "tv2compare.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("panel_occupancy")}, indent=1, default=float))
    for k in [k for k in res if k.startswith("panel_occupancy") and isinstance(res[k], dict)]:
        v = res[k]
        print(f"{k}: full covariance chi2/ndf {v.get('chi2ndf_covariance', float('nan')):.2f}; "
              f"diagnostic Poisson {v['chi2ndf_poisson']:.2f}, diagonal clustered {v['chi2ndf_clustered']:.2f}; overdispersion "
              f"{v['overdispersion_median']:.2f} max|pull| {v['max_abs_pull']:.1f} beyond 5 sigma {v['panels_beyond_5sigma']}; "
              f"{v.get('arm_covariance_assumption', 'independent arms')}")
