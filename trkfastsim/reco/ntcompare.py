"""Summary and matched comparison of two EventNtuples (stock TTree or fast RNTuple), `trk`/`trksegs`/`trkhits` only.

The reader uses only the access forms that give the same awkward arrays on both layouts: `tree["evtinfo"].array()`,
`tree["trk"].array()` (a record of `trk.<leaf>` jagged fields) and `tree["<b>"].array()` for the doubly-jagged
branches. Tracks are matched inside common events by shared straw hits (plane, panel, layer, straw), mutual best
match with at least MIN_SHARED shared hits and at least half of the smaller hit list: an identity match, never by
distance or time. Every number is computed over the full arrays (no per-event loops).
"""
import json
import os

import awkward as ak
import numpy as np
import uproot

TREE = "EventNtuple/ntuple"
MIN_SHARED = 5
QS = (0.05, 0.16, 0.5, 0.84, 0.95)
SEG_Q = ("x", "y", "z", "time", "p", "momerr", "dmom")
HIT_Q = ("ptoca", "stoca", "rdoca", "rdt", "rdrift", "edep", "wdot", "udoca", "udocavar", "rdocavar", "rupos", "wdist")
TRK_DIFF = ("nhits", "nactive", "ndof", "nplanes", "npanels", "chisq", "fitcon", "firsthit", "lasthit", "avgedep",
            "nmat", "nmatactive", "radlen", "nnullambig")
PAR_SID = 4                         # KinematicLine parameters compared at the earliest TT_Outer crossing
PAR_Q = ("d0", "phi0", "z0", "theta", "t0", "mom", "d0err", "phi0err", "z0err", "thetaerr", "t0err", "momerr")


def load(path, tree=TREE):
    t = uproot.open(path)[tree]
    top = set(t.keys(recursive=False))
    ev = t["evtinfo"].array()
    out = {"path": path, "layout": type(t).__name__, "n": int(t.num_entries),
           "keys": np.stack([ak.to_numpy(ev[k]).astype(np.int64) for k in ("run", "subrun", "event")], 1)
           if t.num_entries else np.zeros((0, 3), np.int64)}
    out["evtinfo"] = ev
    from .ntuple import SPLIT_ALIAS, without_alias
    trk = t["trk"].array() if "trk" in top else None
    out["split_alias"] = trk is not None and SPLIT_ALIAS in trk.fields
    out["trk"] = without_alias(trk)
    for b in ("trksegs", "trkhits", "trksegpars_kl", "trkmats"):
        out[b] = t[b].array() if b in top else None
    return out


def _q(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {"n": int(len(a)), "mean": float(a.mean()), "q": [float(x) for x in np.quantile(a, QS)]}


def _counts(a):
    u, c = np.unique(np.asarray(a), return_counts=True)
    return {str(k.item() if hasattr(k, "item") else k): int(v) for k, v in zip(u, c)}


def _ntrk(D):
    if D["trk"] is None:
        return np.zeros(D["n"], np.int64)
    f = D["trk"].fields[0]
    return ak.to_numpy(ak.num(D["trk"][f], axis=1)).astype(np.int64)


def _seg_flat(S):
    """Flat per-crossing numpy columns with the owning flat-track index."""
    nseg = ak.flatten(ak.num(S, axis=2), axis=None)
    trk = np.repeat(np.arange(len(nseg)), ak.to_numpy(nseg))
    f = ak.flatten(ak.flatten(S, axis=1), axis=1)
    pos, mom = f["pos"]["fCoordinates"], f["mom"]["fCoordinates"]
    c = {"trk": trk, "sid": ak.to_numpy(f["sid"]), "sindex": ak.to_numpy(f["sindex"]),
         "inbounds": ak.to_numpy(f["inbounds"]).astype(bool), "time": ak.to_numpy(f["time"]),
         "x": ak.to_numpy(pos["fX"]), "y": ak.to_numpy(pos["fY"]), "z": ak.to_numpy(pos["fZ"]),
         "px": ak.to_numpy(mom["fX"]), "py": ak.to_numpy(mom["fY"]), "pz": ak.to_numpy(mom["fZ"]),
         "momerr": ak.to_numpy(f["momerr"]), "dmom": ak.to_numpy(f["dmom"])}
    c["p"] = np.sqrt(c["px"].astype(np.float64) ** 2 + c["py"] ** 2 + c["pz"] ** 2)
    return c


def _patterns(c, ntrack, sids=(0, 1, 2, 3, 4)):
    """Per track, the time-ordered sequence of tracker-surface crossings (e.g. "4-4", "4-2"): counts of the 12 most
    common sequences."""
    m = np.isin(c["sid"], sids)
    o = np.lexsort((c["time"][m], c["trk"][m]))
    tr, sd = c["trk"][m][o], c["sid"][m][o]
    n = np.bincount(tr, minlength=ntrack)
    K = min(int(n.max()) if len(n) else 0, 15)                         # first 15 crossings per track
    code = np.zeros(ntrack, np.int64)
    pos = np.arange(len(tr)) - np.repeat(np.cumsum(n) - n, n)
    keep = pos < K
    np.add.at(code, tr[keep], (sd[keep].astype(np.int64) + 1) * 10 ** (K - 1 - pos[keep]))
    u, cnt = np.unique(code, return_counts=True)
    top = np.argsort(-cnt, kind="stable")[:12]
    name = lambda v: "-".join(str(int(ch) - 1) for ch in str(v).rstrip("0")) if v else "none"
    return {name(u[i]): int(cnt[i]) for i in top}


def summary(D):
    nt = _ntrk(D)
    res = {"layout": D["layout"], "split_alias": D.get("split_alias", False), "events": D["n"], "tracks": int(nt.sum()),
           "events_with_track": int((nt > 0).sum()),
           "tracks_per_event": _counts(np.minimum(nt, 3))}
    if D["trk"] is not None and nt.sum():
        tr = {}
        for f in D["trk"].fields:
            v = ak.to_numpy(ak.flatten(D["trk"][f]))
            tr[f] = _counts(v) if (v.dtype.kind in "biu" and len(np.unique(v)) <= 12) else _q(v)
        res["trk"] = tr
    if D["trksegs"] is not None and nt.sum():
        c = _seg_flat(D["trksegs"])
        segs = {"per_track": _q(np.bincount(c["trk"], minlength=int(nt.sum())))}
        for s in np.unique(c["sid"]):
            m = c["sid"] == s
            segs[str(int(s))] = {"n": int(m.sum()), "inbounds": int(c["inbounds"][m].sum()),
                                 "sindex": _counts(c["sindex"][m]), "r": _q(np.hypot(c["x"][m], c["y"][m])),
                                 **{k: _q(c[k][m]) for k in SEG_Q}}
        segs["tracker_pattern"] = _patterns(c, int(nt.sum()))
        res["trksegs"] = segs
    if D["trkhits"] is not None and nt.sum():
        h = ak.flatten(D["trkhits"], axis=1)
        hf = ak.flatten(h)
        res["trkhits"] = {"per_track": _q(ak.to_numpy(ak.num(h, axis=1))), "state": _counts(ak.to_numpy(hf["state"])),
                          "algo": _counts(ak.to_numpy(hf["algo"])),
                          **{k: _q(ak.to_numpy(hf[k])) for k in HIT_Q if k in hf.fields}}
    if D["trksegpars_kl"] is not None and nt.sum():
        pf = ak.flatten(ak.flatten(D["trksegpars_kl"], axis=1), axis=1)
        res["trksegpars_kl"] = {k: _q(ak.to_numpy(pf[k])) for k in PAR_Q if k in pf.fields}
    if D["trkmats"] is not None and nt.sum():
        mt = ak.flatten(D["trkmats"], axis=1)
        res["trkmats"] = {"per_track": _q(ak.to_numpy(ak.num(mt, axis=1))),
                          "active_frac": float(ak.mean(ak.flatten(mt["active"]))) if ak.sum(ak.num(mt, axis=1)) else None}
    return res


def match_events(ka, kb):
    """Entry indices of the events present in both, in A's order (same rule as the crvextrap reader)."""
    _, inv = np.unique(np.concatenate([ka, kb]), axis=0, return_inverse=True)
    inv = inv.ravel()
    _, ia, ib = np.intersect1d(inv[:len(ka)], inv[len(ka):], return_indices=True)
    o = np.argsort(ia)
    return ia[o], ib[o]


def _hit_keys(D, ev_index):
    """Per hit of the selected events: (event position in the common list, flat-track index, straw key)."""
    H = D["trkhits"][ev_index]
    nt = ak.to_numpy(ak.num(H, axis=1))
    tr0 = np.concatenate([[0], np.cumsum(nt)])
    hpt = ak.to_numpy(ak.flatten(ak.num(H, axis=2)))
    trk = np.repeat(np.arange(int(nt.sum())), hpt)
    evp = np.repeat(np.repeat(np.arange(len(nt)), nt), hpt)
    f = ak.flatten(ak.flatten(H, axis=1), axis=1)
    sk = ((ak.to_numpy(f["plane"]).astype(np.int64) * 8 + ak.to_numpy(f["panel"])) * 2 + ak.to_numpy(f["layer"])) * 128 \
        + ak.to_numpy(f["straw"])
    return evp, trk, sk, tr0, hpt


def match_tracks(A, B, ia, ib):
    """Mutual-best shared-hit track pairs in the common events. Returns flat-track indices into A[ia] and B[ib]."""
    ea, ta, sa, _, na = _hit_keys(A, ia)
    eb, tb, sb, _, nb = _hit_keys(B, ib)
    ka = np.unique(np.stack([ea, ta, sa], 1), axis=0)              # one entry per (track, straw)
    kb = np.unique(np.stack([eb, tb, sb], 1), axis=0)
    K = sa.max(initial=0) + sb.max(initial=0) + 1
    ga, gb = ka[:, 0] * K + ka[:, 2], kb[:, 0] * K + kb[:, 2]
    ob = np.argsort(gb, kind="stable")
    gb_s, tb_s = gb[ob], kb[ob, 1]
    lo, hi = np.searchsorted(gb_s, ga, "left"), np.searchsorted(gb_s, ga, "right")
    rep = hi - lo
    pa = np.repeat(ka[:, 1], rep)
    pb = tb_s[np.repeat(lo, rep) + (np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep))]
    if not len(pa):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    pair, shared = np.unique(np.stack([pa, pb], 1), axis=0, return_counts=True)
    # mutual best: best B for each A and best A for each B by shared count (ties: lower index)
    oa = np.lexsort((pair[:, 1], -shared, pair[:, 0]))
    besta = oa[np.r_[True, pair[oa, 0][1:] != pair[oa, 0][:-1]]]
    ob2 = np.lexsort((pair[:, 0], -shared, pair[:, 1]))
    bestb = ob2[np.r_[True, pair[ob2, 1][1:] != pair[ob2, 1][:-1]]]
    m = np.intersect1d(besta, bestb)
    nmin = np.minimum(np.bincount(ta, minlength=len(na))[pair[m, 0]] if len(na) else 0,
                      np.bincount(tb, minlength=len(nb))[pair[m, 1]] if len(nb) else 0)
    ok = (shared[m] >= MIN_SHARED) & (2 * shared[m] >= nmin)
    m = m[ok]
    return pair[m, 0], pair[m, 1], shared[m]


def _flat_trk(D, ev_index, field):
    return ak.to_numpy(ak.flatten(D["trk"]["trk." + field][ev_index]))


def _ref_seg(D, ev_index, sid):
    """Per flat track of the selected events: the earliest crossing of `sid` (NaN where none), with the KinematicLine
    parameters of the same crossing (trksegpars_kl is parallel to trksegs)."""
    S = D["trksegs"][ev_index]
    nt = int(ak.sum(ak.num(S, axis=1)))
    c = _seg_flat(S)
    m = c["sid"] == sid
    out = {k: np.full(nt, np.nan) for k in ("x", "y", "z", "time", "p", "ux", "uy", "uz") + PAR_Q}
    if m.any():
        o = np.lexsort((c["time"][m], c["trk"][m]))
        tr = c["trk"][m][o]
        first = np.r_[True, tr[1:] != tr[:-1]]
        idx = np.flatnonzero(m)[o][first]
        for k in ("x", "y", "z", "time", "p"):
            out[k][tr[first]] = c[k][idx]
        for k, kk in (("ux", "px"), ("uy", "py"), ("uz", "pz")):
            out[k][tr[first]] = c[kk][idx] / np.where(c["p"][idx] > 0, c["p"][idx], np.nan)
        if D["trksegpars_kl"] is not None:
            P = ak.flatten(ak.flatten(D["trksegpars_kl"][ev_index], axis=1), axis=1)
            for k in PAR_Q:
                if k in P.fields:
                    out[k][tr[first]] = ak.to_numpy(P[k])[idx]
    return out


HIT_CMP = ("state", "ptoca", "stoca", "rdt", "rdoca", "rdrift", "rupos", "wdot", "edep", "rdocavar", "tottdrift", "wdist",
           "dvel", "sderr", "uderr", "werr")


def _hit_cols(D, ev_index, tracks):
    """Flat hit columns of the selected flat tracks: (owner position in `tracks`, straw key, fields)."""
    H = ak.flatten(D["trkhits"][ev_index], axis=1)[tracks]
    n = ak.to_numpy(ak.num(H, axis=1))
    f = ak.flatten(H)
    own = np.repeat(np.arange(len(tracks)), n)
    sk = ((ak.to_numpy(f["plane"]).astype(np.int64) * 8 + ak.to_numpy(f["panel"])) * 2 + ak.to_numpy(f["layer"])) * 128 \
        + ak.to_numpy(f["straw"])
    cols = {k: ak.to_numpy(f[k]).astype(np.float64) for k in HIT_CMP if k in f.fields}
    pc = f["poca"]["fCoordinates"]
    cols["poca"] = np.stack([ak.to_numpy(pc[k]) for k in ("fX", "fY", "fZ")], 1).astype(np.float64)
    return own, sk, cols


def compare_hits(A, B, ia, ib, ta, tb):
    """Hits of matched tracks paired by straw (first hit per straw and track): per-field B - A."""
    oa, ka, ca = _hit_cols(A, ia, ta)
    ob, kb, cb = _hit_cols(B, ib, tb)
    K = 1 << 20
    ga, gb = oa * K + ka, ob * K + kb
    ua, ja = np.unique(ga, return_index=True)
    ub, jb = np.unique(gb, return_index=True)
    _, xa, xb = np.intersect1d(ua, ub, return_indices=True)
    pa, pb = ja[xa], jb[xb]
    res = {"A_hits": int(len(ga)), "B_hits": int(len(gb)), "paired": int(len(pa))}
    for k in HIT_CMP:
        if k in ca and k in cb:
            va, vb = ca[k][pa], cb[k][pb]
            good = (va != -9999.0) & (vb != -9999.0)
            res[k] = {"B_minus_A": _q((vb - va)[good]), "identical_frac": float(np.mean(va == vb)) if len(va) else None,
                      "B_sentinel_frac": float(np.mean(vb == -9999.0)) if len(vb) else None}
    if len(pa):
        st_a, st_b = ca["state"][pa], cb["state"][pb]
        res["state_AB"] = {f"{int(x)}->{int(y)}": int(c) for (x, y), c in
                           zip(*np.unique(np.stack([st_a, st_b], 1), axis=0, return_counts=True))}
        both = (st_a > -2) & (st_b > -2)
        res["active_both"] = int(both.sum())
        res["poca_dist_mm_active_both"] = _q(np.linalg.norm(cb["poca"][pb] - ca["poca"][pa], axis=1)[both])
        for k in ("rdoca", "rdt", "ptoca"):
            if k in ca and k in cb:
                res[k + "_active_both"] = _q((cb[k][pb] - ca[k][pa])[both])
    return res


def compare(A, B, ref_sids=(0, 1, 2, 3, 4)):
    """A = reference (stock), B = candidate (fast). Waterfall, track matching and matched-track differences."""
    from .ntschema import signature
    res0 = {}
    for br in ("evtinfo", "trk", "trksegs", "trkhits", "trksegpars_kl", "trkmats"):
        xa, xb = A.get(br), B.get(br)
        res0[br] = None if xa is None or xb is None else bool(signature(xa.layout.form.to_dict()) ==
                                                               signature(xb.layout.form.to_dict()))
    ia, ib = match_events(A["keys"], B["keys"])
    na, nb = _ntrk(A), _ntrk(B)
    res = {"schema_identical": res0,
           "events": {"A": A["n"], "B": B["n"], "common": int(len(ia)), "A_only": int(A["n"] - len(ia)),
                      "B_only": int(B["n"] - len(ib))}}
    ca, cb = np.minimum(na[ia], 2), np.minimum(nb[ib], 2)
    res["ntrk_contingency_A_rows_B_cols_0_1_2plus"] = [[int(((ca == i) & (cb == j)).sum()) for j in range(3)] for i in range(3)]
    res["tracks_in_common_events"] = {"A": int(na[ia].sum()), "B": int(nb[ib].sum())}
    if A["trkhits"] is None or B["trkhits"] is None or not len(ia):
        return res
    ta, tb, shared = match_tracks(A, B, ia, ib)
    nA, nB = int(na[ia].sum()), int(nb[ib].sum())
    res["track_match"] = {"matched": int(len(ta)), "A_unmatched": nA - len(ta), "B_unmatched": nB - len(tb),
                          "shared_hits": _q(shared)}
    good_a, good_b = _flat_trk(A, ia, "goodfit"), _flat_trk(B, ib, "goodfit")
    res["track_match"]["matched_goodfit_AB"] = {f"{x}{y}": int(((good_a[ta] == x) & (good_b[tb] == y)).sum())
                                                for x in (0, 1) for y in (0, 1)}
    ua = np.setdiff1d(np.arange(nA), ta)
    ub = np.setdiff1d(np.arange(nB), tb)
    res["unmatched"] = {"A_nhits": _q(_flat_trk(A, ia, "nhits")[ua]), "B_nhits": _q(_flat_trk(B, ib, "nhits")[ub]),
                        "A_goodfit": _counts(good_a[ua]), "B_goodfit": _counts(good_b[ub])}
    d = {}
    for f in TRK_DIFF:
        if "trk." + f in A["trk"].fields and "trk." + f in B["trk"].fields:
            va, vb = _flat_trk(A, ia, f)[ta].astype(np.float64), _flat_trk(B, ib, f)[tb].astype(np.float64)
            d[f] = {"A": _q(va), "B": _q(vb), "B_minus_A": _q(vb - va),
                    "identical_frac": float(np.mean(va == vb)) if len(va) else None}
    res["matched_trk"] = d
    res["matched_hits"] = compare_hits(A, B, ia, ib, ta, tb)
    if A["trksegpars_kl"] is not None and B["trksegpars_kl"] is not None:
        pa, pb = _ref_seg(A, ia, PAR_SID), _ref_seg(B, ib, PAR_SID)
        pd = {}
        for k in ("d0", "phi0", "z0", "theta", "t0", "mom"):
            dv = pb[k][tb] - pa[k][ta]
            if k == "phi0":
                dv = (dv + np.pi) % (2 * np.pi) - np.pi
            e = pa[k + "err"][ta]
            pd[k] = {"A": _q(pa[k][ta]), "B": _q(pb[k][tb]), "B_minus_A": _q(dv),
                     "pull_by_A_err": _q(dv / np.where(e > 0, e, np.nan))}
        res["matched_pars_at_TT_Outer_entry"] = pd
    if A["trksegs"] is not None and B["trksegs"] is not None:
        sd = {}
        for s in ref_sids:
            ra, rb = _ref_seg(A, ia, s), _ref_seg(B, ib, s)
            both = np.isfinite(ra["x"][ta]) & np.isfinite(rb["x"][tb])
            ent = {"A_has": int(np.isfinite(ra["x"][ta]).sum()), "B_has": int(np.isfinite(rb["x"][tb]).sum()),
                   "both": int(both.sum())}
            if both.any():
                for k in ("x", "y", "z", "time", "p"):
                    ent["d" + k] = _q((rb[k][tb] - ra[k][ta])[both])
                cosang = np.clip(sum(ra[k][ta] * rb[k][tb] for k in ("ux", "uy", "uz")), -1, 1)[both]
                ent["angle_mrad"] = _q(1e3 * np.arccos(cosang))
            sd[str(s)] = ent
        res["matched_ref_segment"] = sd
    return res


def reader_check(path):
    """Run the analysis-side readers unmodified (when importable) and report what each returns or why it fails."""
    out = {}
    try:
        from crvextrap import io as XIO
        t = XIO.open_tree(path)
        k = XIO.event_keys(t)
        segs = XIO.load_branch([path], "trksegs")
        fx = XIO.flat_xings(segs)
        out["crvextrap.io"] = {"event_keys": int(len(k)), "flat_xings_tracks": int(fx.ntracks), "crossings": int(len(fx.sid))}
        try:
            q = XIO.load_quality([path])
            out["crvextrap.io"]["load_quality_tracks"] = int(ak.sum(ak.num(q["status"], axis=1)))
        except Exception as e:                       # noqa: BLE001 -- reported, not hidden
            out["crvextrap.io"]["load_quality"] = f"FAILS: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    except ImportError as e:
        out["crvextrap.io"] = f"not importable: {e}"
    try:
        from pyutils.pyimport import Importer
        imp = Importer(path, ["evtinfo", "trk", "trksegs", "trkhits"], verbosity=0)
        d = imp.import_branches()
        out["pyutils.Importer"] = {"fields": list(d.fields), "events": int(len(d)),
                                   "tracks": int(ak.sum(ak.num(d["trk"]["trk.nactive"], axis=1)))}
    except ImportError as e:
        out["pyutils.Importer"] = f"not importable: {e}"
    except Exception as e:                           # noqa: BLE001
        out["pyutils.Importer"] = f"FAILS: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    return out


def main(inputs, compare_to=None, outdir="plots/reco-ntcompare", readers=True):
    os.makedirs(outdir, exist_ok=True)
    B = load(inputs[0])
    res = {"B": {"path": inputs[0], "summary": summary(B)}}
    if compare_to:
        A = load(compare_to)
        res["A"] = {"path": compare_to, "summary": summary(A)}
        res["compare"] = compare(A, B)
    if readers:
        res["readers"] = {"B": reader_check(inputs[0])}
        if compare_to:
            res["readers"]["A"] = reader_check(compare_to)
    with open(os.path.join(outdir, "ntcompare.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res.get("compare", res["B"]["summary"]), indent=1)[:20000])
    return res


def report(res, out=print):
    """Readable digest of an ntcompare.json: waterfall, matching, matched-track and reference-segment differences."""
    c = res.get("compare")
    f3 = lambda q: "-" if not q else f"{q['q'][2]:+.4g} [{q['q'][1]:+.4g}, {q['q'][3]:+.4g}]"
    out(f"readers: {json.dumps(res.get('readers', {}))}")
    if not c:
        return
    out(f"schema identical to A: {c.get('schema_identical')}")
    out(f"events {c['events']}; tracks in common events {c['tracks_in_common_events']}")
    out(f"ntrk contingency (A rows 0/1/2+, B cols): {c['ntrk_contingency_A_rows_B_cols_0_1_2plus']}")
    if "track_match" not in c:
        return
    m = c["track_match"]
    out(f"matched {m['matched']}  A_unmatched {m['A_unmatched']}  B_unmatched {m['B_unmatched']}  "
        f"shared hits q50 {m['shared_hits']['q'][2] if m['shared_hits'] else '-'}; goodfit AB {m['matched_goodfit_AB']}")
    out(f"unmatched nhits A {f3(c['unmatched']['A_nhits'])}  B {f3(c['unmatched']['B_nhits'])}")
    out("matched trk (B - A: median [q16, q84]; identical fraction; A median, B median):")
    for k, v in c["matched_trk"].items():
        out(f"  {k:12s} {f3(v['B_minus_A']):40s} same {v['identical_frac']:.3f}  A {v['A']['q'][2] if v['A'] else '-':.6g}"
            f"  B {v['B']['q'][2] if v['B'] else '-':.6g}" if v["A"] and v["B"] else f"  {k:12s} -")
    h = c.get("matched_hits")
    if h:
        out(f"matched hits: A {h['A_hits']} B {h['B_hits']} paired {h['paired']} active in both {h.get('active_both')}")
        out(f"  state A->B {h.get('state_AB')}")
        for k in HIT_CMP:
            if k in h:
                out(f"  {k:10s} {f3(h[k]['B_minus_A']):40s} same {h[k]['identical_frac']:.3f}  B sentinel {h[k]['B_sentinel_frac']:.3f}")
        for k in ("rdoca", "rdt", "ptoca"):
            if k + "_active_both" in h:
                out(f"  {k:10s} (active in both) {f3(h[k + '_active_both'])}")
        out(f"  |poca B - A| (active in both) {f3(h.get('poca_dist_mm_active_both'))}")
    out("matched KinematicLine at the TT_Outer entry (B - A; pull by A error):")
    for k, v in c.get("matched_pars_at_TT_Outer_entry", {}).items():
        out(f"  {k:6s} {f3(v['B_minus_A']):40s} pull {f3(v['pull_by_A_err'])}")
    out("matched reference segments (earliest crossing per sid):")
    for s, v in c.get("matched_ref_segment", {}).items():
        out(f"  sid {s}: A {v['A_has']} B {v['B_has']} both {v['both']}  " +
            "  ".join(f"{k} {f3(v.get(k))}" for k in ("dx", "dy", "dz", "dtime", "dp", "angle_mrad")))
