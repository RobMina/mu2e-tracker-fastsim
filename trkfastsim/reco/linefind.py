"""Vectorised numpy ports of Offline SimpleTimeCluster and LineFinder (CosmicReco) on single-straw hits.

Reference: CosmicReco/src/{SimpleTimeCluster,LineFinder}_module.cc, CosmicReco/fcl/prolog.fcl settings, TwoLinePCA.
Inputs are hitreco.straw_hits columns (makeSH) and the hit-reco conditions (straw midpoints, directions, half lengths).
"""
import numpy as np

from .hitreco import FLAG, unique_straw

STC_SETTINGS = dict(minnsh=2, minnpanels=1, timewindow=100.0, usetimewindow=False, timestep=20.0, usetimestep=True)
LF_SETTINGS = dict(maxdoca=10.0, minpeak=10, maxpairs=500, t0offset=0.0)
_BKG = (1 << 6) | (1 << FLAG["noisy"]) | (1 << FLAG["dead"])   # HitBackgroundBits: Background, Noisy, Dead
C_LIGHT = 299.9                                                 # mm/ns, as LineFinder


def time_clusters(sh, s=STC_SETTINGS):
    """SimpleTimeCluster with useTimeWindow false: per event, good hits sorted by time split at gaps > TimeStep;
    a run of >= minNStrawHits hits spanning <= TimeWindow is a cluster. Returns cluster arrays and flat members
    (makeSH collection order inside each cluster, as the stock loop over the collection)."""
    good = (sh["flag"] & _BKG) == 0
    gi = np.flatnonzero(good)
    if len(gi) == 0:
        return {"ev": sh["ev"][:0], "n": gi.copy(), "t0": np.empty(0, np.float64),
                "t0err": np.empty(0, np.float64), "member": gi.copy(), "member_cl": gi.copy(), "ind": gi.copy()}
    ev, t = sh["ev"][gi], sh["time"][gi]
    o = np.lexsort((t, ev))
    gi, ev, t = gi[o], ev[o], t[o]
    tt = t.astype(np.float64)
    brk = np.concatenate([[True], (ev[1:] != ev[:-1]) | (s["usetimestep"] & ((tt[1:] - tt[:-1]) > s["timestep"]))])
    seg = np.cumsum(brk) - 1
    start = np.flatnonzero(brk)
    end = np.append(start[1:], len(gi)) - 1
    n = end - start + 1
    span = tt[end] - tt[start]
    keep = (n >= s["minnsh"]) & ~((not s["usetimewindow"]) & (span > s["timewindow"]))
    kseg = np.flatnonzero(keep)
    member = keep[seg]
    mi, ms = gi[member], seg[member]
    o2 = np.lexsort((mi, ms))                                       # collection order within each cluster
    mi, ms = mi[o2], ms[o2]
    newid = np.full(len(start), -1)
    newid[kseg] = np.arange(len(kseg))
    cid = newid[ms]
    ncl = np.bincount(cid, minlength=len(kseg))
    t0 = np.bincount(cid, sh["time"][mi].astype(np.float64), minlength=len(kseg)) / np.maximum(ncl, 1)
    ev0 = np.searchsorted(sh["ev"], sh["ev"], side="left")
    return {"ev": ev[start[kseg]], "n": ncl, "t0": t0, "t0err": span[kseg] / 2.0,
            "member": mi, "member_cl": cid, "ind": mi - ev0[mi]}


def _pairs(n, maxpairs):
    """LineFinder pair order (i ascending, j descending from n-1 to i+1), the first maxpairs+1 pairs (it stops after
    the pair whose running count exceeds MaxPairs)."""
    i, j = [], []
    for a in range(n):
        for b in range(n - 1, a, -1):
            i.append(a); j.append(b)
            if len(i) > maxpairs:
                return np.array(i), np.array(j)
    return np.array(i, dtype=np.int64), np.array(j, dtype=np.int64)


def _pca(mid, d, p2, t2, cut=1e-8):
    """TwoLinePCA of wire lines (mid, d) with track lines (p2, t2), broadcast; returns s1 (along the wire), dca, pca2."""
    t1 = d / np.linalg.norm(d, axis=-1, keepdims=True)
    t2 = t2 / np.linalg.norm(t2, axis=-1, keepdims=True)
    c = np.sum(t1 * t2, -1)
    sinsq = 1.0 - c * c
    par = sinsq < cut
    delta = mid - p2
    d1, d2 = np.sum(delta * t1, -1), np.sum(delta * t2, -1)
    ss = np.where(par, 1.0, sinsq)
    s1 = np.where(par, 0.0, (d2 * c - d1) / ss)
    s2 = np.where(par, 0.0, -(d1 * c - d2) / ss)
    pca1 = mid + t1 * s1[..., None]
    pca2 = p2 + t2 * s2[..., None]
    dca = np.linalg.norm(pca1 - pca2, axis=-1)
    dist = np.sum((pca1 - mid) * d, -1)
    return dist, dca, pca2


def line_seeds(sh, tc, cond, s=LF_SETTINGS, chunk=512):
    """LineFinder with NSteps = NTSteps = 0 (one line per hit pair): per time cluster, the first pair in scan order
    with the most hits within maxDOCA of their wire (and inside the straw), ties broken by the smaller wire-distance
    chi2; scan stops at the first pair that takes every hit or after MaxPairs. Seed if >= minPeak hits; t0 from them."""
    st = cond["straw"]
    us = unique_straw(sh["sid"].astype(np.int64))
    pos = np.stack([sh["x"], sh["y"], sh["z"]], 1).astype(np.float64)
    mid = np.stack([st["midx"][us], st["midy"][us], st["midz"][us]], 1)
    sdir = np.stack([st["dirx"][us], st["diry"][us], st["dirz"][us]], 1)
    hlen = st["hlen"][us]
    wd, wv = sh["wdist"].astype(np.float64), sh["uvar"].astype(np.float64)
    tearly = np.where(sh["eend"] == 1, sh["et1"], sh["et0"]).astype(np.float32)
    tcorr = (tearly - sh["dtime"].astype(np.float32) - sh["ptime"].astype(np.float32)).astype(np.float64)  # float32 ops, as stock
    ncl = len(tc["n"])
    mstart = np.concatenate([[0], np.cumsum(tc["n"])[:-1]])
    out = {k: np.zeros(ncl) for k in ("t0", "a0", "b0", "a1", "b1")}
    nseed = np.zeros(ncl, np.int64)
    seedmask = np.zeros(len(tc["member"]), bool)
    for n in np.unique(tc["n"]):
        cls = np.flatnonzero(tc["n"] == n)
        pi, pj = _pairs(int(n), s["maxpairs"])
        for c0 in range(0, len(cls), chunk):
            cc = cls[c0:c0 + chunk]
            hidx = tc["member"][mstart[cc][:, None] + np.arange(n)[None, :]]          # [C, n]
            P = pos[hidx]
            ipos, jpos = P[:, pi], P[:, pj]                                            # [C, Np, 3]
            ndir = jpos - ipos
            ndir = ndir / np.linalg.norm(ndir, axis=-1, keepdims=True)
            dist, dca, _ = _pca(mid[hidx][:, None], sdir[hidx][:, None], ipos[:, :, None], ndir[:, :, None])  # [C, Np, n]
            inl = (dca < s["maxdoca"]) & (np.abs(dist) < hlen[hidx][:, None])
            count = inl.sum(-1)
            ll = np.where(inl, (dist - wd[hidx][:, None]) ** 2 / wv[hidx][:, None], 0.0)
            ll = np.add.reduce(ll, -1)                                                 # stock adds in hit order
            full = count == n
            stop = np.where(full.any(1), np.argmax(full, 1), count.shape[1] - 1)
            valid = np.arange(count.shape[1])[None, :] <= stop[:, None]
            cnt = np.where(valid, count, -1)
            best_c = cnt.max(1)
            cand = valid & (cnt == best_c[:, None]) & (best_c[:, None] > 0)
            llc = np.where(cand, ll, np.inf)
            best = np.argmin(llc, 1)                                                   # first minimum = earliest
            has = best_c > 0
            r = np.arange(len(cc))
            ip, nd = ipos[r, best], ndir[r, best]
            nd = np.where((nd[:, 1] > 0)[:, None], -nd, nd)
            ny = np.where(nd[:, 1] != 0, nd[:, 1], 1.0)
            sint = ip - nd * (ip[:, 1] / ny)[:, None]
            sdirn = nd / (-nd[:, 1:2] * np.where(nd[:, 1:2] != 0, 1.0, np.nan))
            sint = sint - sdirn * (sint[:, 1] / np.where(sdirn[:, 1] != 0, sdirn[:, 1], 1.0))[:, None]
            dist2, dca2, pca2 = _pca(mid[hidx], sdir[hidx], sint[:, None], sdirn[:, None])        # [C, n]
            ok = (dca2 < s["maxdoca"]) & (np.abs(dist2) < hlen[hidx]) & has[:, None]
            u = sdirn / np.linalg.norm(sdirn, axis=-1, keepdims=True)
            ttraj = np.sum((pca2 - sint[:, None]) * u[:, None], -1) / C_LIGHT
            ht0 = tcorr[hidx] - ttraj
            ng = ok.sum(1)
            t0 = np.add.reduce(np.where(ok, ht0, 0.0), 1) / np.maximum(ng, 1) - s["t0offset"]
            out["t0"][cc], nseed[cc] = t0, ng
            out["a0"][cc], out["b0"][cc] = sint[:, 0], sint[:, 2]
            out["a1"][cc], out["b1"][cc] = sdirn[:, 0], sdirn[:, 2]
            seedmask[mstart[cc][:, None] + np.arange(n)[None, :]] = ok
    keep = nseed >= s["minpeak"]
    ev0 = np.searchsorted(sh["ev"], sh["ev"], side="left")
    mi = tc["member"][seedmask & keep[tc["member_cl"]]]
    res = {k: v[keep] for k, v in out.items()}
    res.update(tc=np.flatnonzero(keep), ev=tc["ev"][keep], n=nseed[keep], member=mi, ind=mi - ev0[mi])
    return res
