"""JAX twin of linefind.line_seeds (LineFinder with NSteps = NTSteps = 0): clusters padded to hit-count buckets, all
candidate hit pairs of a bucket evaluated at once, the stock scan order (i ascending, j descending), early stop and
MaxPairs cap reproduced by ranking the valid pairs per cluster. One compiled shape per bucket; the numpy port is the
reference and handles clusters above the largest bucket.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import linefind
from .hitreco import unique_straw
from .hitreco_jax import bucket

jax.config.update("jax_enable_x64", True)

L_BUCKETS = (16, 32, 64, 128, 256)


@functools.lru_cache(maxsize=None)
def _candidate_pairs(nmin, L, maxpairs):
    """Union over n in [nmin, L] of the first maxpairs+1 pairs in the stock order; sorted by that order (i, -j)."""
    keep = set()
    for n in range(max(nmin, 2), L + 1):
        i, j = linefind._pairs(n, maxpairs)
        keep.update(zip(i.tolist(), j.tolist()))
    pi, pj = map(np.array, zip(*sorted(keep, key=lambda p: (p[0], -p[1])))) if keep else (np.zeros(0, int),) * 2
    return pi.astype(np.int64), pj.astype(np.int64)


def _pca(mid, d, p2, t2, cut=1e-8):
    t1 = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    t2 = t2 / jnp.linalg.norm(t2, axis=-1, keepdims=True)
    c = jnp.sum(t1 * t2, -1)
    sinsq = 1.0 - c * c
    par = sinsq < cut
    delta = mid - p2
    d1, d2 = jnp.sum(delta * t1, -1), jnp.sum(delta * t2, -1)
    ss = jnp.where(par, 1.0, sinsq)
    s1 = jnp.where(par, 0.0, (d2 * c - d1) / ss)
    s2 = jnp.where(par, 0.0, -(d1 * c - d2) / ss)
    pca1 = mid + t1 * s1[..., None]
    pca2 = p2 + t2 * s2[..., None]
    dca = jnp.linalg.norm(pca1 - pca2, axis=-1)
    return jnp.sum((pca1 - mid) * d, -1), dca, pca2


@functools.partial(jax.jit, static_argnames=("maxpairs",))
def _seed_bucket(P, mid, sdir, hlen, wd, wv, tcorr, n, pi, pj, maxdoca, maxpairs):
    """P, mid, sdir [C, L, 3]; hlen, wd, wv, tcorr [C, L]; n [C]; pi, pj [Np] candidate pairs in stock order."""
    C, L = hlen.shape
    hvalid = jnp.arange(L)[None, :] < n[:, None]                               # [C, L]
    pvalid = (pi[None, :] < n[:, None]) & (pj[None, :] < n[:, None])           # [C, Np]
    rank = jnp.cumsum(pvalid, 1) - 1
    pvalid = pvalid & (rank <= maxpairs)
    ipos, jpos = P[:, pi], P[:, pj]                                            # [C, Np, 3]
    ndir = jpos - ipos
    ndir = ndir / jnp.linalg.norm(ndir, axis=-1, keepdims=True)
    dist, dca, _ = _pca(mid[:, None], sdir[:, None], ipos[:, :, None], ndir[:, :, None])      # [C, Np, L]
    inl = (dca < maxdoca) & (jnp.abs(dist) < hlen[:, None]) & hvalid[:, None]
    count = jnp.where(pvalid, inl.sum(-1), -1)
    ll = jnp.sum(jnp.where(inl, (dist - wd[:, None]) ** 2 / wv[:, None], 0.0), -1)
    full = count == n[:, None]
    first_full = jnp.where(full.any(1), jnp.argmax(full, 1), count.shape[1] - 1)
    inscan = jnp.arange(count.shape[1])[None, :] <= first_full[:, None]
    cnt = jnp.where(inscan, count, -1)
    best_c = cnt.max(1)
    cand = inscan & (cnt == best_c[:, None]) & (best_c[:, None] > 0)
    best = jnp.argmin(jnp.where(cand, ll, jnp.inf), 1)
    has = best_c > 0
    r = jnp.arange(C)
    ip, nd = ipos[r, best], ndir[r, best]
    nd = jnp.where((nd[:, 1] > 0)[:, None], -nd, nd)
    ny = jnp.where(nd[:, 1] != 0, nd[:, 1], 1.0)
    sint = ip - nd * (ip[:, 1] / ny)[:, None]
    sdn = nd / (-nd[:, 1:2])
    sint = sint - sdn * (sint[:, 1] / jnp.where(sdn[:, 1] != 0, sdn[:, 1], 1.0))[:, None]
    dist2, dca2, pca2 = _pca(mid, sdir, sint[:, None], sdn[:, None])            # [C, L]
    ok = (dca2 < maxdoca) & (jnp.abs(dist2) < hlen) & has[:, None] & hvalid
    u = sdn / jnp.linalg.norm(sdn, axis=-1, keepdims=True)
    ttraj = jnp.sum((pca2 - sint[:, None]) * u[:, None], -1) / linefind.C_LIGHT
    ng = ok.sum(1)
    t0 = jnp.sum(jnp.where(ok, tcorr - ttraj, 0.0), 1) / jnp.maximum(ng, 1)
    return t0, sint[:, 0], sint[:, 2], sdn[:, 0], sdn[:, 2], ng, ok


def line_seeds(sh, tc, cond, s=linefind.LF_SETTINGS, budget=4e7):
    """Same outputs as linefind.line_seeds; clusters with more hits than the largest bucket go through numpy."""
    st = cond["straw"]
    us = unique_straw(sh["sid"].astype(np.int64))
    pos = np.stack([sh["x"], sh["y"], sh["z"]], 1).astype(np.float64)
    mid = np.stack([st["midx"][us], st["midy"][us], st["midz"][us]], 1)
    sdir = np.stack([st["dirx"][us], st["diry"][us], st["dirz"][us]], 1)
    hlen = st["hlen"][us]
    wd, wv = sh["wdist"].astype(np.float64), sh["uvar"].astype(np.float64)
    tearly = np.where(sh["eend"] == 1, sh["et1"], sh["et0"]).astype(np.float32)
    tcorr = (tearly - sh["dtime"].astype(np.float32) - sh["ptime"].astype(np.float32)).astype(np.float64)
    ncl = len(tc["n"])
    mstart = np.concatenate([[0], np.cumsum(tc["n"])[:-1]]).astype(np.int64)
    out = {k: np.zeros(ncl) for k in ("t0", "a0", "b0", "a1", "b1")}
    nseed = np.zeros(ncl, np.int64)
    seedmask = np.zeros(len(tc["member"]), bool)
    lo = 0
    for L in L_BUCKETS:
        cls = np.flatnonzero((tc["n"] > lo) & (tc["n"] <= L))
        nmin = lo + 1
        lo = L
        if not len(cls):
            continue
        pi, pj = _candidate_pairs(nmin, L, s["maxpairs"])
        chunk = int(max(64, min(1 << 14, budget // max(len(pi) * L, 1))))
        chunk = 1 << int(np.log2(chunk))
        for c0 in range(0, len(cls), chunk):
            cc = cls[c0:c0 + chunk]
            C = chunk if len(cls) > chunk else bucket(len(cc), 64)
            nn = np.zeros(C, np.int64); nn[:len(cc)] = tc["n"][cc]
            idx = np.zeros((C, L), np.int64)
            k = np.arange(L)[None, :]
            m = k < nn[:len(cc), None]
            idx[:len(cc)] = np.where(m, tc["member"][np.minimum(mstart[cc][:, None] + k, len(tc["member"]) - 1)], 0)
            g = lambda a: jnp.asarray(a[idx])
            res = _seed_bucket(g(pos), g(mid), g(sdir), g(hlen), g(wd), g(wv), g(tcorr), jnp.asarray(nn),
                               jnp.asarray(pi), jnp.asarray(pj), s["maxdoca"], s["maxpairs"])
            t0, a0, b0, a1, b1, ng, ok = (np.asarray(x)[:len(cc)] for x in res)
            out["t0"][cc] = t0 - s["t0offset"]
            out["a0"][cc], out["b0"][cc], out["a1"][cc], out["b1"][cc] = a0, b0, a1, b1
            nseed[cc] = ng
            rows, cols = np.nonzero(ok & m)
            seedmask[mstart[cc][rows] + cols] = True
    big = np.flatnonzero(tc["n"] > L_BUCKETS[-1])
    if len(big):   # the rare oversize clusters go through numpy, as a sub-collection of just those clusters
        mrows = np.concatenate([np.arange(mstart[b], mstart[b] + tc["n"][b]) for b in big])
        sub = {"ev": tc["ev"][big], "n": tc["n"][big], "member": tc["member"][mrows],
               "member_cl": np.repeat(np.arange(len(big)), tc["n"][big])}
        ref = linefind.line_seeds(sh, sub, cond, s)
        # numpy drops clusters below minPeak; refill only the kept ones (the others stay at nseed 0)
        kb = big[ref["tc"]]
        for k in ("t0", "a0", "b0", "a1", "b1"):
            out[k][kb] = ref[k]
        nseed[kb] = ref["n"]
        seedmask[mrows] = np.isin(tc["member"][mrows], ref["member"])
    keep = nseed >= s["minpeak"]
    ev0 = np.searchsorted(sh["ev"], sh["ev"], side="left")
    mi = tc["member"][seedmask & keep[tc["member_cl"]]]
    res = {k: v[keep] for k, v in out.items()}
    res.update(tc=np.flatnonzero(keep), ev=tc["ev"][keep], n=nseed[keep], member=mi, ind=mi - ev0[mi])
    return res
