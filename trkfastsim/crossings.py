"""Vectorised crossings of straight tracks with finite straw cylinders: a coarse panel-slab cull, then ray-cylinder.

For a track o + s d (unit d) and a straw (mid m, unit axis a, radius r, half length h): with w = o - m,
u = d - (d.a) a, v = w - (w.a) a, the chord solves |v + s u|^2 = r^2; the chord is then clipped to |(o + s d - m).a| <= h.
Only straws of panels whose bounding box the track enters are tested, so the dense test is [tracks, 96] per panel.
JAX is not used yet (numpy throughout); the chunking keeps memory bounded.
"""
from __future__ import annotations

import numpy as np

from mu2efastsim import crossings as box


def panel_boxes(geom, pad=None):
    """Axis-aligned bounding box of each panel's straws (lo, hi [216, 3]), padded by the straw radius."""
    pad = geom.radius if pad is None else pad
    ends = np.concatenate([geom.mid - geom.dir * geom.halflen[:, None], geom.mid + geom.dir * geom.halflen[:, None]])
    up = np.concatenate([geom.upanel, geom.upanel])
    lo = np.full((up.max() + 1, 3), np.inf)
    hi = np.full((up.max() + 1, 3), -np.inf)
    np.minimum.at(lo, up, ends)
    np.maximum.at(hi, up, ends)
    return lo - pad, hi + pad


def straw_crossings(origin, direction, geom, chunk: int = 8192):
    """Rows (track, straw index, s_in, s_out) of every straw gas volume each track crosses, sorted by track, s_in."""
    lo, hi = panel_boxes(geom)
    straws_of = np.argsort(geom.upanel, kind="stable").reshape(lo.shape[0], -1)   # [216, 96] straw indices
    out = []
    for c0 in range(0, len(origin), chunk):
        o, d = origin[c0:c0 + chunk], direction[c0:c0 + chunk]
        pc = box.crossings(o, d, lo, hi, backend="numpy")          # (track, panel, ...)
        if not len(pc):
            continue
        it = np.repeat(pc[:, 0].astype(np.int64), straws_of.shape[1])
        js = straws_of[pc[:, 1].astype(np.int64)].ravel()
        w = o[it] - geom.mid[js]
        a = geom.dir[js]
        dd = d[it]
        da = np.einsum("ij,ij->i", dd, a)
        wa = np.einsum("ij,ij->i", w, a)
        u = dd - da[:, None] * a
        v = w - wa[:, None] * a
        A = np.einsum("ij,ij->i", u, u)
        B = 2 * np.einsum("ij,ij->i", u, v)
        C = np.einsum("ij,ij->i", v, v) - geom.radius ** 2
        disc = B * B - 4 * A * C
        par = A <= 1e-15                                   # parallel to the axis: inside the gas iff C <= 0
        ok = np.where(par, C <= 0, disc > 0)
        sq = np.sqrt(np.where(ok & ~par, disc, 0.0))
        A_ = np.where(par, 1.0, A)
        s0 = np.where(par, -np.inf, (-B - sq) / (2 * A_))
        s1 = np.where(par, np.inf, (-B + sq) / (2 * A_))
        # clip to the straw length: axial coordinate wa + s da within [-h, h]
        h = geom.halflen[js]
        with np.errstate(divide="ignore", invalid="ignore"):
            z0, z1 = (-h - wa) / da, (h - wa) / da
        zl = np.where(np.abs(da) > 1e-12, np.minimum(z0, z1), -np.inf)
        zh = np.where(np.abs(da) > 1e-12, np.maximum(z0, z1), np.inf)
        inside_len = (np.abs(da) > 1e-12) | (np.abs(wa) <= h)
        s_in, s_out = np.maximum(np.maximum(s0, zl), 0.0), np.minimum(s1, zh)
        keep = ok & inside_len & (s_out > s_in)
        out.append(np.stack([it[keep] + c0, js[keep], s_in[keep], s_out[keep]], 1))
    if not out:
        return np.zeros((0, 4))
    r = np.concatenate(out)
    return r[np.lexsort((r[:, 2], r[:, 0]))]


def point_straw(points, geom, chunk: int = 65536):
    """Straw index whose gas volume contains each point [N, 3] (Mu2e frame), or -1 (wall, gap, outside the tracker).

    Panel boxes cull first; within a candidate panel the 96 straws are tested (axis distance <= radius, |axial| <=
    half length). Panel boxes overlap only at their edges, so a point is tested against at most a few panels.
    """
    lo, hi = panel_boxes(geom)          # padded by the radius: the straw volumes, not just their axes
    straws_of = np.argsort(geom.upanel, kind="stable").reshape(lo.shape[0], -1)
    out = np.full(len(points), -1, dtype=np.int64)
    for c0 in range(0, len(points), chunk):
        x = points[c0:c0 + chunk]
        inbox = np.all((x[:, None, :] >= lo[None]) & (x[:, None, :] <= hi[None]), axis=2)   # [n, 216]
        ip, pp = np.nonzero(inbox)
        if not len(ip):
            continue
        ii = np.repeat(ip, straws_of.shape[1])
        js = straws_of[pp].ravel()
        w = x[ii] - geom.mid[js]
        wa = np.einsum("ij,ij->i", w, geom.dir[js])
        r2 = np.einsum("ij,ij->i", w, w) - wa * wa
        ok = (r2 <= geom.radius ** 2) & (np.abs(wa) <= geom.halflen[js])
        out[c0 + ii[ok]] = js[ok]
    return out


def nearest_straw(points, geom, max_dist=None, pad=10.0, chunk: int = 32768):
    """Index of the straw whose axis segment is nearest to each point [N, 3] (Mu2e frame), and that distance [mm].

    Candidates: straws of panels whose box (padded by `pad` beyond the straw volume) contains the point. Distance to
    the finite axis segment (axial coordinate clipped to the half length). Points with no candidate panel, or farther
    than max_dist (default: the straw radius + pad), get -1. Used to snap pasted secondaries onto the straw lattice,
    which a continuous frame transform cannot hit (5 mm straws, gaps between layers).
    """
    lo, hi = panel_boxes(geom, pad=geom.radius + pad)
    straws_of = np.argsort(geom.upanel, kind="stable").reshape(lo.shape[0], -1)
    max_dist = geom.radius + pad if max_dist is None else max_dist
    best = np.full(len(points), -1, dtype=np.int64)
    bestd = np.full(len(points), np.inf)
    for c0 in range(0, len(points), chunk):
        x = points[c0:c0 + chunk]
        inbox = np.all((x[:, None, :] >= lo[None]) & (x[:, None, :] <= hi[None]), axis=2)
        ip, pp = np.nonzero(inbox)
        if not len(ip):
            continue
        ii = np.repeat(ip, straws_of.shape[1])
        js = straws_of[pp].ravel()
        w = x[ii] - geom.mid[js]
        wa = np.clip(np.einsum("ij,ij->i", w, geom.dir[js]), -geom.halflen[js], geom.halflen[js])
        d = np.linalg.norm(w - wa[:, None] * geom.dir[js], axis=1)
        order = np.lexsort((d, ii))
        first = np.r_[True, ii[order][1:] != ii[order][:-1]]
        k = order[first]
        best[c0 + ii[k]] = js[k]
        bestd[c0 + ii[k]] = d[k]
    far = bestd > max_dist
    best[far] = -1
    return best, bestd
