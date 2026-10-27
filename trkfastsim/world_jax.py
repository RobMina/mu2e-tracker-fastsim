"""Device world column and transport to the tracker plane, using the core transport conventions.

Geometry packing and generator selection remain on the host. Convex clipping, ordered band/material loss,
air loss and correlated scattering run on the device. This is the legacy transport law, not a new ledger.
Random keys are indexed by muon, band, material and purpose; the numpy backend never imports this module.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mu2efastsim import eloss, eloss_jax, materials, transport as tr
from . import steps as st, steps_jax as sj

WORLD_LANES = 16384
KEY_WORLD = 17
PIECE_TILE = 32


def _fold(keys, value):
    return jax.vmap(lambda k: jax.random.fold_in(k, value))(keys)


def _p_after(p, loss):
    tk = jnp.sqrt(p * p + eloss.MMU ** 2) - eloss.MMU - loss
    return jnp.sqrt(jnp.maximum((jnp.maximum(tk, 0) + eloss.MMU) ** 2 - eloss.MMU ** 2, 0))


@jax.jit
def _band_segment(o, d, y_top, y_bot, y_plane, valid):
    L = (o[:, 1] - y_plane) / -d[:, 1]
    top = jnp.clip((o[:, 1] - y_top) / -d[:, 1], 0, L)
    bot = jnp.clip((o[:, 1] - y_bot) / -d[:, 1], 0, L)
    seg = jnp.where(valid, bot - top, 0.0)
    return o + d * top[:, None], seg, L - 0.5 * (top + bot)


@jax.jit
def _max_hits(lo, hi, o, d, L, valid):
    return jnp.max(sj._box_hits(lo, hi, o, d, L, valid).sum(1), initial=0)


@partial(jax.jit, static_argnames=("nm",))
def _tiled_column(lo, hi, planes, mat, o, d, length, valid, nm):
    """Clip every piece in bounded tiles; int64 column sums are exact across tile boundaries."""
    ntiles = lo.shape[0]

    def tile(i, acc):
        pp = planes[i]
        hit = sj._box_hits(lo[i], hi[i], o, d, length, valid)
        den = jnp.einsum("bj,pkj->bpk", d, pp[..., :3])
        num = pp[None, ..., 3] - jnp.einsum("bj,pkj->bpk", o, pp[..., :3])
        ss = num / jnp.where(den == 0, 1.0, den)
        enter = jnp.max(jnp.where(den < 0, ss, -jnp.inf), -1)
        leave = jnp.min(jnp.where(den > 0, ss, jnp.inf), -1)
        outside = jnp.any((den == 0) & (num < 0), -1)
        ll = jnp.where(hit & ~outside, jnp.maximum(jnp.minimum(leave, length[:, None]) - jnp.maximum(enter, 0), 0), 0.0)
        quant = sj._q(ll, 0)
        onehot = mat[i, :, None] == jnp.arange(nm)[None]
        return acc + jnp.sum(jnp.where(onehot[None], quant[..., None], 0), axis=1)

    return lax.fori_loop(0, ntiles, tile, jnp.zeros((len(o), nm), jnp.int64))


@jax.jit
def _count_material(col, m):
    return jnp.sum(col[:, m] > 0)


@partial(jax.jit, static_argnames=("C",))
def _material_pairs(col, m, p, lost, keys, C):
    hit = col[:, m] > 0
    idx = jnp.nonzero(hit, size=C, fill_value=0)[0]
    valid = jnp.arange(C) < hit.sum()
    length = jnp.where(valid, col[idx, m], 0.0)
    return idx, valid, length, _p_after(p[idx], lost[idx]), _fold(keys[idx], m)


@jax.jit
def _material_add(lost, idx, valid, dl):
    return lost.at[jnp.where(valid, idx, len(lost))].add(dl, mode="drop")


@jax.jit
def _band_finish(MC, coeff, col, seg, p, before, lost):
    air = jnp.maximum(seg - col.sum(1), 0.0)
    ca = eloss_jax.MatConsts(*(f[-1] for f in MC))
    lost = lost + eloss_jax._dedx(ca, jnp.maximum(_p_after(p, lost), 1.0), 1e9, eloss.MMU) * air
    columns = jnp.concatenate([col, air[:, None]], 1)
    xx0 = jnp.sum(columns * coeff[0], 1)
    bp = jnp.where(seg > 0, 0.5 * (_p_after(p, before) + _p_after(p, lost)), 0.0)
    pm = jnp.maximum(bp, 1.0)
    pb2 = (pm * pm / jnp.sqrt(pm * pm + eloss.MMU ** 2)) ** 2
    x2 = columns * coeff[1] / pb2[:, None]
    chi = x2.sum(1)
    thw = (x2 * coeff[2] / pm[:, None]).sum(1)
    thn = jnp.where(chi > 0, thw / jnp.maximum(chi, 1e-300), 1.0)
    return lost, xx0, bp, chi, thn


def _compact_band_loss(MC, coeff, col, seg, p, lost, keys):
    """Sequential materials with active slice compaction and unchanged per-muon/material/slice keys."""
    before = lost
    for m in range(col.shape[1]):
        count = int(_count_material(col, m))
        if not count:
            continue
        C = sj._bucket(count)
        idx, valid, length, pp, mk = _material_pairs(col, m, p, lost, keys, C)
        mat = jnp.full(C, m, jnp.int32)
        carry = (jnp.zeros(C, jnp.int32), length, jnp.zeros(C))
        work = jnp.arange(C)
        while True:
            carry, todo, nt = sj._slab_iter(MC, mat, pp, length, mk, carry, work, sj.SLICE_ROUNDS)
            nt = int(nt)
            if not nt:
                break
            work = sj._slab_next(work, todo, sj._bucket(nt))
        dl = sj._slab_finish(MC, mat, pp, carry)
        lost = _material_add(lost, idx, valid, dl)
    return _band_finish(MC, coeff, col, seg, p, before, lost)


@jax.jit
def _band_loss(MC, coeff, col, seg, p, lost, keys):
    """Sequential material loss, then mean air loss; no change to the core stop or slice policy."""
    before = lost
    nmat = col.shape[1]

    def material(m, acc):
        c = eloss_jax.MatConsts(*(f[m] for f in MC))
        pp = _p_after(p, acc)
        length = col[:, m]
        tk0 = jnp.sqrt(pp * pp + eloss.MMU ** 2) - eloss.MMU
        mk = _fold(keys, m)
        initial = (jnp.zeros(len(p), jnp.int32), length, jnp.zeros_like(p))

        def pending(s):
            return (s[0] < tr.N_SLICES_MAX) & (s[1] > 0) & (tk0 - s[2] > tr.TK_STOP)

        def body(s):
            return jax.vmap(lambda k, q, l, i, left, loss: sj._slab_one(k, c, q, l, 1, (i, left, loss)))(
                mk, pp, length, *s)

        carry = lax.while_loop(lambda s: jnp.any(pending(s)), body, initial)
        dl = sj._slab_finish(MC, jnp.full(len(p), m), pp, carry)
        return acc + jnp.where(length > 0, dl, 0.0)

    if nmat:
        lost = lax.fori_loop(0, nmat, material, lost)
    air = jnp.maximum(seg - col.sum(1), 0.0)
    ca = eloss_jax.MatConsts(*(f[-1] for f in MC))
    lost = lost + eloss_jax._dedx(ca, jnp.maximum(_p_after(p, lost), 1.0), 1e9, eloss.MMU) * air
    columns = jnp.concatenate([col, air[:, None]], 1)
    xx0 = jnp.sum(columns * coeff[0], 1)
    bp = jnp.where(seg > 0, 0.5 * (_p_after(p, before) + _p_after(p, lost)), 0.0)
    pm = jnp.maximum(bp, 1.0)
    pb2 = (pm * pm / jnp.sqrt(pm * pm + eloss.MMU ** 2)) ** 2
    x2 = columns * coeff[1] / pb2[:, None]
    chi = x2.sum(1)
    thw = (x2 * coeff[2] / pm[:, None]).sum(1)
    thn = jnp.where(chi > 0, thw / jnp.maximum(chi, 1e-300), 1.0)
    return lost, xx0, bp, chi, thn


def _tail(keys, chi, thmin, thn):
    """Uncapped nuclear-tail Poisson process with per-proposal keys and core form-factor thinning."""
    a = 1 / jnp.maximum(thmin, 1e-12) ** 2
    b = 1 / tr.THETA_TAIL_MAX ** 2
    lam = jnp.maximum(chi * (a - b), 0.0)
    nk = _fold(keys, 0)
    count = jax.vmap(lambda k, l: jax.random.poisson(k, l))(nk, lam)
    maxcount = count.max(initial=0)

    def body(i, out):
        u = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, i + 1), (3,), jnp.float64))(keys)
        th = 1 / jnp.sqrt(a - u[:, 0] * (a - b))
        keep = (i < count) & (u[:, 1] < (1 + (th / thn) ** 2) ** -4)
        ph = 2 * np.pi * u[:, 2]
        return out + jnp.where(keep[:, None], th[:, None] * jnp.stack([jnp.cos(ph), jnp.sin(ph)], 1), 0.0)

    return lax.fori_loop(0, maxcount, body, jnp.zeros((len(chi), 2)))


@partial(jax.jit, static_argnames=("sliced", "spread", "tail", "tail_k"))
def _scatter(o, d, p0, lost, xx0, bh, bp, chi, thn, blen, keys, plane, sliced, spread, tail, tail_k):
    pw = _p_after(p0, lost)
    pm = jnp.maximum(jnp.sqrt(p0 * jnp.maximum(pw, 1.0)), 1.0)
    xt = xx0.sum(1)
    beta2 = pm * pm / (pm * pm + eloss.MMU ** 2)
    th0 = 13.6 / (jnp.sqrt(beta2) * pm) * jnp.sqrt(xt) * (
        1 + 0.038 * jnp.log(jnp.maximum(xt, 1e-12) / beta2))
    if sliced:
        pc = jnp.maximum(bp, 1.0)
        b2 = pc * pc / (pc * pc + eloss.MMU ** 2)
        vk = (13.6 / (jnp.sqrt(b2) * pc)) ** 2 * xx0
        beff = (vk * b2).sum(1) / jnp.maximum(vk.sum(1), 1e-30)
        corr = (1 + 0.038 * jnp.log(jnp.maximum(xt, 1e-12) / jnp.maximum(beff, 1e-6))) ** 2
        vk = vk * corr[:, None]
        w = vk / jnp.maximum(vk.sum(1, keepdims=True), 1e-30)
        th0 = jnp.sqrt(vk.sum(1))
    else:
        w = xx0 / jnp.maximum(xt[:, None], 1e-30)
    stt = th0 ** 2
    syt = stt * (w * bh).sum(1)
    syy = stt * (w * bh ** 2).sum(1)
    if spread:
        syy = syy + stt * (w * blen ** 2).sum(1) / 12
    a = jnp.sqrt(stt)
    b = jnp.sqrt(jnp.maximum(syy - syt ** 2 / jnp.maximum(stt, 1e-30), 0))
    z = jax.vmap(lambda k: jax.random.normal(k, (4,), jnp.float64))(_fold(keys, 0))
    dth = z[:, :2] * a[:, None]
    dy = z[:, :2] * (syt / jnp.maximum(a, 1e-30))[:, None] + z[:, 2:] * b[:, None]
    if tail:
        thmin = sj._tail_threshold(chi.sum(1), th0, tail_k)

        def add(k, s):
            angle = _tail(_fold(keys, k + 1), chi[:, k], thmin, thn[:, k])
            return s[0] + angle, s[1] + angle * bh[:, k, None]

        dth, dy = lax.fori_loop(0, chi.shape[1], add, (dth, dy))
    L = (o[:, 1] - plane) / -d[:, 1]
    end, dn = sj._kink(o + d * L[:, None], d, dth, dy)
    back = (end[:, 1] - plane) / dn[:, 1]
    tk = jnp.sqrt(p0 * p0 + eloss.MMU ** 2) - eloss.MMU - lost
    return end - dn * back[:, None], dn, pw, L - back, tk > 0.5


class DeviceWorld:
    """Immutable packed BandedWorld with bounded dense box tests and exact convex clipping."""

    def __init__(self, world, edges):
        self.edges = np.asarray(edges, float)
        if not isinstance(world, tr.BandedWorld):
            raise NotImplementedError("jax world transport requires a packed BandedWorld to preserve material order")
        if not np.array_equal(world.edges, self.edges):
            raise ValueError("world band edges do not match tracker plane")
        self.bands = []
        self.tiles = []
        for k, (top, bot) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            ps = world.band(k)
            mats = [materials.BY_NAME.get(m, materials.CONCRETE_MARS) for m in ps.mats] + [materials.AIR]
            cs = [eloss_jax.consts(m) for m in mats]
            mc = eloss_jax.MatConsts(*(jnp.asarray([getattr(c, f) for c in cs]) for f in eloss_jax.MatConsts._fields))
            a13 = np.array([sum(w * materials.ELEMENTS[e][1] ** (1 / 3) for e, w in m.fractions.items()) for m in mats])
            coeff = jnp.asarray([1 / np.array([eloss.radiation_length_mm(m) for m in mats]),
                                 [0.157 * tr._zz_over_a(m, False) * m.rho * 0.1 for m in mats],
                                 np.sqrt(20) * tr.HBARC_MEV_FM / (tr.R_NUCLEUS_FM * a13)])
            arrays = tuple(jnp.asarray(x) for x in (ps.lo, ps.hi, ps.P, ps.mat))
            self.bands.append((arrays, mc, coeff, len(ps.mats)))
            padded = (len(ps) + PIECE_TILE - 1) // PIECE_TILE * PIECE_TILE
            pad = padded - len(ps)
            lo = np.pad(ps.lo, ((0, pad), (0, 0)), constant_values=1e30)
            hi = np.pad(ps.hi, ((0, pad), (0, 0)), constant_values=1e30)
            planes = np.pad(ps.P, ((0, pad), (0, 0), (0, 0)))
            mi = np.pad(ps.mat, (0, pad), constant_values=len(ps.mats))
            self.tiles.append(tuple(jnp.asarray(v.reshape((padded // PIECE_TILE, PIECE_TILE) + v.shape[1:]))
                                    for v in (lo, hi, planes, mi)))

    def column(self, k, o, d, L, valid):
        _, _, _, nm = self.bands[k]
        if not nm:
            return jnp.zeros((len(o), 0))
        return _tiled_column(*self.tiles[k], o, d, L, valid, nm) / sj.QSCALE[0]

    def transport(self, key, o, d, p, indices=None, sliced=True, spread=False, tail=True, tail_k=None, lanes=None):
        lanes = lanes or min(sj.LANES, WORLD_LANES)
        indices = np.arange(len(p)) if indices is None else np.asarray(indices)
        if len(indices) != len(p):
            raise ValueError("one world key index is required per muon")
        if not len(p):
            return o.copy(), d.copy(), p.copy(), np.zeros(0), np.zeros(0, bool)
        outputs = []
        for start in range(0, len(p), lanes):
            n = min(lanes, len(p) - start)
            B = sj._bucket(n)
            sl = slice(start, start + n)
            oo = jnp.asarray(np.pad(o[sl], ((0, B - n), (0, 0))))
            dd = jnp.asarray(np.vstack([d[sl], np.tile([0.0, -1.0, 0.0], (B - n, 1))]))
            pp = jnp.asarray(np.pad(p[sl], (0, B - n), constant_values=1000.0))
            valid = jnp.arange(B) < n
            ix = jnp.asarray(np.pad(indices[sl], (0, B - n)).astype(np.uint32))
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(key, KEY_WORLD), i))(ix)
            lost = jnp.zeros(B)
            bands = []
            for k, (top, bot) in enumerate(zip(self.edges[:-1], self.edges[1:])):
                origin, length, height = _band_segment(oo, dd, top, bot, self.edges[-1], valid)
                col = self.column(k, origin, dd, length, valid)
                _, mc, coeff, _ = self.bands[k]
                lost, xx0, bp, chi, thn = _compact_band_loss(mc, coeff, col, length, pp, lost, _fold(keys, k))
                bands.append((xx0, height, bp, chi, thn, length))
            arrays = [jnp.stack([b[i] for b in bands], 1) for i in range(6)]
            result = _scatter(oo, dd, pp, lost, *arrays, _fold(keys, len(bands)), self.edges[-1],
                              sliced, spread, tail, tail_k)
            outputs.append(tuple(np.asarray(v)[:n] for v in result))
        return tuple(np.concatenate([v[i] for v in outputs]) for i in range(5))


def to_plane(gen, geom, key, world, sliced_msc=False, near_margin=st.FOOTPRINT_MARGIN, msc_tail=True,
             tail_k=None, band_spread=False, all_muons=True):
    """JAX twin of steps.to_plane; selection and DataFrames stay on the host."""
    mu = st._muons(gen, all_muons)
    p = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy().copy()
    o = mu[["x", "y", "z"]].to_numpy().copy()
    d = mu[["px", "py", "pz"]].to_numpy() / p[:, None]
    plane = st.plane_y(geom)
    edges = np.r_[st.BAND_EDGES_Y[st.BAND_EDGES_Y > plane], plane]
    near = st.near_tracker(o, d, geom, plane, margin=near_margin)
    alive = near.copy()
    length = np.zeros(len(mu))
    if near.any():
        dev = getattr(world, "_tracker_jax_world", None)
        if dev is None:
            dev = DeviceWorld(world, edges)
            if hasattr(world, "__dict__"):
                world._tracker_jax_world = dev
        if not np.array_equal(dev.edges, edges):
            raise ValueError("cached world band edges do not match tracker plane")
        result = dev.transport(key, o[near], d[near], p[near], np.flatnonzero(near), sliced=sliced_msc,
                               spread=band_spread, tail=msc_tail, tail_k=tail_k)
        o[near], d[near], p[near], length[near], alive[near] = result
    return mu, o, d, p, length, alive
