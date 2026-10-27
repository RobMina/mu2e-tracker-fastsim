"""JAX backend (CPU or GPU) of the stage C in-tracker transport, the straw crossings and the straw-gas eIon.

Opt-in with make_steps(..., backend="jax") (CLI `steps --backend jax`); numpy stays the default and the reference.
What runs where:
  * host, numpy, unchanged: generator selection, the library secondaries, path vertices and DataFrame assembly;
  * device, JAX: world_jax transport to the plane (convex column, ordered sliced loss, air and scattering);
  * device, JAX: TrackerInterior.transport (straw crossings re-found after every kink, support column of the coaxial
    shells and the calorimeter pieces, kink splitting by bisection, sliced Urban loss in the support, Highland core and
    single-scattering tail per kink group, the gas-exit back-off), TrackerInterior.exit_scatter, the straight-line
    crossings of stage A, and the Urban eIon in the straw gas (mu2efastsim.eloss_jax).
Same algorithm, cuts and conventions as the numpy code; random streams differ (explicit keys per muon, per pass and per
purpose). Statistical agreement and cross-device tolerance must be checked on each supported geometry/configuration;
outputs are reproducible per seed on one device and lane count.
Masked and padding writes go out of bounds with mode="drop" (a common dummy index serialises GPU scatters), and the
host loops sync only for bucket sizes and loop ends; the remaining per-pass glue is jitted.

Fixed shapes: muons are processed in lanes (padded to power-of-two buckets, one dummy lane), each pass keeps the first
K_ROWS straw crossings ahead of the muon (the kink trigger comes within ~10 straws), the support column runs on a
compacted work list of (segment, shell z-interval) items, and every cap is counted in the stats (*_overflow must be 0).
"""
from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eloss, eloss_jax, materials
from mu2efastsim import transport as tr

from . import crossings, geometry
from . import steps as st

jax.config.update("jax_enable_x64", True)

MMU = eloss.MMU
TWO_PI = 2 * np.pi
K_ROWS = 32       # straw crossings ahead of the muon per pass
P_CAP = 64        # nearest panel boxes per line and pass (beyond them a pass sees only crossings before the next box)
R_CAP = 192       # recorded crossings per muon
KK_CAP = 128      # transport kinks per muon
KX_CAP = 64       # exit_scatter kinks per muon (its iteration cap)
PT_CAP = 4        # panel boxes holding one point
N_TAIL = 8        # single-scatter proposals per kink group and process (Poisson means are < 0.5)
K_ALL = 128       # straight-line crossings per line (stage A)
LANES = 4096      # muons per transport chunk
SLICE_ROUNDS = 8  # slab slices per device call between compactions
KEY_TRANSPORT, KEY_EXIT, KEY_EION, KEY_SLAB, KEY_CORE, KEY_TAIL = 1, 2, 3, 2, 0, 1
# column sums in int64 fixed point (exact and order-free, so independent of batching and of the device): length in
# 1e-9 mm, first moment in 1e-6 mm^2, second in 1e-3 mm^3
QSCALE = (1e9, 1e6, 1e3)
_NBIG = 1 << 30   # out-of-bounds index of padding entries (gathers clamp, mode="drop" writes skip it)


def _bucket(n, lo=64):
    n = max(int(n), 1)
    return max(lo, 1 << (n - 1).bit_length())


def key_from(rng):
    """JAX key from one draw of the numpy generator, before any device transport."""
    return jax.random.key(int(rng.integers(0, 2 ** 62)))


def to_plane(*args, **kwargs):
    """Device world transport; lazy import keeps its kernels separate from interior transport."""
    from .world_jax import to_plane as world_to_plane
    return world_to_plane(*args, **kwargs)


def _to_np(x):
    return np.asarray(jax.device_get(x))


class Cfg(NamedTuple):
    """Static (hashable) configuration of the device kernels."""
    ns: int
    npanel: int
    m_sup: int
    m_all: int
    ngroup: int
    nsub: int
    rflag2: float
    r_in: float
    r_out: float
    f_metal: float
    rw: float
    rw_in: float
    has_calo: bool


# --------------------------------------------------------------------------- small physics pieces (jnp twins)

def _p_of(tk):
    tk = jnp.maximum(tk, 0.5)
    return jnp.sqrt((tk + MMU) ** 2 - MMU ** 2)


def _highland_var(p, x_prev, x_new):
    p = jnp.maximum(p, 1e-3)
    beta2 = p * p / (p * p + MMU * MMU)

    def g(x):
        x = jnp.maximum(x, 0.0)
        return jnp.where(x > 0, x * (1 + 0.038 * jnp.log(jnp.maximum(x, 1e-30) / beta2)) ** 2, 0.0)

    return (13.6 ** 2 / (beta2 * p * p)) * jnp.maximum(g(x_new) - g(x_prev), 0.0)


def _tail_threshold(chic2, theta0, tail_k):
    th0 = jnp.maximum(theta0, 1e-12)
    if tail_k is not None:
        return tail_k * th0
    r = jnp.maximum(chic2 / th0 ** 2, 1e-300)
    lr = jnp.log(2 * r)

    def body(_, u):
        h = 4 * jnp.log(u) - 0.5 * u * u - lr
        return jnp.maximum(u - h / (4 / u - u), 2.0)

    return lax.fori_loop(0, 30, body, jnp.full_like(r, 6.0)) * th0


def _kink_core(z, V, w1, w2):
    """steps._kink_core from given normals z [B, 4] (z_t = z[:, :2], z_y = z[:, 2:])."""
    a = jnp.sqrt(V)
    b = jnp.sqrt(jnp.clip(V * w2 - (V * w1) ** 2 / jnp.maximum(V, 1e-30), 0, None))
    dth = z[:, :2] * a[:, None]
    dy = z[:, :2] * (V * w1 / jnp.maximum(a, 1e-30))[:, None] + z[:, 2:] * b[:, None]
    return dth, dy, a


def _kink(pos, d, dth, dy):
    """steps._kink."""
    ref = jnp.where(jnp.abs(d[:, [0]]) < 0.9, jnp.array([[1.0, 0, 0]]), jnp.array([[0, 0, 1.0]]))
    u = jnp.cross(d, ref)
    u = u / jnp.linalg.norm(u, axis=1)[:, None]
    v = jnp.cross(d, u)
    dn = d + u * jnp.tan(dth[:, [0]]) + v * jnp.tan(dth[:, [1]])
    dn = dn / jnp.linalg.norm(dn, axis=1)[:, None]
    return pos + u * dy[:, [0]] + v * dy[:, [1]], dn


def _poisson_small(lam, u, n):
    """Inverse-CDF Poisson counts capped at n (for means << 1); also whether the cap was reached."""
    pmf = jnp.exp(-lam)
    cdf = pmf
    k = jnp.zeros(lam.shape, jnp.int32)
    for j in range(n):
        above = u > cdf
        k = k + above
        pmf = pmf * lam / (j + 1)
        cdf = cdf + pmf
    return k, u > cdf


def _group_tail(T, key, L, p, a, tail_k):
    """Single-scattering tail [B, 2] of kink groups (steps._group_tail): L [B, M] path per material, p [B] |p|.

    The nuclear parts of all materials form one Poisson process (rate summed, material drawn by its chi_c^2 share, kept
    with its own form factor), which is the same distribution as one process per material; electrons separately."""
    pc = jnp.maximum(p, 1e-3)
    pb2 = (pc * pc / jnp.sqrt(pc * pc + MMU * MMU)) ** 2
    x2 = T["coef_nuc"][None] * L / pb2[:, None]
    xn = _dot(x2, jnp.ones(x2.shape[1]))
    che = _dot(L, T["coef_el"]) / pb2
    th_min = _tail_threshold(xn + che, a, tail_k)
    ia = 1.0 / jnp.maximum(th_min, 1e-12) ** 2
    bn, be = 1.0 / tr.THETA_TAIL_MAX ** 2, 1.0 / tr.ME_OVER_MMU ** 2
    lam_n = jnp.clip(xn * (ia - bn), 0, None)
    lam_e = jnp.clip(che * (ia - be), 0, None)
    B = p.shape[0]
    u = jax.vmap(lambda k: jax.random.uniform(k, (2 + 7 * N_TAIL,), jnp.float64))(key)
    nn, ovn = _poisson_small(lam_n, u[:, 0], N_TAIL)
    ne, ove = _poisson_small(lam_e, u[:, 1], N_TAIL)
    r = u[:, 2:].reshape(B, 7, N_TAIL)
    j = jnp.arange(N_TAIL)[None]
    # nuclear proposals: angle, material, form-factor thinning, azimuth
    th = 1.0 / jnp.sqrt(ia[:, None] - r[:, 0] * (ia - bn)[:, None])
    cx = jnp.cumsum(x2, 1)
    m = jnp.sum(r[:, 1][:, :, None] * xn[:, None, None] > cx[:, None, :], -1)
    m = jnp.minimum(m, x2.shape[1] - 1)
    tn = (np.sqrt(20.0) * tr.HBARC_MEV_FM / (pc[:, None] * tr.R_NUCLEUS_FM)) / T["a13"][m]
    keep = (j < nn[:, None]) & (r[:, 2] < (1 + (th / tn) ** 2) ** -4)
    ph = r[:, 3] * TWO_PI
    tt = jnp.stack([jnp.sum(jnp.where(keep, th * jnp.cos(ph), 0.0), 1), jnp.sum(jnp.where(keep, th * jnp.sin(ph), 0.0), 1)], 1)
    # electron proposals, up to m_e / m_mu, no form factor
    the = 1.0 / jnp.sqrt(ia[:, None] - r[:, 4] * (ia - be)[:, None])
    keep_e = j < ne[:, None]
    phe = r[:, 5] * TWO_PI
    tt = tt + jnp.stack([jnp.sum(jnp.where(keep_e, the * jnp.cos(phe), 0.0), 1),
                         jnp.sum(jnp.where(keep_e, the * jnp.sin(phe), 0.0), 1)], 1)
    return tt, jnp.sum(ovn | ove)


def _envelope_exit(o, d):
    """steps._envelope_exit."""
    from . import surfaces as S
    x = o - jnp.asarray(geometry.DETECTOR_ORIGIN_EXTRACTED)
    A = d[:, 0] ** 2 + d[:, 1] ** 2
    B = 2 * (x[:, 0] * d[:, 0] + x[:, 1] * d[:, 1])
    C = x[:, 0] ** 2 + x[:, 1] ** 2 - S.R_OUT ** 2
    disc = jnp.clip(B * B - 4 * A * C, 0, None)
    t_cyl = jnp.where(A > 1e-15, (-B + jnp.sqrt(disc)) / (2 * jnp.maximum(A, 1e-15)), jnp.inf)
    dz = jnp.where(jnp.abs(d[:, 2]) > 1e-15, d[:, 2], 1.0)
    tz = jnp.where(d[:, 2] > 1e-15, (S.Z_BACK - x[:, 2]) / dz,
                   jnp.where(d[:, 2] < -1e-15, (S.Z_FRONT - x[:, 2]) / dz, jnp.inf))
    inside = (C < 0) & (x[:, 2] > S.Z_FRONT) & (x[:, 2] < S.Z_BACK)
    return jnp.where(inside, jnp.clip(jnp.minimum(t_cyl, tz), 0, None), 0.0)


# --------------------------------------------------------------------------- straws

def _line_straws(T, cfg, o, d, P):
    """Straw candidates of each line in its P nearest panel boxes (by entry distance): js, s_in, s_out, keep [B, P * 96],
    panel count [B] and the cut [B]: the entry distance of the nearest box left out (inf if none), before which the
    candidates are complete (keep is False at and beyond it). P >= the panel count: every box, no cut."""
    inv = 1.0 / jnp.where(d == 0.0, 1e-300, d)
    t1 = (T["plo"][None] - o[:, None]) * inv[:, None]
    t2 = (T["phi"][None] - o[:, None]) * inv[:, None]
    tin = jnp.maximum(jnp.max(jnp.minimum(t1, t2), -1), 0.0)
    tout = jnp.min(jnp.maximum(t1, t2), -1)
    hit = tout > tin
    if P >= cfg.npanel:
        pan = jax.vmap(lambda h: jnp.nonzero(h, size=cfg.npanel, fill_value=cfg.npanel)[0])(hit)
        cut = jnp.full(o.shape[0], jnp.inf)
    else:
        v, near = lax.top_k(jnp.where(hit, -tin, -jnp.inf), P + 1)
        pan = jnp.where(v[:, :P] > -jnp.inf, near[:, :P], cfg.npanel)
        cut = jnp.where(v[:, P] > -jnp.inf, -v[:, P], jnp.inf)
    js = T["straws_of"][pan].reshape(o.shape[0], -1)
    w = o[:, None] - T["smid"][js]
    a = T["sdir"][js]
    dd = d[:, None]
    da = jnp.sum(dd * a, -1)
    wa = jnp.sum(w * a, -1)
    u = dd - da[..., None] * a
    v = w - wa[..., None] * a
    A = jnp.sum(u * u, -1)
    Bq = 2 * jnp.sum(u * v, -1)
    C = jnp.sum(v * v, -1) - cfg.r_in ** 2
    disc = Bq * Bq - 4 * A * C
    par = A <= 1e-15
    ok = jnp.where(par, C <= 0, disc > 0)
    sq = jnp.sqrt(jnp.where(ok & ~par, disc, 0.0))
    A_ = jnp.where(par, 1.0, A)
    s0 = jnp.where(par, -jnp.inf, (-Bq - sq) / (2 * A_))
    s1 = jnp.where(par, jnp.inf, (-Bq + sq) / (2 * A_))
    h = T["shalf"][js]
    nz = jnp.abs(da) > 1e-12
    das = jnp.where(nz, da, 1.0)
    z0, z1 = (-h - wa) / das, (h - wa) / das
    zl = jnp.where(nz, jnp.minimum(z0, z1), -jnp.inf)
    zh = jnp.where(nz, jnp.maximum(z0, z1), jnp.inf)
    inside_len = nz | (jnp.abs(wa) <= h)
    s_in, s_out = jnp.maximum(jnp.maximum(s0, zl), 0.0), jnp.minimum(s1, zh)
    keep = ok & inside_len & (s_out > s_in) & (js < cfg.ns) & (s_in < cut[:, None])
    return js, s_in, s_out, keep, hit.sum(1), cut


def _pad_candidates(js, s_in, s_out, keep, K, ns):
    """At least K candidate columns (top_k needs them; fewer when few panels are taken)."""
    n = js.shape[1]
    if n >= K:
        return js, s_in, s_out, keep
    pad = lambda x, v: jnp.concatenate([x, jnp.full((x.shape[0], K - n), v, x.dtype)], 1)
    return pad(js, ns), pad(s_in, 0.0), pad(s_out, 0.0), pad(keep, False)


@partial(jax.jit, static_argnames=("cfg", "K"))
def _crossings_all(T, cfg, o, d, K):
    js, s_in, s_out, keep, _, _ = _line_straws(T, cfg, o, d, cfg.npanel)
    js, s_in, s_out, keep = _pad_candidates(js, s_in, s_out, keep, K, cfg.ns)
    v, pos = lax.top_k(jnp.where(keep, -s_in, -jnp.inf), K)
    valid = v > -jnp.inf
    return jnp.take_along_axis(js, pos, 1), -v, jnp.take_along_axis(s_out, pos, 1), valid, keep.sum(1)


def _point_straw(T, cfg, x):
    """crossings.point_straw: straw whose gas holds each point [B, 3], or -1 (last match, as numpy)."""
    inbox = jnp.all((x[:, None] >= T["plo"][None]) & (x[:, None] <= T["phi"][None]), -1)
    pan = jax.vmap(lambda h: jnp.nonzero(h, size=PT_CAP, fill_value=cfg.npanel)[0])(inbox)
    js = T["straws_of"][pan].reshape(x.shape[0], -1)
    w = x[:, None] - T["smid"][js]
    wa = jnp.sum(w * T["sdir"][js], -1)
    r2 = jnp.sum(w * w, -1) - wa * wa
    ok = (r2 <= cfg.r_in ** 2) & (jnp.abs(wa) <= T["shalf"][js]) & (js < cfg.ns)
    last = jnp.argmax(jnp.where(ok, jnp.arange(js.shape[1])[None], -1), 1)
    return jnp.where(ok.any(1), js[jnp.arange(x.shape[0]), last], -1), jnp.sum(inbox.sum(1) > PT_CAP)


def _exit_gas_backwards(T, cfg, pos, d):
    """steps._exit_gas_backwards."""
    j, over = _point_straw(T, cfg, pos)
    ins = j >= 0
    jj = jnp.where(ins, j, 0)
    a = T["sdir"][jj]
    w = pos - T["smid"][jj]
    uu = d - jnp.sum(d * a, -1)[:, None] * a
    ww = w - jnp.sum(w * a, -1)[:, None] * a
    A = jnp.sum(uu * uu, -1)
    B = 2 * jnp.sum(uu * ww, -1)
    C = jnp.sum(ww * ww, -1) - cfg.r_in ** 2
    t = (-B - jnp.sqrt(jnp.clip(B * B - 4 * A * C, 0, None))) / (2 * jnp.maximum(A, 1e-15))
    return jnp.where(ins[:, None], pos + d * (jnp.minimum(t, 0.0) - 1e-4)[:, None], pos), over


# --------------------------------------------------------------------------- support column

@partial(jax.jit, static_argnames=("cfg",))
def _coax_windows(T, cfg, o, d, L, valid):
    """Per (segment, shell group, annulus interval): the interval [s0, s1] and the window of z-intervals it can
    overlap (CoaxialShells.column: shells with z0 < zhi and z1 > zlo); also the calorimeter-box test."""
    w = o - T["c"]
    ox, oy, oz = w @ T["e1"], w @ T["e2"], w @ T["ax"]
    dx, dy, dz = d @ T["e1"], d @ T["e2"], d @ T["ax"]
    A = dx * dx + dy * dy
    Bq = 2 * (ox * dx + oy * dy)
    r2 = ox * ox + oy * oy

    def chord(r):
        Cq = r2[:, None] - r[None] ** 2
        disc = Bq[:, None] ** 2 - 4 * A[:, None] * Cq
        par = (A < 1e-15)[:, None]
        ok = jnp.where(par, Cq <= 0, disc > 0)
        sq = jnp.sqrt(jnp.where(ok & ~par, disc, 0.0))
        a_ = jnp.where(par, 1.0, A[:, None])
        return (jnp.where(par, -jnp.inf, (-Bq[:, None] - sq) / (2 * a_)),
                jnp.where(par, jnp.inf, (-Bq[:, None] + sq) / (2 * a_)), ok)

    a0, a1, aok = chord(T["rmax"])
    lo, hi = jnp.maximum(a0, 0.0), jnp.minimum(a1, L[:, None])
    cand = aok & (hi > lo) & valid[:, None]
    b0, b1, bok = chord(T["rmin"])
    hole = bok & (T["rmin"] > 0)[None]
    b0c, b1c = jnp.where(hole, b0, hi), jnp.where(hole, b1, hi)
    s0 = jnp.stack([lo, jnp.maximum(lo, jnp.minimum(b1c, hi))], -1)
    s1 = jnp.stack([jnp.minimum(hi, jnp.maximum(b0c, lo)), hi], -1)
    live = cand[..., None] & (s1 > s0)
    za, zb = oz[:, None, None] + dz[:, None, None] * s0, oz[:, None, None] + dz[:, None, None] * s1
    zlo, zhi = jnp.minimum(za, zb), jnp.maximum(za, zb)
    G = cfg.ngroup
    zhi_g = jnp.moveaxis(zhi, 1, 0).reshape(G, -1)
    zlo_g = jnp.moveaxis(zlo, 1, 0).reshape(G, -1)
    j_hi = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="left"))(T["Z0"], zhi_g) - 1
    j_lo = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="right"))(T["ZRUN"], zlo_g)
    j_hi = jnp.moveaxis(j_hi.reshape(G, -1, 2), 0, 1)
    j_lo = jnp.moveaxis(j_lo.reshape(G, -1, 2), 0, 1)
    W = jnp.where(live, jnp.clip(j_hi - j_lo + 1, 0, None), 0).astype(jnp.int32)
    if cfg.has_calo:
        b = o + d * L[:, None]
        near = valid & jnp.all(jnp.maximum(o, b) >= T["calo_lo"], 1) & jnp.all(jnp.minimum(o, b) <= T["calo_hi"], 1)
    else:
        near = jnp.zeros(o.shape[0], bool)
    win = {"ox": ox, "oy": oy, "oz": oz, "dx": dx, "dy": dy, "dz": dz, "s0": s0, "s1": s1, "zlo": zlo,
           "j_lo": j_lo.astype(jnp.int32), "W": W}
    return win, jnp.sum(W), near, jnp.sum(near)


def _q(x, k):
    return jnp.round(x * QSCALE[k]).astype(jnp.int64)


def _qsum(q, oh, axis):
    """Integer sum over `axis` of q into the one-hot material slots oh (exact)."""
    return jnp.sum(jnp.where(oh, q[..., None], 0), axis)


def _seg_reduce(q, cstart, cend):
    """Sums of contiguous runs [cstart, cend) of rows of the int64 array q [T, M]: cumsum difference, exact (wrap-around
    of the running sum cancels in the difference)."""
    cs = jnp.concatenate([jnp.zeros((1,) + q.shape[1:], q.dtype), jnp.cumsum(q, 0)], 0)
    return cs[cend] - cs[cstart]


def _dot(x, w):
    """x [..., M] . w [M] summed in a fixed order (no batch-dependent contraction)."""
    out = x[..., 0] * w[0]
    for i in range(1, x.shape[-1]):
        out = out + x[..., i] * w[i]
    return out


@partial(jax.jit, static_argnames=("cfg", "TB", "moments"))
def _coax_items(T, cfg, win, TB, moments):
    """Coaxial-shell column per segment and support material from the flat (segment, group, interval, z-interval)
    work list: exact z clipping, phi sectors by the nsub-point sampling of CoaxialShells.column."""
    C = win["ox"].shape[0]
    G = cfg.ngroup
    W = win["W"].reshape(-1)
    cum = jnp.cumsum(W)
    start = cum - W
    t = jnp.arange(TB)
    tv = t < cum[-1]
    src = jnp.minimum(jnp.searchsorted(cum, t, side="right"), W.shape[0] - 1)
    c = src // (2 * G)
    g = (src // 2) % G
    j = win["j_lo"].reshape(-1)[src] + (t - start[src])
    j = jnp.clip(j, 0, T["Z0"].shape[1] - 1)
    zlo = win["zlo"].reshape(-1)[src]
    s0, s1 = win["s0"].reshape(-1)[src], win["s1"].reshape(-1)[src]
    z0, z1 = T["Z0"][g, j], T["Z1"][g, j]
    over = tv & (z1 > zlo)
    oz, dz = win["oz"][c], win["dz"][c]
    nz = jnp.abs(dz) > 1e-15
    dzs = jnp.where(nz, dz, 1.0)
    t0 = jnp.where(nz, (z0 - oz) / dzs, -jnp.inf)
    t1 = jnp.where(nz, (z1 - oz) / dzs, jnp.inf)
    u0 = jnp.maximum(s0, jnp.minimum(t0, t1))
    u1 = jnp.minimum(s1, jnp.maximum(t0, t1))
    ln = jnp.where(over, jnp.clip(u1 - u0, 0, None), 0.0)
    kk = (jnp.arange(cfg.nsub) + 0.5) / cfg.nsub
    sm = u0[:, None] + ln[:, None] * kk[None]
    ph = jnp.arctan2(win["oy"][c][:, None] + win["dy"][c][:, None] * sm, win["ox"][c][:, None] + win["dx"][c][:, None] * sm)
    p0, dp, full, mat = T["P0"][g, j], T["DP"][g, j], T["FULL"][g, j], T["MAT"][g, j]     # [TB, S]
    ins = jnp.mod(ph[:, None, :] - p0[:, :, None], TWO_PI) <= dp[:, :, None]           # [TB, S, nsub]
    sector = ~full & (ln[:, None] > 0)
    frac = jnp.where(sector, jnp.mean(ins, -1), 1.0)
    val = ln[:, None] * frac
    oh = mat[:, :, None] == jnp.arange(cfg.m_sup)[None, None]
    Wc = W.reshape(C, 2 * G).sum(1)
    cend = jnp.cumsum(Wc)
    cstart = cend - Wc
    out = _seg_reduce(_qsum(_q(val, 0), oh, 1), cstart, cend)
    if not moments:
        return out
    e1 = u0 + ln
    m1f, m2f = 0.5 * (e1 ** 2 - u0 ** 2), (e1 ** 3 - u0 ** 3) / 3.0
    m1 = jnp.where(sector, ln[:, None] * jnp.mean(ins * sm[:, None, :], -1), m1f[:, None])
    m2 = jnp.where(sector, ln[:, None] * jnp.mean(ins * sm[:, None, :] ** 2, -1), m2f[:, None])
    pos = val > 0
    return (out, _seg_reduce(_qsum(jnp.where(pos, _q(m1, 1), 0), oh, 1), cstart, cend),
            _seg_reduce(_qsum(jnp.where(pos, _q(m2, 2), 0), oh, 1), cstart, cend))


def _box_hits(lo, hi, o, d, L, valid):
    inv = 1.0 / jnp.where(d == 0, 1e-300, d)
    t1 = (lo[None] - o[:, None]) * inv[:, None]
    t2 = (hi[None] - o[:, None]) * inv[:, None]
    tin = jnp.maximum(jnp.max(jnp.minimum(t1, t2), -1), 0.0)
    tout = jnp.minimum(jnp.min(jnp.maximum(t1, t2), -1), L[:, None])
    return (tout > tin) & valid[:, None]


@partial(jax.jit, static_argnames=("m_sup", "moments", "H"))
def _calo_set(lo, hi, P, mat, sign, o, d, L, valid, m_sup, moments, H):
    """One PieceSet of the calorimeter: signed column per support material of each segment (dense box test over the
    set's pieces, then plane clipping of the boxes hit, at most H per segment: the caller sizes H)."""
    hit = _box_hits(lo, hi, o, d, L, valid)
    nh = hit.sum(1)
    ip = jax.vmap(lambda h: jnp.nonzero(h, size=H, fill_value=0)[0])(hit)
    iv = jnp.arange(H)[None] < nh[:, None]
    Pp = P[ip]
    den = jnp.einsum("cj,chkj->chk", d, Pp[..., :3])
    num = Pp[..., 3] - jnp.einsum("cj,chkj->chk", o, Pp[..., :3])
    s = num / jnp.where(den == 0, 1.0, den)
    enter = jnp.max(jnp.where(den < 0, s, -jnp.inf), -1)
    leave = jnp.min(jnp.where(den > 0, s, jnp.inf), -1)
    outside = jnp.any((den == 0) & (num < 0), -1)
    a = jnp.maximum(enter, 0.0)
    b = jnp.minimum(leave, L[:, None])
    Lp = jnp.where(outside | ~iv, 0.0, jnp.clip(b - a, 0.0, None))
    oh = mat[ip][..., None] == jnp.arange(m_sup)[None, None]
    out = sign * _qsum(_q(Lp, 0), oh, 1)
    over = jnp.sum(nh > H)
    if not moments:
        return out, over
    e1 = a + Lp
    m1 = sign * _qsum(_q(0.5 * (e1 ** 2 - a ** 2), 1), oh, 1)
    m2 = sign * _qsum(_q((e1 ** 3 - a ** 3) / 3.0, 2), oh, 1)
    return (out, m1, m2), over


@partial(jax.jit, static_argnames=("cfg", "TB", "moments"))
def _coax_column(T, cfg, win, TB, moments):
    return _dequant(_coax_items(T, cfg, win, TB, moments), moments)


@partial(jax.jit, static_argnames=("CB",))
def _calo_prep(boxes, o, d, L, near, CB):
    """Segments near the calorimeter, compacted, and the most boxes one of them meets in each piece set."""
    idx = jnp.nonzero(near, size=CB, fill_value=near.shape[0])[0]
    v = jnp.arange(CB) < jnp.sum(near)
    so, sd, sL = (jnp.concatenate([x, x[:1]])[idx] for x in (o, d, L))
    nh = jnp.stack([jnp.max(_box_hits(lo, hi, so, sd, sL, v).sum(1)) for lo, hi in boxes])
    return idx, v, so, sd, sL, nh


@jax.jit
def _tree_add(a, b):
    return jax.tree_util.tree_map(jnp.add, a, b)


@partial(jax.jit, static_argnames=("C", "moments"))
def _merge_dequant(out, acc, idx, v, C, moments):
    return _dequant(_calo_merge(out, acc, idx, v, C, moments), moments)


@partial(jax.jit, static_argnames=("n",))
def _bisect_start(seg, n):
    lo = jnp.zeros(seg.shape, jnp.int32)
    hi = jnp.full(seg.shape, n, jnp.int32)
    mid = (lo + hi) // 2
    return lo, hi, mid, seg * mid / st.TrackerInterior.SPLIT_PROBES


@jax.jit
def _bisect_step(lo, hi, mid, x, target, seg):
    ok = x >= target
    hi, lo = jnp.where(ok, mid, hi), jnp.where(ok, lo, mid)
    mid = (lo + hi) // 2
    return lo, hi, mid, seg * mid / st.TrackerInterior.SPLIT_PROBES


@jax.jit
def _bisect_len(seg, hi):
    return seg * hi / st.TrackerInterior.SPLIT_PROBES


# --------------------------------------------------------------------------- sliced support loss

def _slab_one(key, c, p, L, rounds, carry):
    """transport.slab_loss for one (material, |p|, length), `rounds` slices from `carry` = (i, left, lost)."""
    tk0 = jnp.sqrt(p * p + MMU * MMU) - MMU

    def body(_, s):
        i, left, lost = s
        go = (i < tr.N_SLICES_MAX) & (left > 0) & (tk0 - lost > tr.TK_STOP)
        tk = tk0 - lost
        pg = jnp.sqrt(jnp.clip((tk + MMU) ** 2 - MMU ** 2, 1e-6, None))
        dedx = eloss_jax._dedx(c, pg, 1e9, MMU)
        dedx_s = jnp.where(dedx > 0, dedx, 1.0)
        need = dedx * left
        coarse = tk > 3.0 * need
        step = jnp.where(coarse, left / tr.MAX_SLICES_PER_SLAB, tr.SLICE_FRAC * tk / dedx_s)
        step = jnp.minimum(left, jnp.maximum(step, tr.SLICE_FRAC * tk / dedx_s))
        step = jnp.where(left <= step * 1.0001, left, step)
        dl = eloss_jax._sample(jax.random.fold_in(key, i), c, pg, jnp.where(go, step, 0.0), 1e9, MMU)
        dl = jnp.where(dl >= tk, dedx * step, dl)
        return (i + go, jnp.where(go, left - step, left), jnp.where(go, lost + dl, lost))

    return lax.fori_loop(0, rounds, body, carry)


@partial(jax.jit, static_argnames=("rounds",))
def _slab_rounds(MC, mat, p, L, keys, carry, rounds, valid=None):
    c = eloss_jax.MatConsts(*(f[mat] for f in MC))
    tk0 = jnp.sqrt(p * p + MMU * MMU) - MMU
    valid = jnp.ones(p.shape, bool) if valid is None else valid

    def pending(s):
        return valid & (s[0] < tr.N_SLICES_MAX) & (s[1] > 0) & (tk0 - s[2] > tr.TK_STOP)

    def body(state):
        r, s, _ = state
        out = jax.vmap(lambda k, cc, pp, ll, i, lf, lo: _slab_one(k, cc, pp, ll, 1, (i, lf, lo)))(
            keys, c, p, L, *s)
        return r + 1, out, pending(out)

    _, out, todo = lax.while_loop(lambda s: (s[0] < rounds) & jnp.any(s[2]), body,
                                   (jnp.int32(0), carry, pending(carry)))
    return out, todo


@jax.jit
def _slab_finish(MC, mat, p, carry):
    c = eloss_jax.MatConsts(*(f[mat] for f in MC))
    i, left, lost = carry
    tk0 = jnp.sqrt(p * p + MMU * MMU) - MMU
    rest = (left > 0) & (tk0 - lost > tr.TK_STOP)
    tk = tk0 - lost
    pg = jnp.sqrt(jnp.clip((tk + MMU) ** 2 - MMU ** 2, 1e-6, None))
    lost = jnp.where(rest, lost + eloss_jax._dedx(c, pg, 1e9, MMU) * left, lost)
    stopped = tk0 - lost <= tr.TK_STOP
    return jnp.where(stopped, tk0, jnp.minimum(lost, tk0))


@jax.jit
def _count_pos(col):
    return jnp.sum(col > 0)


@partial(jax.jit, static_argnames=("C",))
def _slab_pairs(col, keys, p, C):
    """(row, material) pairs with support length, compacted: row, mat, length, per-pair key, valid, max(|p|, 1), the
    initial slice state (slices, length left, loss) and the work list."""
    R, M = col.shape
    flat = (col > 0).reshape(-1)
    idx = jnp.nonzero(flat, size=C, fill_value=0)[0]
    v = jnp.arange(C) < flat.sum()
    row, mat = idx // M, idx % M
    k = jax.vmap(jax.random.fold_in)(keys[row], mat)
    L = jnp.where(v, col[row, mat], 0.0)
    return row, mat, L, k, v, jnp.maximum(p[row], 1.0), (jnp.zeros(C, jnp.int32), L, jnp.zeros(C)), jnp.arange(C)


@partial(jax.jit, static_argnames=("rounds",))
def _slab_iter(MC, mat, pp, L, k, carry, idx, rounds):
    """`rounds` slices of the work list idx (out-of-bounds padding: gathers clamp, writes dropped); pending count."""
    out, todo = _slab_rounds(MC, mat[idx], pp[idx], L[idx], k[idx], tuple(x[idx] for x in carry), rounds,
                             valid=(idx >= 0) & (idx < pp.shape[0]))
    return _carry_set(carry, idx, out), todo, jnp.sum(todo)


@partial(jax.jit, static_argnames=("CB",))
def _slab_next(idx, todo, CB):
    sel = jnp.nonzero(todo, size=CB, fill_value=todo.shape[0])[0]
    return _pad_oob(idx, sel, jnp.arange(CB) < jnp.sum(todo), _NBIG)


@partial(jax.jit, static_argnames=("R", "M"))
def _slab_done(MC, mat, pp, carry, row, v, R, M):
    return _slab_scatter(row, mat, _slab_finish(MC, mat, pp, carry), v, R, M)


@partial(jax.jit, static_argnames=("R", "M"))
def _slab_scatter(row, mat, loss, v, R, M):
    out = jnp.zeros((R, M)).at[jnp.where(v, row, R), mat].set(loss, mode="drop")
    return _dot(out, jnp.ones(M))


@partial(jax.jit, static_argnames=("n",))
def _pad_oob(idx, sel, sv, n):
    """idx[sel] for the valid entries of a compaction, n (out of bounds) for the padding."""
    return jnp.where(sv, jnp.concatenate([idx, idx[:1]])[sel], n)


@jax.jit
def _carry_set(carry, idx, out):
    """Write the slab-loop state of the entries idx back (out-of-bounds padding dropped; gathers clamp, so the padding
    lanes compute on a real entry's inputs and are discarded)."""
    return tuple(c.at[idx].set(x, mode="drop") for c, x in zip(carry, out))


@jax.jit
def _xsup(col, invx0):
    return _dot(col, invx0)


# --------------------------------------------------------------------------- transport pass

@partial(jax.jit, static_argnames=("cfg", "K", "P"))
def _pass(T, cfg, S, lane, done, kink_xx0, K, P=P_CAP):
    """Straw rows of one pass for the active lanes: the first K crossings ahead (recorded straws skipped), the thin
    material per row, the thin-material trigger row and the rows that may hold support (candidates for the column).
    Candidates come from the P nearest panel boxes; lanes with more boxes are flagged (rows["cut"]) and redone with
    every box (_pass_cut_lanes), so the rows are exact."""
    o, d, restarted = S["o"][lane], S["d"][lane], S["restarted"][lane]
    B = o.shape[0]
    real = lane < S["o"].shape[0] - 1
    js, s_in, s_out, keep, npan, cut = _line_straws(T, cfg, o, d, P)
    js, s_in, s_out, keep = _pad_candidates(js, s_in, s_out, keep, K, cfg.ns)
    keep = keep & (s_out > 1e-9) & ~done[lane[:, None], js] & real[:, None]
    ntot = keep.sum(1)
    v, pos = lax.top_k(jnp.where(keep, -s_in, -jnp.inf), K)
    valid = v > -jnp.inf
    js = jnp.where(valid, jnp.take_along_axis(js, pos, 1), cfg.ns)
    s_in = jnp.where(valid, -v, 0.0)
    s_out = jnp.where(valid, jnp.take_along_axis(s_out, pos, 1), 0.0)
    nval = valid.sum(1)
    k = jnp.arange(K)[None]
    first = k == 0
    prev = jnp.concatenate([jnp.zeros((B, 1)), s_out[:, :-1]], 1)
    seg = jnp.where(first & restarted[:, None], 0.0, jnp.clip(s_in - prev, 0.0, None))
    seg = jnp.where(valid, seg, 0.0)
    a = T["sdir"][js]
    da = jnp.sum(d[:, None] * a, -1)
    ua = jnp.sqrt(jnp.clip(1 - da ** 2, 1e-12, None))
    gas = s_out - s_in
    q = 0.5 * gas * ua
    wall = 2 * (jnp.sqrt(cfg.r_out ** 2 - cfg.r_in ** 2 + q ** 2) - q) / ua
    nx = jnp.cross(jnp.broadcast_to(d[:, None], a.shape), a)
    b = jnp.abs(jnp.sum((o[:, None] - T["smid"][js]) * nx, -1)) / jnp.maximum(jnp.linalg.norm(nx, axis=-1), 1e-12)
    c_out = 2 * jnp.sqrt(jnp.clip(cfg.rw ** 2 - b ** 2, 0, None)) / ua
    c_in = 2 * jnp.sqrt(jnp.clip(cfg.rw_in ** 2 - b ** 2, 0, None)) / ua
    hitw = gas > 0
    lens = jnp.stack([wall * (1 - cfg.f_metal), wall * cfg.f_metal, gas, jnp.where(hitw, c_in, 0.0),
                      jnp.where(hitw, c_out - c_in, 0.0)], -1) * valid[..., None]
    x_own = _dot(lens, T["thin_invx0"])
    x_str = jnp.concatenate([jnp.zeros((B, 1)), x_own[:, :-1]], 1)
    h_mid = jnp.concatenate([jnp.zeros((B, 1)), 0.5 * (s_in + s_out)[:, :-1]], 1)
    trig = (jnp.cumsum(x_str, 1) >= kink_xx0) & valid
    f_thin = jnp.where(trig.any(1), jnp.argmax(trig, 1), nval)
    pa = o[:, None] + d[:, None] * prev[..., None]
    pb = o[:, None] + d[:, None] * (prev + seg)[..., None]
    wa_, wb_ = pa - T["c"], pb - T["c"]
    ra2 = (wa_ @ T["e1"]) ** 2 + (wa_ @ T["e2"]) ** 2
    rb2 = (wb_ @ T["e1"]) ** 2 + (wb_ @ T["e2"]) ** 2
    flag = jnp.maximum(ra2, rb2) > cfg.rflag2
    if cfg.has_calo:
        flag = flag | (jnp.all(jnp.maximum(pa, pb) >= T["calo_lo"], -1) & jnp.all(jnp.minimum(pa, pb) <= T["calo_hi"], -1))
    cand = flag & valid & (seg > 0) & (k <= f_thin[:, None])
    rows = {"js": js, "s_in": s_in, "s_out": s_out, "valid": valid, "nval": nval, "ntot": ntot, "prev": prev,
            "seg": seg, "lens": lens, "x_str": x_str, "h_mid": h_mid, "f_thin": f_thin, "cand": cand, "pa": pa}
    cutl = real & (cut < jnp.inf)
    stats = jnp.stack([jnp.sum(hitw & (c_out > 0) & valid & ~cutl[:, None]), jnp.sum(cutl)])   # wire hits, cut lanes
    return rows, jnp.stack([jnp.sum(cand), jnp.sum(cutl)]), stats, cutl


@partial(jax.jit, static_argnames=("CB",))
def _cut_lanes(lane, cutl, dummy, CB):
    """Positions in the batch (padding: out of bounds) and lanes (padding: the dummy lane) of the cut lanes."""
    pos = jnp.nonzero(cutl, size=CB, fill_value=cutl.shape[0])[0]
    pv = jnp.arange(CB) < jnp.sum(cutl)
    return pos, jnp.where(pv, lane[jnp.minimum(pos, cutl.shape[0] - 1)], dummy)


@jax.jit
def _splice_rows(rows, sub, pos, stats, sub_stats):
    """Rows of the cut lanes from the every-box pass written over the nearest-box ones."""
    rows = {k: v.at[pos].set(sub[k], mode="drop") for k, v in rows.items()}
    return rows, jnp.sum(rows["cand"]), stats.at[0].add(sub_stats[0])


@partial(jax.jit, static_argnames=("C", "K"))
def _cand_segments(rows, Sd, lanes, C, K):
    flat = rows["cand"].reshape(-1)
    idx = jnp.nonzero(flat, size=C, fill_value=0)[0]
    v = jnp.arange(C) < flat.sum()
    lane, row = idx // K, idx % K
    d = Sd[lanes]
    return lane, row, rows["pa"][lane, row], d[lane], jnp.where(v, rows["seg"][lane, row], 0.0), v


@partial(jax.jit, static_argnames=("B",))
def _first_support(lane, row, col, v, invx0, f_thin, nval, kink_sup_xx0, B):
    """First row holding support per lane (x/X0 > 1e-9), its column; the kink row fm = min(thin trigger, that row)."""
    x = _dot(col, invx0)
    hit = v & (x > 1e-9)
    K_big = jnp.iinfo(jnp.int32).max
    f_sup = jnp.full(B, K_big, jnp.int32).at[jnp.where(hit, lane, B)].min(row.astype(jnp.int32), mode="drop")
    match = hit & (row == f_sup[lane])
    col_fm = jnp.zeros((B, col.shape[1])).at[jnp.where(match, lane, B)].set(col, mode="drop")
    fm = jnp.minimum(f_thin, f_sup)
    has = (f_sup <= f_thin) & (f_sup < nval)
    col_fm = jnp.where(has[:, None], col_fm, 0.0)
    x_fm = _dot(col_fm, invx0)
    cut = has & (x_fm > kink_sup_xx0) if kink_sup_xx0 is not None else jnp.zeros(B, bool)
    return fm.astype(jnp.int32), has, col_fm, x_fm, cut, jnp.sum(cut)


@partial(jax.jit, static_argnames=("cfg", "tail", "tail_k", "R", "KK"), donate_argnames=("S", "done", "rec", "kb"))
def _kink_step(T, cfg, S, done, rec, kb, lane, rows, fm, has, cut, col_fm, x_fm, seg_fm, s_kink, lseg, it, tail,
               tail_k, R, KK):
    """Record the final rows of the pass, kink the group at row fm, update the lanes (TrackerInterior.transport)."""
    N = S["o"].shape[0] - 1
    real = lane < N
    o, d = S["o"][lane], S["d"][lane]
    tk0, lost, trav, xcum = S["tk0"][lane], S["lost"][lane], S["trav"][lane], S["xcum"][lane]
    K = rows["valid"].shape[1]
    k = jnp.arange(K)[None]
    nval = rows["nval"]
    kinked = fm < nval
    lseg = jnp.where(has, lseg, 0.0)
    dead0 = tk0 - lost <= 0.5
    deadk = kinked & (tk0 - lost - lseg <= 0.5)
    nrec_now = jnp.where(dead0, 0, jnp.where(kinked, fm, nval))
    rmask = (k < nrec_now[:, None]) & real[:, None]
    slot = S["nrec"][lane][:, None] + k
    r_over = jnp.sum(rmask & (slot >= R))
    rmask = rmask & (slot < R)
    gl = jnp.where(rmask, lane[:, None], N + 1)          # masked: out of bounds, dropped (no write contention)
    sl = jnp.where(rmask, slot, 0)
    s_in, s_out = rows["s_in"], rows["s_out"]
    rec = dict(rec)
    rec["js"] = rec["js"].at[gl, sl].set(rows["js"], mode="drop")
    rec["pos0"] = rec["pos0"].at[gl, sl].set(o[:, None] + d[:, None] * s_in[..., None], mode="drop")
    rec["pos1"] = rec["pos1"].at[gl, sl].set(o[:, None] + d[:, None] * s_out[..., None], mode="drop")
    rec["dir"] = rec["dir"].at[gl, sl].set(jnp.broadcast_to(d[:, None], (d.shape[0], K, 3)), mode="drop")
    rec["path"] = rec["path"].at[gl, sl].set(trav[:, None] + s_in, mode="drop")
    rec["lost"] = rec["lost"].at[gl, sl].set(jnp.broadcast_to(lost[:, None], s_in.shape), mode="drop")
    done = done.at[gl, jnp.where(rmask, rows["js"], cfg.ns)].set(True, mode="drop")
    nrec = (S["nrec"][lane] + rmask.sum(1)).astype(jnp.int32)
    go = kinked & ~dead0 & ~deadk & real
    # kink group: rows 0..fm; support only in row fm (rows before it hold none, else they would have triggered)
    member = (k <= fm[:, None]) & rows["valid"]
    is_fm = k == fm[:, None]
    xs_m = jnp.where(is_fm, x_fm[:, None], 0.0)
    seg_k = jnp.where(is_fm, seg_fm[:, None], rows["seg"])
    h_seg = s_kink[:, None] - (rows["prev"] + 0.5 * seg_k)
    h_str = s_kink[:, None] - rows["h_mid"]
    xt = rows["x_str"]
    one = jnp.ones(K)
    X = _dot(jnp.where(member, xs_m + xt, 0.0), one)
    S1 = _dot(jnp.where(member, xs_m * h_seg + xt * h_str, 0.0), one)
    S2 = _dot(jnp.where(member, xs_m * h_seg ** 2 + xt * h_str ** 2, 0.0), one)
    w1, w2 = S1 / jnp.maximum(X, 1e-30), S2 / jnp.maximum(X, 1e-30)
    p_k = _p_of(tk0 - lost - lseg)
    V = _highland_var(p_k, xcum, xcum + X)
    ki = jax.vmap(jax.random.fold_in)(S["key"][lane], jnp.full(lane.shape, it, jnp.uint32))
    z = jax.vmap(lambda kk: jax.random.normal(jax.random.fold_in(kk, KEY_CORE), (4,), jnp.float64))(ki)
    dth, dy, a = _kink_core(z, V, w1, w2)
    ntail = jnp.zeros((), jnp.int32)
    tover = jnp.zeros((), jnp.int32)
    if tail:
        lthin = _dot(jnp.moveaxis(jnp.where(((k < fm[:, None]) & rows["valid"])[..., None], rows["lens"], 0.0), 1, 2), one)
        Lall = jnp.concatenate([col_fm, lthin], 1)
        tt, tover = _group_tail(T, jax.vmap(lambda kk: jax.random.fold_in(kk, KEY_TAIL))(ki), Lall, p_k, a, tail_k)
        ntail = jnp.sum(go & jnp.any(tt != 0, 1))
        dth = dth + tt
        dy = dy + tt * w1[:, None]
    back = jnp.where(cut, 0.0, jnp.minimum(st.KINK_BACKOFF_MM, 0.5 * seg_fm))
    end = o + d * (s_kink - back)[:, None]
    pos, dn = _kink(end, d, dth, dy)
    pos, pover = _exit_gas_backwards(T, cfg, pos, dn)
    ks = S["nkink"][lane]
    kmask = go & (ks < KK)
    k_over = jnp.sum(go & (ks >= KK))
    gk, sk = jnp.where(kmask, lane, N + 1), jnp.where(kmask, ks, 0)
    kb = dict(kb)
    kb["end"] = kb["end"].at[gk, sk].set(end, mode="drop")
    kb["pos"] = kb["pos"].at[gk, sk].set(pos, mode="drop")
    kb["dir"] = kb["dir"].at[gk, sk].set(dn, mode="drop")
    kb["path"] = kb["path"].at[gk, sk].set(trav + s_kink - back, mode="drop")
    kb["p"] = kb["p"].at[gk, sk].set(p_k, mode="drop")
    S = dict(S)
    g = go[:, None]
    lw = jnp.where(real, lane, N + 1)                       # padding lanes: dropped
    S["o"] = S["o"].at[lw].set(jnp.where(g, pos, o), mode="drop")
    S["d"] = S["d"].at[lw].set(jnp.where(g, dn, d), mode="drop")
    S["restarted"] = S["restarted"].at[lw].set(jnp.where(go, ~cut, S["restarted"][lane]), mode="drop")
    S["trav"] = S["trav"].at[lw].set(jnp.where(go, trav + s_kink - back, trav), mode="drop")
    S["lost"] = S["lost"].at[lw].set(jnp.where(go, lost + lseg, lost), mode="drop")
    S["xcum"] = S["xcum"].at[lw].set(jnp.where(go, xcum + X, xcum), mode="drop")
    S["nkink"] = S["nkink"].at[lw].set(jnp.where(kmask, ks + 1, ks), mode="drop")
    S["nrec"] = S["nrec"].at[lw].set(nrec, mode="drop")
    S["active"] = S["active"].at[lw].set(go, mode="drop")
    row_over = jnp.sum(real & ~kinked & (rows["ntot"] > K))
    stats = jnp.stack([jnp.sum(go), ntail, jnp.sum(real & (dead0 | deadk)), r_over, k_over, row_over, pover, tover])
    return S, done, rec, kb, stats, jnp.sum(S["active"])


@partial(jax.jit, static_argnames=("B",))
def _fm_segments(rows, Sd, lanes, fm, B):
    """Segment origin, direction and length of row fm per lane."""
    i = jnp.arange(B)
    f = jnp.minimum(fm, rows["seg"].shape[1] - 1)
    return rows["pa"][i, f], Sd[lanes], rows["seg"][i, f], rows["prev"][i, f], rows["s_in"][i, f]


@partial(jax.jit, static_argnames=("B",))
def _active_lanes(active, B):
    """The active lanes (the dummy lane excluded), padded with the dummy lane to B, and their validity."""
    a = active[:-1]
    return jnp.nonzero(a, size=B, fill_value=a.shape[0])[0], jnp.arange(B) < jnp.sum(a)


@partial(jax.jit, static_argnames=("B", "m_sup"))
def _no_support(f_thin, B, m_sup):
    return f_thin.astype(jnp.int32), jnp.zeros(B, bool), jnp.zeros((B, m_sup)), jnp.zeros(B), jnp.zeros(B, bool), \
        jnp.zeros((), jnp.int32)


@partial(jax.jit, static_argnames=("ksx",))
def _cut_target(x_fm, ksx):
    return x_fm / jnp.ceil(x_fm / ksx)


@jax.jit
def _cut_apply(cut, cc, col_fm, x_fm, seg_fm, prev_fm, seg_c, s_kink, invx0):
    return (jnp.where(cut[:, None], cc, col_fm), jnp.where(cut, _dot(cc, invx0), x_fm), jnp.where(cut, seg_c, seg_fm),
            jnp.where(cut, prev_fm + seg_c, s_kink))


def _iter_keys(S, lane, it):
    return jax.vmap(jax.random.fold_in)(S["key"][lane], jnp.full(lane.shape, it, jnp.uint32))


@jax.jit
def _slab_inputs(S, lane, it, has, col_fm):
    """Support column, |p| and slab keys of the kink row (per lane, pass `it`)."""
    ki = _iter_keys(S, lane, it)
    return (jnp.where(has[:, None], col_fm, 0.0), _p_of(S["tk0"][lane] - S["lost"][lane]),
            jax.vmap(lambda kk: jax.random.fold_in(kk, KEY_SLAB))(ki))


@jax.jit
def _exit_inputs(S, lane):
    return S["o"][lane], S["d"][lane], S["length"][lane] - S["trav"][lane]


@jax.jit
def _exit_go(col, invx0, v):
    x = _dot(col, invx0)
    return v & (x > 1e-6), jnp.minimum(x, st.KINK_SUP_XX0)


@jax.jit
def _exit_slab_inputs(S, lane, it, go, col):
    ki = _iter_keys(S, lane, it)
    return jnp.where(go[:, None], col, 0.0), S["p"][lane], jax.vmap(lambda kk: jax.random.fold_in(kk, KEY_SLAB))(ki)


# --------------------------------------------------------------------------- exit scatter kink

@partial(jax.jit, static_argnames=("cfg", "tail", "tail_k", "KX"), donate_argnames=("S", "kb"))
def _exit_step(T, cfg, S, kb, lane, go, seg, col, m1, m2, lost, it, tail, tail_k, KX):
    """One exit_scatter kink of the lanes `lane` (TrackerInterior.exit_scatter after the support column and loss)."""
    N = S["o"].shape[0] - 1
    real = (lane < N) & go
    o, d, p, xc, trav, length = S["o"][lane], S["d"][lane], S["p"][lane], S["xc"][lane], S["trav"][lane], S["length"][lane]
    invx0 = T["invx0"][:cfg.m_sup]
    X, sm1, sm2 = _dot(col, invx0), _dot(m1, invx0), _dot(m2, invx0)
    tk = jnp.sqrt(p ** 2 + MMU ** 2) - MMU
    alive = tk - lost > 0.5
    pk = _p_of(jnp.maximum(tk - lost, 0.0))
    pm = 0.5 * (p + pk)
    V = _highland_var(jnp.maximum(pm, 1.0), xc, xc + X)
    Xs = jnp.maximum(X, 1e-30)
    w1 = jnp.clip(seg - sm1 / Xs, 0.0, seg)
    w2 = jnp.clip(seg ** 2 - 2 * seg * sm1 / Xs + sm2 / Xs, w1 ** 2, seg ** 2)
    ki = jax.vmap(jax.random.fold_in)(S["key"][lane], jnp.full(lane.shape, it, jnp.uint32))
    z = jax.vmap(lambda kk: jax.random.normal(jax.random.fold_in(kk, KEY_CORE), (4,), jnp.float64))(ki)
    dth, dy, a = _kink_core(z, V, w1, w2)
    tover = jnp.zeros((), jnp.int32)
    if tail:
        Lall = jnp.concatenate([col, jnp.zeros((col.shape[0], cfg.m_all - cfg.m_sup))], 1)
        tt, tover = _group_tail(T, jax.vmap(lambda kk: jax.random.fold_in(kk, KEY_TAIL))(ki), Lall,
                                jnp.maximum(pm, 1.0), a, tail_k)
        dth = dth + tt
        dy = dy + tt * w1[:, None]
    th = jnp.hypot(dth[:, 0], dth[:, 1])
    f = jnp.where(th > st.EXIT_KINK_MAX, st.EXIT_KINK_MAX / jnp.where(th > 0, th, 1.0), 1.0)[:, None]
    dth, dy = dth * f, dy * f
    dth = jnp.where(alive[:, None], dth, 0.0)
    dy = jnp.where(alive[:, None], dy, 0.0)
    end = o + d * seg[:, None]
    pos, dn = _kink(end, d, dth, dy)
    rec = real & alive
    ks = S["nk"][lane]
    kmask = rec & (ks < KX)
    gk, sk = jnp.where(kmask, lane, N + 1), jnp.where(kmask, ks, 0)
    kb = dict(kb)
    kb["end"] = kb["end"].at[gk, sk].set(end, mode="drop")
    kb["pos"] = kb["pos"].at[gk, sk].set(pos, mode="drop")
    kb["dir"] = kb["dir"].at[gk, sk].set(dn, mode="drop")
    kb["path"] = kb["path"].at[gk, sk].set(trav + seg, mode="drop")
    kb["p"] = kb["p"].at[gk, sk].set(pk, mode="drop")
    S = dict(S)
    lr = jnp.where(real, lane, N + 1)                       # padding and finished lanes: dropped
    S["o"] = S["o"].at[lr].set(pos, mode="drop")
    S["d"] = S["d"].at[lr].set(dn, mode="drop")
    S["p"] = S["p"].at[lr].set(pk, mode="drop")
    S["xc"] = S["xc"].at[lr].set(xc + X, mode="drop")
    S["trav"] = S["trav"].at[lr].set(trav + seg, mode="drop")
    S["nk"] = S["nk"].at[gk].set(ks + 1, mode="drop")
    S["active"] = S["active"].at[jnp.where(lane < N, lane, N + 1)].set(real & alive & (trav + seg < length - 1.0), mode="drop")
    return S, kb, jnp.stack([jnp.sum(rec & (ks >= KX)), tover]), jnp.sum(S["active"])


# --------------------------------------------------------------------------- the device interior

class DeviceInterior:
    """Device tables of a TrackerInterior and the straws, and the host loops that drive the kernels."""

    THIN = (materials.MYLAR, materials.ALUMINIUM_6061, st.STRAW_GAS, materials.TUNGSTEN, materials.GOLD)

    def __init__(self, interior, geom, wire=True):
        sh = interior.shells if interior is not None else None
        coax = sh.coax if sh is not None else None
        if coax is not None and coax.other:
            raise NotImplementedError("jax backend: non-coaxial tubes in the tracker interior")
        names = set()
        if coax is not None:
            names |= {str(m) for g in coax.groups for m in g["mat"]}
        calo = sh.calo if sh is not None else None
        if calo is not None:
            names |= {str(m) for m in calo.mats}
        self.sup_names = sorted(names)
        self.sup = [materials.get(m) for m in self.sup_names]
        mats = self.sup + list(self.THIN)
        mi = {m: i for i, m in enumerate(self.sup_names)}
        M_sup, M = len(self.sup), len(mats)
        x0 = np.array([eloss.radiation_length_mm(m) for m in mats])
        coef = lambda m, e: 0.157 * tr._zz_over_a(m, e) * m.rho * 0.1
        a13 = np.array([sum(w * materials.ELEMENTS[e][1] ** (1 / 3) for e, w in m.fractions.items()) for m in mats])
        mc = [eloss_jax.consts(m) for m in mats]
        T = {"invx0": 1.0 / x0, "coef_nuc": np.array([coef(m, False) for m in mats]),
             "coef_el": np.array([coef(m, True) - coef(m, False) for m in mats]), "a13": a13}
        self.MC = eloss_jax.MatConsts(*(jnp.asarray(np.array([getattr(c, f) for c in mc])) for f in eloss_jax.MatConsts._fields))
        # straws (+ one dummy straw that no line crosses) and panel boxes (+ one dummy panel)
        ns = len(geom.mid)
        T["smid"] = np.r_[geom.mid, [[1e12, 1e12, 1e12]]]
        T["sdir"] = np.r_[geom.dir, [[1.0, 0.0, 0.0]]]
        T["shalf"] = np.r_[geom.halflen, [0.0]]
        lo, hi = crossings.panel_boxes(geom)
        T["plo"], T["phi"] = lo, hi
        straws_of = np.argsort(geom.upanel, kind="stable").reshape(lo.shape[0], -1)
        T["straws_of"] = np.r_[straws_of, np.full((1, straws_of.shape[1]), ns)].astype(np.int32)
        r_in = geom.radius
        r_out = geom.wall_rmax if geom.wall_rmax is not None else r_in + 0.015
        f_metal = min(max(geom.metal_t / max(r_out - r_in, 1e-9), 0.0), 1.0)
        rw, rw_in = (geom.wire_r, max(geom.wire_r - geom.wire_plate_t, 0.0)) if wire else (0.0, 0.0)
        T["thin_invx0"] = 1.0 / x0[M_sup:]
        # coaxial shells: per group, unique z-intervals sorted by z0, the shells of each interval padded
        G = len(coax.groups) if coax is not None else 0
        groups = []
        for g in (coax.groups if coax is not None else []):
            pairs = np.c_[g["z0"], g["z1"]]
            uz, inv = np.unique(pairs, axis=0, return_inverse=True)
            inv = inv.ravel()
            members = [np.flatnonzero(inv == q) for q in range(len(uz))]
            groups.append((g, uz, members))
        I = max([len(u) for _, u, _ in groups] + [1])
        Smax = max([len(m) for _, _, ms in groups for m in ms] + [1])
        Z0, Z1, ZRUN = np.full((max(G, 1), I), np.inf), np.full((max(G, 1), I), -np.inf), np.full((max(G, 1), I), np.inf)
        P0, DP = np.zeros((max(G, 1), I, Smax)), np.full((max(G, 1), I, Smax), -1.0)
        FULL, MAT = np.zeros((max(G, 1), I, Smax), bool), np.full((max(G, 1), I, Smax), M_sup, np.int32)
        rmin, rmax = np.zeros(max(G, 1)), np.zeros(max(G, 1))
        for gi, (g, uz, members) in enumerate(groups):
            rmin[gi], rmax[gi] = g["rmin"], g["rmax"]
            n = len(uz)
            Z0[gi, :n], Z1[gi, :n] = uz[:, 0], uz[:, 1]
            ZRUN[gi, :n] = np.maximum.accumulate(uz[:, 1])
            for q, mem in enumerate(members):
                P0[gi, q, :len(mem)] = g["p0"][mem]
                DP[gi, q, :len(mem)] = g["dp"][mem]
                FULL[gi, q, :len(mem)] = g["dp"][mem] >= 2 * np.pi - 1e-9
                MAT[gi, q, :len(mem)] = [mi[str(m)] for m in g["mat"][mem]]
        T.update({"Z0": Z0, "Z1": Z1, "ZRUN": ZRUN, "P0": P0, "DP": DP, "FULL": FULL, "MAT": MAT, "rmin": rmin,
                  "rmax": rmax})
        if coax is not None:
            T.update({"c": coax.c, "e1": coax.e1, "e2": coax.e2, "ax": coax.a})
        else:
            T.update({"c": np.zeros(3), "e1": np.array([1.0, 0, 0]), "e2": np.array([0, 1.0, 0]), "ax": np.array([0, 0, 1.0])})
        rflag = (min(rmin[:G]) if G else 1e9) - 1e-3
        self.calo_sets = []
        if calo is not None:
            T["calo_lo"], T["calo_hi"] = sh.calo_lo, sh.calo_hi
            for sign, ps in calo.sets:
                gm = np.array([mi[str(ps.mats[q])] for q in ps.mat], np.int32)
                self.calo_sets.append(tuple(jnp.asarray(x) for x in (ps.lo, ps.hi, ps.P, gm)) + (float(sign),))
        self.calo_boxes = tuple((lo_, hi_) for lo_, hi_, _, _, _ in self.calo_sets)
        self.calo_arrays = tuple((lo_, hi_, P_, gm_) for lo_, hi_, P_, gm_, _ in self.calo_sets)
        self.calo_signs = tuple(int(sg) for *_, sg in self.calo_sets)
        self.T = {k: jnp.asarray(v) for k, v in T.items()}
        self.cfg = Cfg(ns=ns, npanel=lo.shape[0], m_sup=M_sup, m_all=M, ngroup=G, nsub=int(interior.nsub) if interior is not None else 8,
                       rflag2=float(rflag) ** 2, r_in=float(r_in), r_out=float(r_out), f_metal=float(f_metal),
                       rw=float(rw), rw_in=float(rw_in), has_calo=calo is not None)
        self.invx0_sup = self.T["invx0"][:M_sup]
        self.geom = geom
        self._stats = {}

    # ---- support column on device segments (o, d, L, valid [C]); returns [C, M_sup] (and the moments)
    def column(self, o, d, L, valid, moments=False):
        """Host syncs: the work-list size (and, near the calorimeter, its boxes hit per piece set); the rest is fused."""
        cfg = self.cfg
        win, ntot, near, nnear = _coax_windows(self.T, cfg, o, d, L, valid)
        ntot, nnear = (int(x) for x in jax.device_get((ntot, nnear)))
        TB = _bucket(ntot, 256)
        if not (cfg.has_calo and nnear):
            return _coax_column(self.T, cfg, win, TB, moments)
        CB = _bucket(nnear, 16)
        idx, v, so, sd, sL, nh = _calo_prep(self.calo_boxes, o, d, L, near, CB)
        out = _coax_items(self.T, cfg, win, TB, moments)
        acc = None   # one call per piece set hit (a fused call per hit pattern compiles too many variants)
        for (lo, hi, P, gm), sign, h in zip(self.calo_arrays, self.calo_signs, jax.device_get(nh)):
            if not int(h):
                continue
            r, over = _calo_set(lo, hi, P, gm, sign, so, sd, sL, v, cfg.m_sup, moments, _bucket(int(h), 8))
            self._stat("calo_overflow", over)
            acc = r if acc is None else _tree_add(acc, r)
        if acc is None:
            return _dequant(out, moments)
        return _merge_dequant(out, acc, idx, v, o.shape[0], moments)

    def xsup(self, col):
        return _xsup(col, self.invx0_sup)

    def split_length(self, o, d, seg, target, valid):
        """TrackerInterior._split_length (bisection over SPLIT_PROBES fractions)."""
        n = st.TrackerInterior.SPLIT_PROBES
        lo, hi, mid, Lp = _bisect_start(seg, n)
        while n > 1:
            x = self.xsup(self.column(o, d, Lp, valid))
            n //= 2
            lo, hi, mid, Lp = _bisect_step(lo, hi, mid, x, target, seg)
        return _bisect_len(seg, hi)

    def slab(self, col, p, keys):
        """Sum over support materials of transport.slab_loss(mat, max(p, 1), col[:, m]) per row [R]."""
        R, M = col.shape
        npairs = int(_count_pos(col))
        if not npairs:
            return jnp.zeros(R)
        C = _bucket(npairs, 64)
        row, mat, L, k, v, pp, carry, idx = _slab_pairs(col, keys, p, C)
        while True:
            carry, todo, nt = _slab_iter(self.MC, mat, pp, L, k, carry, idx, SLICE_ROUNDS)
            nt = int(nt)
            if not nt:
                break
            idx = _slab_next(idx, todo, _bucket(nt, 64))
        return _slab_done(self.MC, mat, pp, carry, row, v, R, M)

    def _stat(self, name, x):
        self._stats.setdefault(name, []).append(x)

    # ---- TrackerInterior.transport
    def transport(self, key, o, d, p, tail=True, tail_k=None, kink_xx0=st.KINK_XX0, kink_sup_xx0=st.KINK_SUP_XX0,
                  max_iter=400, lanes=None):
        """Straw crossings of muons scattered and slowed inside the tracker (TrackerInterior.transport with the default
        gap, wire, lever arms and no msc_cal). Returns (rows dict as numpy's, last_stats, last_kinks, last_final);
        muons are processed in chunks of `lanes` (the random streams do not depend on it, keys are per muon; the float
        rounding of the vectorised math can, so the outputs are bit-identical per seed for one device and lane count)."""
        n = len(p)
        lanes = lanes or LANES
        self._stats = {}
        parts = [self._transport_chunk(key, o[s:s + lanes], d[s:s + lanes], p[s:s + lanes], s, tail, tail_k, kink_xx0,
                                       kink_sup_xx0, max_iter) for s in range(0, max(n, 1), lanes)] if n else []
        st_ = {"kinks": 0, "tail_scatters": 0, "stopped": 0, "iterations": 0, "capped": 0, "wire_hits": 0,
               "support_splits": 0}
        for q in parts:
            for k_, v in q[1].items():
                st_[k_] = max(st_.get(k_, 0), v) if k_ == "iterations" else st_.get(k_, 0) + v
        for name, xs in self._stats.items():
            st_[name] = st_.get(name, 0) + int(sum(int(x) for x in jax.device_get(xs)))
        cat = lambda key_, i: np.concatenate([q[i][key_] for q in parts]) if parts else None
        z3 = np.zeros((0, 3))
        if not parts:
            rows = {"it": np.zeros(0, np.int64), "js": np.zeros(0, np.int64), "pos0": z3, "pos1": z3, "dir": z3,
                    "path": np.zeros(0), "lost": np.zeros(0)}
            kinks = {"it": np.zeros(0, np.int64), "end": z3, "pos": z3, "dir": z3, "path": np.zeros(0), "p": np.zeros(0)}
            final = {"o": z3, "d": z3, "p": np.zeros(0), "path": np.zeros(0), "xcum": np.zeros(0)}
            return rows, st_, kinks, final
        rows = {k_: cat(k_, 0) for k_ in parts[0][0]}
        kinks = {k_: cat(k_, 2) for k_ in parts[0][2]}
        final = {k_: cat(k_, 3) for k_ in parts[0][3]}
        order = np.lexsort((rows["path"], rows["it"]))
        return {k_: v[order] for k_, v in rows.items()}, st_, kinks, final

    def _transport_chunk(self, key, o, d, p, base, tail, tail_k, kink_xx0, kink_sup_xx0, max_iter):
        cfg, T = self.cfg, self.T
        N = len(p)
        pad = lambda x, v=0.0: jnp.asarray(np.concatenate([x, np.full((1,) + x.shape[1:], v, x.dtype)]))
        tk0 = np.sqrt(p ** 2 + MMU ** 2) - MMU
        kt = jax.random.fold_in(key, KEY_TRANSPORT)
        S = {"o": pad(o), "d": jnp.asarray(np.r_[d, [[0.0, -1.0, 0.0]]]), "tk0": pad(tk0), "lost": jnp.zeros(N + 1), "trav": jnp.zeros(N + 1),
             "xcum": jnp.zeros(N + 1), "restarted": jnp.zeros(N + 1, bool),
             "active": jnp.asarray(np.r_[np.ones(N, bool), False]), "nrec": jnp.zeros(N + 1, jnp.int32),
             "nkink": jnp.zeros(N + 1, jnp.int32),
             "key": jax.vmap(lambda i: jax.random.fold_in(kt, i))(jnp.arange(base, base + N + 1, dtype=jnp.uint32))}
        done = jnp.zeros((N + 1, cfg.ns + 1), bool)
        rec = {"js": jnp.zeros((N + 1, R_CAP), jnp.int32), "pos0": jnp.zeros((N + 1, R_CAP, 3)),
               "pos1": jnp.zeros((N + 1, R_CAP, 3)), "dir": jnp.zeros((N + 1, R_CAP, 3)),
               "path": jnp.zeros((N + 1, R_CAP)), "lost": jnp.zeros((N + 1, R_CAP))}
        kb = {"end": jnp.zeros((N + 1, KK_CAP, 3)), "pos": jnp.zeros((N + 1, KK_CAP, 3)),
              "dir": jnp.zeros((N + 1, KK_CAP, 3)), "path": jnp.zeros((N + 1, KK_CAP)), "p": jnp.zeros((N + 1, KK_CAP))}
        stats = {"kinks": 0, "tail_scatters": 0, "stopped": 0, "iterations": 0, "capped": 0, "wire_hits": 0,
                 "support_splits": 0}
        acc = []
        ksx = float(kink_sup_xx0) if (kink_sup_xx0 is not None and kink_sup_xx0 > 0) else None
        n_act = N
        for it in range(max_iter):
            if not n_act:
                break
            stats["iterations"] += 1
            B = _bucket(n_act, 32)
            lane, _ = _active_lanes(S["active"], B)
            rows, cnt, pst, cutl = _pass(T, cfg, S, lane, done, kink_xx0, K_ROWS, P=P_CAP)
            ncand, ncutl = (int(x) for x in jax.device_get(cnt))
            if ncutl:   # lines with more panel boxes than P_CAP: their rows again with every box
                pos, sl = _cut_lanes(lane, cutl, N, _bucket(ncutl, 8))
                sub, _, sst, _ = _pass(T, cfg, S, sl, done, kink_xx0, K_ROWS, P=cfg.npanel)
                rows, ncand, pst = _splice_rows(rows, sub, pos, pst, sst)
                ncand = int(ncand)
            acc.append(("pass", pst))
            if ncand:
                C = _bucket(ncand, 32)
                cl, cr, so, sd, sL, cv = _cand_segments(rows, S["d"], lane, C, K_ROWS)
                col = self.column(so, sd, sL, cv)
                fm, has, col_fm, x_fm, cut, ncut = _first_support(cl, cr, col, cv, self.invx0_sup, rows["f_thin"],
                                                                  rows["nval"], ksx, B)
            else:
                fm, has, col_fm, x_fm, cut, ncut = _no_support(rows["f_thin"], B, cfg.m_sup)
            so, sd, seg_fm, prev_fm, s_in_fm = _fm_segments(rows, S["d"], lane, fm, B)
            s_kink = s_in_fm
            ncut = int(ncut)
            if ncut:
                stats["support_splits"] += ncut
                seg_c = self.split_length(so, sd, seg_fm, _cut_target(x_fm, ksx), cut)
                cc = self.column(so, sd, seg_c, cut)
                col_fm, x_fm, seg_fm, s_kink = _cut_apply(cut, cc, col_fm, x_fm, seg_fm, prev_fm, seg_c, s_kink,
                                                          self.invx0_sup)
            col_in, pk, skeys = _slab_inputs(S, lane, it, has, col_fm)
            lseg = self.slab(col_in, pk, skeys)
            S, done, rec, kb, kst, n_act = _kink_step(T, cfg, S, done, rec, kb, lane, rows, fm, has, cut, col_fm, x_fm,
                                                      seg_fm, s_kink, lseg, it, tail, tail_k, R_CAP, KK_CAP)
            acc.append(("kink", kst))
            n_act = int(n_act)
        else:
            stats["capped"] += n_act
        for kind, x in jax.device_get(acc):
            x = np.asarray(x)
            if kind == "pass":
                stats["wire_hits"] += int(x[0])
                self._stat("panel_cut", int(x[1]))   # lines with more boxes than P_CAP (no loss: see _line_straws)
            else:
                stats["kinks"] += int(x[0])
                stats["tail_scatters"] += int(x[1])
                stats["stopped"] += int(x[2])
                for name, v in zip(("rec_overflow", "kink_overflow", "row_overflow", "point_overflow", "tail_overflow"), x[3:]):
                    self._stat(name, int(v))
        # pull rows, kinks and the final state back
        h = jax.device_get({"S": S, "rec": rec, "kb": kb})
        Sh, rh, kh = h["S"], h["rec"], h["kb"]
        nrec = np.minimum(np.asarray(Sh["nrec"][:N]), R_CAP)
        m = np.arange(R_CAP)[None] < nrec[:, None]
        it_ = np.broadcast_to(np.arange(N)[:, None], m.shape)[m] + base
        rows = {"it": it_.astype(np.int64), "js": np.asarray(rh["js"][:N])[m].astype(np.int64),
                "pos0": np.asarray(rh["pos0"][:N])[m], "pos1": np.asarray(rh["pos1"][:N])[m],
                "dir": np.asarray(rh["dir"][:N])[m], "path": np.asarray(rh["path"][:N])[m],
                "lost": np.asarray(rh["lost"][:N])[m]}
        nk = np.minimum(np.asarray(Sh["nkink"][:N]), KK_CAP)
        mk = np.arange(KK_CAP)[None] < nk[:, None]
        kinks = {"it": (np.broadcast_to(np.arange(N)[:, None], mk.shape)[mk] + base).astype(np.int64),
                 **{k_: np.asarray(kh[k_][:N])[mk] for k_ in ("end", "pos", "dir", "path", "p")}}
        tk_end = np.asarray(Sh["tk0"][:N]) - np.asarray(Sh["lost"][:N])
        final = {"o": np.asarray(Sh["o"][:N]), "d": np.asarray(Sh["d"][:N]),
                 "p": st._p_of(np.maximum(tk_end, 0.0)), "path": np.asarray(Sh["trav"][:N]),
                 "xcum": np.asarray(Sh["xcum"][:N])}
        return rows, stats, kinks, final

    # ---- TrackerInterior.exit_scatter
    def exit_scatter(self, key, o, d, p, xcum, length=st.EXIT_SCATTER_MM, tail=True, tail_k=None, lanes=None):
        n = len(p)
        lanes = lanes or LANES
        self._stats = {}
        parts = [self._exit_chunk(key, o[s:s + lanes], d[s:s + lanes], p[s:s + lanes], xcum[s:s + lanes], s, length,
                                  tail, tail_k) for s in range(0, n, lanes)]
        self.last_exit_stats = {name: int(sum(int(x) for x in jax.device_get(xs))) for name, xs in self._stats.items()}
        z3 = np.zeros((0, 3))
        if not parts:
            return {"it": np.zeros(0, np.int64), "end": z3, "pos": z3, "dir": z3, "path": np.zeros(0), "p": np.zeros(0)}
        return {k_: np.concatenate([q[k_] for q in parts]) for k_ in parts[0]}

    def _exit_chunk(self, key, o, d, p, xcum, base, length, tail, tail_k):
        cfg, T = self.cfg, self.T
        N = len(p)
        pad = lambda x, v=0.0: jnp.asarray(np.concatenate([x, np.full((1,) + x.shape[1:], v, x.dtype)]))
        oj, dj = pad(o), jnp.asarray(np.r_[d, [[0.0, -1.0, 0.0]]])
        L = jnp.minimum(jnp.full(N + 1, float(length)), _envelope_exit(oj, dj) + st.ENVELOPE_PAD_MM)
        kt = jax.random.fold_in(key, KEY_EXIT)
        S = {"o": oj, "d": dj, "p": pad(p), "xc": pad(xcum), "trav": jnp.zeros(N + 1), "length": L,
             "nk": jnp.zeros(N + 1, jnp.int32),
             "active": jnp.concatenate([(pad(p)[:N] > 1.0) & (L[:N] > 1.0), jnp.zeros(1, bool)]),
             "key": jax.vmap(lambda i: jax.random.fold_in(kt, i))(jnp.arange(base, base + N + 1, dtype=jnp.uint32))}
        kb = {"end": jnp.zeros((N + 1, KX_CAP, 3)), "pos": jnp.zeros((N + 1, KX_CAP, 3)),
              "dir": jnp.zeros((N + 1, KX_CAP, 3)), "path": jnp.zeros((N + 1, KX_CAP)), "p": jnp.zeros((N + 1, KX_CAP))}
        acc = []
        n_act = int(jnp.sum(S["active"]))
        for it in range(64):
            if not n_act:
                break
            B = _bucket(n_act, 32)
            lane, v = _active_lanes(S["active"], B)
            ol, dl, left = _exit_inputs(S, lane)
            col = self.column(ol, dl, left, v)
            go, target = _exit_go(col, self.invx0_sup, v)
            seg = self.split_length(ol, dl, left, target, go)
            col, m1, m2 = self.column(ol, dl, seg, go, moments=True)
            col_in, pl, skeys = _exit_slab_inputs(S, lane, it, go, col)
            lost = self.slab(col_in, pl, skeys)
            S, kb, est, n_act = _exit_step(T, cfg, S, kb, lane, go, seg, col, m1, m2, lost, it, tail, tail_k, KX_CAP)
            acc.append(est)
            n_act = int(n_act)
        for x in jax.device_get(acc):
            self._stat("kink_overflow", int(x[0]))
            self._stat("tail_overflow", int(x[1]))
        h = jax.device_get(kb)
        nk = np.minimum(np.asarray(jax.device_get(S["nk"]))[:N], KX_CAP)
        mk = np.arange(KX_CAP)[None] < nk[:, None]
        return {"it": (np.broadcast_to(np.arange(N)[:, None], mk.shape)[mk] + base).astype(np.int64),
                **{k_: np.asarray(h[k_][:N])[mk] for k_ in ("end", "pos", "dir", "path", "p")}}


@partial(jax.jit, static_argnames=("C", "moments"))
def _calo_merge(out, acc, idx, v, C, moments):
    """Clip the signed calorimeter column (SignedPieceSet: length > 1e-9 mm, else 0) and add it to the coaxial one at the
    near segments (integer fixed point)."""
    tgt = jnp.where(v, idx, C)                               # padding: out of bounds, dropped
    if moments:
        ok = acc[0] > 1
        acc = tuple(jnp.where(ok, a, 0) for a in acc)
        return tuple(o_.at[tgt].add(a, mode="drop") for o_, a in zip(out, acc))
    acc = jnp.clip(acc, 0, None)
    return out.at[tgt].add(acc, mode="drop")


@partial(jax.jit, static_argnames=("moments",))
def _dequant(q, moments):
    if moments:
        return tuple(x.astype(jnp.float64) / QSCALE[k] for k, x in enumerate(q))
    return q.astype(jnp.float64) / QSCALE[0]


def device_interior(interior, geom):
    """DeviceInterior of `interior` (cached on it), or of the straws alone for interior=None."""
    holder = interior if interior is not None else geom
    dev = getattr(holder, "_jax_device", None)
    if dev is None:
        dev = DeviceInterior(interior, geom)
        try:
            object.__setattr__(holder, "_jax_device", dev)
        except (AttributeError, TypeError):
            pass
    return dev


def straw_crossings(geom, o, d, chunk=None):
    """crossings.straw_crossings on the device: rows (track, straw, s_in, s_out) sorted by track, s_in."""
    dev = device_interior(None, geom)
    chunk = chunk or max(LANES // 8, 64)   # every panel box per line: [chunk, all straws] candidates
    out = []
    for s in range(0, len(o), chunk):
        oc, dc = o[s:s + chunk], d[s:s + chunk]
        B = _bucket(len(oc), 64)
        op = np.r_[oc, np.zeros((B - len(oc), 3))]
        dp_ = np.r_[dc, np.tile([[0.0, -1.0, 0.0]], (B - len(oc), 1))]
        op[len(oc):, 1] = -1e9                        # padding lines far below the tracker
        js, s_in, s_out, valid, ntot = jax.device_get(_crossings_all(dev.T, dev.cfg, jnp.asarray(op), jnp.asarray(dp_),
                                                                     K_ALL))
        if int(np.sum(ntot > K_ALL)):
            raise RuntimeError("jax straw_crossings: more than K_ALL crossings on a line")
        valid = np.asarray(valid)[:len(oc)]
        it = np.broadcast_to(np.arange(len(oc))[:, None], valid.shape)[valid] + s
        out.append(np.stack([it, np.asarray(js)[:len(oc)][valid], np.asarray(s_in)[:len(oc)][valid],
                             np.asarray(s_out)[:len(oc)][valid]], 1).astype(float))
    if not out:
        return np.zeros((0, 4))
    r = np.concatenate(out)
    return r[np.lexsort((r[:, 2], r[:, 0]))]


@jax.jit
def _gas_eion(key, mu, rank, p, path, c):
    def one(m, r, pp, ll):
        return eloss_jax._sample(jax.random.fold_in(jax.random.fold_in(key, m), r), c, pp, ll, st.delta_cut_mev(), MMU)
    return jax.vmap(one)(mu, rank, p, path)


def sample_gas_eion(key, p, path, mu, chunk=1 << 16):
    """Urban eIon in the straw gas per crossing (eloss_jax), keyed by (muon, crossing rank within the muon)."""
    n = len(p)
    if not n:
        return np.zeros(0)
    first = np.r_[True, mu[1:] != mu[:-1]]
    start = np.flatnonzero(first)
    rank = np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))
    c = eloss_jax.consts(st.STRAW_GAS)
    ke = jax.random.fold_in(key, KEY_EION)
    out = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        B = _bucket(m, 256)
        padf = lambda x, v: np.r_[x[s:s + m], np.full(B - m, v)]
        out.append(np.asarray(_gas_eion(ke, jnp.asarray(padf(mu, 0).astype(np.uint32)),
                                        jnp.asarray(padf(rank, 0).astype(np.uint32)), jnp.asarray(padf(p, 1000.0)),
                                        jnp.asarray(padf(path, 0.0)), c))[:m])
    return np.concatenate(out)


UNSUPPORTED = {"interior_msc_cal": None, "lever_moments": False, "entry_ledger": "legacy", "capture_entry_ledger": False}


def check_options(interior, interior_msc, **kw):
    """Raise for make_steps options the jax backend does not implement (diagnostic knobs; numpy has them)."""
    for k, default in UNSUPPORTED.items():
        if kw.get(k, default) != default:
            raise NotImplementedError(f"jax backend: {k}={kw[k]!r} (use the numpy backend)")
    if interior is not None and not interior_msc:
        raise NotImplementedError("jax backend: --no-interior-msc (loss-only interior); use the numpy backend")
