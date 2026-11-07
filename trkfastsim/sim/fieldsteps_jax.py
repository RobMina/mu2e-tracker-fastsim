"""Device (JAX) twin of the field-on tracker transport of electrons: fieldsteps.transport with the electron loss model.

Opt-in. trkfastsim.sim.fieldsteps (NumPy) stays the default and the reference. This module runs the same chord
transport on the device for electrons (fieldsteps.electron_loss, no single-scattering tail), from start states that are
already device arrays (e.g. the conversion-electron chain's tracker-entry states), and leaves the StrawGasSteps on the
device as fixed-shape per-track record buffers. Per chord, as fieldsteps.transport:

  * chord of the local helix, field at the helix midpoint, length min(chord_mm, sqrt(8 rho sagitta_mm));
  * straw gas crossings of the chord (candidate panels by z, then their 96 straws), the wall chord (outer cylinder
    minus gas), the wire and plating chords at the closest approach;
  * the support column of the coaxial shells (exact z clipping, phi sectors by the same nsub-point sampling as
    world.CoaxialShells.column), only for chords that reach the smallest shell radius;
  * energy loss per material at the chord-start momentum: restricted Moller + Seltzer-Berger mean with Urban
    fluctuations (eloss_jax.sample_loss_electron), deltas above E_CUT (Poisson count, G4MollerBhabha transfer) and
    brem photons above E_CUT (free paths at the running energy, inverse-CDF photon energy), as fieldsteps.electron_loss;
  * Highland core kink at the chord end (cumulative x/X0, lever-arm moments), angle capped at 0.5 rad;
  * a gas crossing cut by a chord end continues on the next chord (one record per passage through a straw).

Rows follow fieldsteps._merge_rows: entry/exit points, summed gas path, mean of the entry and exit momentum vectors,
width, shape, ionisation type, time at the gas entry, the deposited eIon sampled once per passage (Moller mean below
the tracker delta cut, Urban fluctuations: fieldsteps.electron_gas_eion). Positions in the detector frame.

Statistically equivalent to the NumPy transport, not identical per seed (independent random streams). With the loss
and scattering switched off (`eloss=False, msc=False`) the crossings are deterministic and agree with NumPy run the
same way to rounding. dtype float64 (default) or float32: positions are kept in the detector frame (relative to
the tracker origin), so float32 resolves the straws to ~1e-4 mm. Fixed shapes: every cap is counted (overflow statistics must be 0):
  P_CAP candidate panels per chord, K_CHORD crossings per chord, R_CAP records per track, the gathered support
  sub-batch (lanes beyond it wait one iteration unchanged), ND deltas and NB brem photons per material and chord.
"""
import time
from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import ebrem, eioni, eloss, eloss_jax, materials
from mu2efastsim import eslab_jax as SJ

from trkfastsim import crossings
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.sim import steps as st

jax.config.update("jax_enable_x64", True)

ME = eloss.ME
KB = 0.299792458                      # MeV/c per (T mm)
CMM = fs.C_MM_PER_NS
P_CAP = 4        # candidate panel boxes per chord (Run 1A: at most 3 overlap a 10 mm chord in z)
K_CHORD = 8      # gas crossings kept per chord
R_CAP = 256      # records (merged passages) per track
ND, NB = 4, 4    # deltas and brem photons per (track, material) and chord: thin straw materials
ND_SUP, NB_SUP = 24, 32   # the same for the support pieces (up to a chord of solid metal)
SUP_PAIRS = 4    # support (track, material) pairs per gathered lane, on average
SUP_FRAC = 0.125 # lanes per iteration that may take the support column (others wait one iteration)
PAIRS = 6        # (track, material) pairs with material per lane and chord, on average over the bucket
UNROLL = True    # unroll the samplers' fixed-trip loops inside the chord loop (a device loop trip costs a predicate
                 # read per trip on a GPU; unrolled, the chord compiles once into fewer, larger kernels)
MERGE_TOL_F32 = 0.02   # [mm] float32: continuation of a gas crossing on the next chord (path rounding ~5e-4 mm at 5 m)
LIVE, LEFT, STOPPED, CAPPED = -1, 0, 1, 2


class Cfg(NamedTuple):
    """Static configuration (hashable)."""
    chord_mm: float
    sagitta_mm: float
    max_path_mm: float
    r_in: float
    r_out: float
    f_metal: float
    rw: float
    rw_in: float
    m_sup: int
    nsub: int
    rflag: float
    zcap: tuple          # per shell group: z-intervals one chord can overlap
    smax: tuple          # per shell group: shells per z-interval
    grmin: tuple         # per shell group: inner and outer radius [mm]
    grmax: tuple
    eloss: bool
    msc: bool
    bnomc: float
    pstraw: float


def _shell_groups(coax, mi, origin, chord_mm):
    """Per coaxial group (numpy): unique z-intervals sorted by z0 (detector frame), their running max z1 and the
    shells of each interval (phi0, dphi, full ring, material index), as steps_jax.DeviceInterior."""
    out = []
    zc = coax.c[2] - origin[2]
    for g in coax.groups:
        pairs = np.c_[g["z0"], g["z1"]]
        uz, inv = np.unique(pairs, axis=0, return_inverse=True)
        inv = inv.ravel()
        members = [np.flatnonzero(inv == q) for q in range(len(uz))]
        smax = max(len(m) for m in members)
        n = len(uz)
        P0, DP = np.zeros((n, smax)), np.full((n, smax), -1.0)
        FULL, MAT = np.zeros((n, smax), bool), np.zeros((n, smax), np.int32)
        for q, mem in enumerate(members):
            P0[q, :len(mem)] = g["p0"][mem]
            DP[q, :len(mem)] = g["dp"][mem]
            FULL[q, :len(mem)] = g["dp"][mem] >= 2 * np.pi - 1e-9
            MAT[q, :len(mem)] = [mi[str(m)] for m in g["mat"][mem]]
        # most intervals one chord (z extent <= chord_mm) overlaps, from a sweep over the interval edges
        edges = np.unique(np.r_[uz[:, 0], uz[:, 1]])
        zcap = 2
        for z in edges:
            zcap = max(zcap, int(((uz[:, 0] < z + chord_mm + 1.0) & (uz[:, 1] > z - chord_mm - 1.0)).sum()))
        out.append(dict(rmin=float(g["rmin"]), rmax=float(g["rmax"]), Z0=uz[:, 0] + zc, Z1=uz[:, 1] + zc,
                        ZRUN=np.maximum.accumulate(uz[:, 1]) + zc, P0=P0, DP=DP, FULL=FULL, MAT=MAT,
                        zcap=zcap, smax=smax))
    return out


def _material_tables(mats, dt):
    """Per material on the eslab_jax ln T grid: continuous restricted loss (Moller-Bhabha + Seltzer-Berger below
    E_CUT), delta and brem cross sections above E_CUT, brem photon inverse CDF; MatConsts; 1/X0."""
    lnt, T = SJ._grid()
    cut = fs.E_CUT_MEV
    dedx = np.stack([eioni.dedx_restricted(m, T, cut) + ebrem.dedx_restricted(m, T, cut) for m in mats])
    sig_d = np.stack([eioni.xsec_per_mm(m, T, cut) for m in mats])
    sig_b = np.stack([ebrem.xsec_per_mm(m, T, cut) for m in mats])
    brem_x = np.stack([SJ._brem_inverse_cdf(m, T, cut, False) for m in mats])
    cs = [eloss_jax.consts(m) for m in mats]
    mc = eloss_jax.MatConsts(*(jnp.asarray(np.array([getattr(c, f) for c in cs]), dt)
                               for f in eloss_jax.MatConsts._fields))
    invx0 = np.array([1.0 / eloss.radiation_length_mm(m) for m in mats])
    return dict(lnt0=float(lnt[0]), dlnt=float(lnt[1] - lnt[0]), dedx=dedx, sig_d=sig_d, sig_b=sig_b,
                brem_x=brem_x, invx0=invx0), mc


class Tracker:
    """Device tables of the tracker for the field transport: straws and panels, the field grid, the envelope, the
    support shells and the material tables. Built once per geometry and process (host, a few seconds)."""

    THIN = (materials.MYLAR, materials.ALUMINIUM_6061, None, materials.TUNGSTEN, materials.GOLD)   # None: the gas

    def __init__(self, geom, field_grid, interior, envelope, origin, bnom, chord_mm=10.0, sagitta_mm=0.02,
                 max_path_mm=30000.0, wire=True, gas=None, dtype=np.float64, eloss_on=True, msc=True):
        dt = np.dtype(dtype)
        self.dt = dt
        self.origin = np.asarray(origin, float)
        org = self.origin
        gas = st.STRAW_GAS if gas is None else gas
        self.gas = gas
        coax = interior.shells.coax if interior is not None else None
        if coax is not None and coax.other:
            raise NotImplementedError("device field transport: non-coaxial tubes in the tracker interior")
        if interior is not None and interior.shells.calo is not None:
            raise NotImplementedError("device field transport: calorimeter pieces in the tracker interior")
        sup_names = sorted({str(m) for g in coax.groups for m in g["mat"]}) if coax is not None else []
        self.sup_names = sup_names
        thin = [gas if m is None else m for m in self.THIN]
        mats = [materials.get(m) for m in sup_names] + thin
        self.mats = mats
        m_sup = len(sup_names)
        tabs, self.mc = _material_tables(mats, dt)
        # the deposited eIon in the gas: Moller restricted below the tracker delta cut
        lnt, Tg = SJ._grid()
        self.tcut_gas = st.delta_cut_mev(gas)
        tabs["gas_dedx"] = eioni.dedx_restricted(gas, Tg, self.tcut_gas)
        self.gas_mc = eloss_jax.consts(gas)
        groups = _shell_groups(coax, {m: i for i, m in enumerate(sup_names)}, org, chord_mm) if coax is not None else []
        lo, hi, straws_of = crossings.panel_index(geom)
        order = np.argsort(lo[:, 2], kind="stable")
        r_in = geom.radius
        r_out = geom.wall_rmax if geom.wall_rmax is not None else r_in + 0.015
        f_metal = min(max(geom.metal_t / max(r_out - r_in, 1e-9), 0.0), 1.0)
        rw, rw_in = (geom.wire_r, max(geom.wire_r - geom.wire_plate_t, 0.0)) if wire else (0.0, 0.0)
        lo_d, hi_d = lo - org, hi - org
        T = dict(smid=geom.mid - org, sdir=geom.dir, shalf=geom.halflen, plo=lo_d[order], phi=hi_d[order],
                 plo_z=lo_d[order, 2], pstraws=straws_of[order].astype(np.int32),
                 sid=np.asarray(geom.sid, np.int32), plane=np.asarray(geom.plane, np.int32),
                 panel=np.asarray(geom.panel, np.int32), straw=np.asarray(geom.straw, np.int32),
                 upanel=np.asarray(geom.upanel, np.int32))
        self.pwidth = float((hi[:, 2] - lo[:, 2]).max())
        flo, fd, fn, fB = field_grid
        T.update(flo=np.asarray(flo, float) - org, fd=np.asarray(fd, float), fB=np.asarray(fB, float))
        self.fn = tuple(int(v) for v in fn)
        env = envelope
        T.update(env_axis=np.asarray(env.axis, float) - org[:2])
        self.env = (float(env.r), float(env.z0 - org[2]), float(env.z1 - org[2]))
        T.update({k: v for k, v in tabs.items() if k not in ("lnt0", "dlnt")})
        self.lnt0, self.dlnt = tabs["lnt0"], tabs["dlnt"]
        if coax is not None:
            T.update(sc=coax.c - org, se1=coax.e1, se2=coax.e2, sax=coax.a)
        bnom = np.asarray(bnom, float)
        bmag = float(np.linalg.norm(bnom))
        T.update(bdir=bnom / bmag if bmag > 0 else np.array([0.0, 0.0, 1.0]))
        self.T = {k: (jnp.asarray(v, dt) if np.asarray(v).dtype.kind == "f" else jnp.asarray(v)) for k, v in T.items()}
        self.groups = [{k: (jnp.asarray(v, dt) if v.dtype.kind == "f" else jnp.asarray(v))
                        for k, v in g.items() if isinstance(v, np.ndarray)} for g in groups]
        rflag = min([g["rmin"] for g in groups] + [1e9]) - 1e-3
        self.cfg = Cfg(float(chord_mm), float(sagitta_mm), float(max_path_mm), float(r_in), float(r_out),
                       float(f_metal), float(rw), float(rw_in), m_sup, int(getattr(interior, "nsub", 8)),
                       float(rflag), tuple(g["zcap"] for g in groups), tuple(g["smax"] for g in groups),
                       tuple(g["rmin"] for g in groups), tuple(g["rmax"] for g in groups),
                       bool(eloss_on), bool(msc), float(bmag * fs.ehelix.K), float(bmag * fs.ehelix.K * geom.radius))

    @classmethod
    def from_gdml(cls, gdml, field_header, origin, **kw):
        """Tracker of a GDML (straws and the support selected as fieldsteps.interior) and one G4BL field map (the DS
        map), with the nominal field at `origin`."""
        from mu2efastsim import ehelix
        from trkfastsim import geometry
        geom = geometry.from_gdml(str(gdml))
        org = np.asarray(origin, float)
        env = fs.Envelope(geom, axis_xy=org[:2])
        support = fs.interior(str(gdml), geom, env, origin=org)
        grid = ehelix.read_g4bl(field_header)
        bnom = ehelix.field_at(ehelix.grid_field(*grid), org[None])[0]
        return cls(geom, grid, support, env, org, bnom, **kw)


# --------------------------------------------------------------------------- device pieces

def _field(T, fn, x):
    """ehelix.grid_field on the device: trilinear on the regular grid, zero outside (detector frame)."""
    n = jnp.asarray(fn)
    uu = (x - T["flo"]) / T["fd"]
    i = jnp.floor(uu).astype(jnp.int32)
    inside = jnp.all((i >= 0) & (i < n - 1), axis=1)
    i = jnp.clip(i, 0, n - 2)
    f = uu - i
    B = T["fB"]
    out = jnp.zeros_like(x)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                out = out + (wx * wy * wz)[:, None] * B[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
    return jnp.where(inside[:, None], out, 0.0)


def _helix(x, u, p, q, B, s):
    """ehelix.helix."""
    bm = jnp.linalg.norm(B, axis=1)
    bh = jnp.where(bm[:, None] > 0, B / jnp.where(bm > 0, bm, 1.0)[:, None], jnp.array([0.0, 0.0, 1.0], x.dtype))
    upar = (u * bh).sum(1)
    uperp = u - upar[:, None] * bh
    w = jnp.cross(uperp, bh)
    ks = KB * q * bm / p * s
    small = jnp.abs(ks) < 1e-6
    k2 = ks * ks
    ksn = jnp.where(small, 1.0, ks)
    s1 = jnp.where(small, 1 - k2 / 6, jnp.sin(ksn) / ksn)
    c1 = jnp.where(small, ks / 2 - ks * k2 / 24, (1 - jnp.cos(ksn)) / ksn)
    xn = x + (upar * s)[:, None] * bh + (s * s1)[:, None] * uperp + (s * c1)[:, None] * w
    un = upar[:, None] * bh + jnp.cos(ks)[:, None] * uperp + jnp.sin(ks)[:, None] * w
    return xn, un / jnp.linalg.norm(un, axis=1, keepdims=True)


def _leaving(T, env, x, u):
    rv = x[:, :2] - T["env_axis"]
    r = jnp.hypot(rv[:, 0], rv[:, 1])
    ur = (rv * u[:, :2]).sum(1)
    R, z0, z1 = env
    return ((r > R) & (ur > 0)) | ((x[:, 2] < z0) & (u[:, 2] < 0)) | ((x[:, 2] > z1) & (u[:, 2] > 0))


def _straws(T, cfg, pwidth, x, c, L):
    """Gas crossings of the chords (fieldsteps.ChordIndex.crossings + _straw_pieces): per lane the K_CHORD first by
    s_in: straw index, s_in, s_out (clipped to the chord), wall, wire and plating lengths; and the overflow counts."""
    n = x.shape[0]
    z0 = jnp.minimum(x[:, 2], x[:, 2] + c[:, 2] * L)
    z1 = jnp.maximum(x[:, 2], x[:, 2] + c[:, 2] * L)
    a = jnp.searchsorted(T["plo_z"], z0 - pwidth, side="left", method="compare_all")
    b = jnp.searchsorted(T["plo_z"], z1, side="right", method="compare_all")
    npan = T["plo_z"].shape[0]
    pi = a[:, None] + jnp.arange(P_CAP)[None]
    pv = pi < b[:, None]
    pover = jnp.sum(b - a > P_CAP)
    pi = jnp.minimum(pi, npan - 1)
    lo, hi = T["plo"][pi], T["phi"][pi]                              # [n, P, 3]
    inv = 1.0 / jnp.where(c == 0, jnp.finfo(c.dtype).tiny, c)
    t1 = (lo - x[:, None]) * inv[:, None]
    t2 = (hi - x[:, None]) * inv[:, None]
    tin = jnp.maximum(jnp.minimum(t1, t2).max(-1), 0.0)
    tout = jnp.minimum(jnp.maximum(t1, t2).min(-1), L[:, None])
    pv = pv & (tout >= tin)
    js = T["pstraws"][pi]                                            # [n, P, 96]
    js = js.reshape(n, -1)
    sv = jnp.repeat(pv, T["pstraws"].shape[1], axis=1)
    mid, ax, h = T["smid"][js], T["sdir"][js], T["shalf"][js]
    w = x[:, None] - mid
    dd = c[:, None]
    da = (dd * ax).sum(-1)
    wa = (w * ax).sum(-1)
    u = dd - da[..., None] * ax
    v = w - wa[..., None] * ax
    A = (u * u).sum(-1)
    B = 2 * (u * v).sum(-1)
    vv = (v * v).sum(-1)
    C = vv - cfg.r_in ** 2
    disc = B * B - 4 * A * C
    par = A <= 1e-15
    good = jnp.where(par, C <= 0, disc > 0)
    sq = jnp.sqrt(jnp.where(good & ~par, disc, 0.0))
    A_ = jnp.where(par, 1.0, A)
    s0 = jnp.where(par, -jnp.inf, (-B - sq) / (2 * A_))
    s1 = jnp.where(par, jnp.inf, (-B + sq) / (2 * A_))
    nz = jnp.abs(da) > 1e-12
    das = jnp.where(nz, da, 1.0)
    q0, q1 = (-h - wa) / das, (h - wa) / das
    zl = jnp.where(nz, jnp.minimum(q0, q1), -jnp.inf)
    zh = jnp.where(nz, jnp.maximum(q0, q1), jnp.inf)
    inside_len = nz | (jnp.abs(wa) <= h)
    s_in, s_out = jnp.maximum(jnp.maximum(s0, zl), 0.0), jnp.minimum(s1, zh)
    keep = sv & good & inside_len & (s_out > s_in) & (s_in < L[:, None])
    nk = keep.sum(1)
    kover = jnp.sum(nk > K_CHORD)
    key = jnp.where(keep, s_in, jnp.inf)
    neg, top = lax.top_k(-key, K_CHORD)                              # the K_CHORD smallest s_in, sorted
    kv = jnp.isfinite(neg)
    g = lambda arr: jnp.take_along_axis(arr, top, axis=1)
    js, s_in, s_out = g(js), g(s_in), jnp.minimum(g(s_out), L[:, None])
    A, B, vv, zl, zh, sg = g(A), g(B), g(vv), g(zl), g(zh), g(jnp.ones_like(A))
    uv = 0.5 * B
    # wall: outer cylinder minus gas, both clipped to the chord and the straw length (_straw_pieces)
    A = jnp.maximum(A, 1e-30)
    Co = vv - cfg.r_out ** 2
    sqo = jnp.sqrt(jnp.clip(B * B - 4 * A * Co, 0.0, None))
    o0 = jnp.clip((-B - sqo) / (2 * A), 0.0, L[:, None])
    o1 = jnp.clip((-B + sqo) / (2 * A), 0.0, L[:, None])
    o0, o1 = jnp.maximum(o0, zl), jnp.minimum(o1, zh)
    gas = jnp.where(kv, s_out - s_in, 0.0)
    wall = jnp.where(kv, jnp.clip(jnp.clip(o1 - o0, 0.0, None) - gas, 0.0, None), 0.0)
    sb = jnp.clip(-uv / A, -1e30, 1e30)
    bb = jnp.sqrt(jnp.clip(vv - uv ** 2 / A, 0.0, None))
    ua = jnp.sqrt(A)
    own = kv & (sb >= 0) & (sb < L[:, None]) & (gas > 0)
    wire = jnp.where(own, 2 * jnp.sqrt(jnp.clip(cfg.rw_in ** 2 - bb ** 2, 0, None)) / ua, 0.0)
    plate = jnp.where(own, 2 * jnp.sqrt(jnp.clip(cfg.rw ** 2 - bb ** 2, 0, None)) / ua, 0.0) - wire
    del sg
    return dict(js=js, s_in=jnp.where(kv, s_in, 0.0), s_out=jnp.where(kv, s_out, 0.0), v=kv, gas=gas, wall=wall,
                wire=wire, plate=plate), pover, kover


def _support(T, cfg, groups, x, c, L):
    """Coaxial-shell column of the chords (world.CoaxialShells.column): [n, m_sup] path length per support material.
    Exact z clipping, phi sectors by nsub-point sampling; and the count of z-interval windows over the cap."""
    n = x.shape[0]
    dt = x.dtype
    w = x - T["sc"]
    ox, oy, oz = w @ T["se1"], w @ T["se2"], w @ T["sax"]
    dx, dy, dz = c @ T["se1"], c @ T["se2"], c @ T["sax"]
    A = dx * dx + dy * dy
    Bq = 2 * (ox * dx + oy * dy)
    r2 = ox * ox + oy * oy
    kk = ((jnp.arange(cfg.nsub) + 0.5) / cfg.nsub).astype(dt)
    out = jnp.zeros((n, cfg.m_sup), dt)
    over = jnp.int32(0)
    par = A < 1e-15
    a_ = jnp.where(par, 1.0, A)
    nzz = jnp.abs(dz) > 1e-15
    dzs = jnp.where(nzz, dz, 1.0)

    def chord(r):
        Cq = r2 - r * r
        disc = Bq * Bq - 4 * A * Cq
        ok = jnp.where(par, Cq <= 0, disc > 0)
        sq = jnp.sqrt(jnp.where(ok & ~par, disc, 0.0))
        return (jnp.where(par, -jnp.inf, (-Bq - sq) / (2 * a_)), jnp.where(par, jnp.inf, (-Bq + sq) / (2 * a_)), ok)

    for gi, g in enumerate(groups):
        zcap, rmin, rmax = cfg.zcap[gi], cfg.grmin[gi], cfg.grmax[gi]
        a0, a1, aok = chord(rmax)
        lo, hi = jnp.maximum(a0, 0.0), jnp.minimum(a1, L)
        cand = aok & (hi > lo)
        if rmin > 0:
            b0, b1, bok = chord(rmin)
            b0c, b1c = jnp.where(bok, b0, hi), jnp.where(bok, b1, hi)
        else:
            b0c = b1c = hi
        s0 = jnp.stack([lo, jnp.maximum(lo, jnp.minimum(b1c, hi))], -1)      # [n, 2]
        s1 = jnp.stack([jnp.minimum(hi, jnp.maximum(b0c, lo)), hi], -1)
        live = cand[:, None] & (s1 > s0)
        za, zb = oz[:, None] + dz[:, None] * s0, oz[:, None] + dz[:, None] * s1
        zlo, zhi = jnp.minimum(za, zb), jnp.maximum(za, zb)
        j_hi = jnp.searchsorted(g["Z0"], zhi, side="left", method="compare_all") - 1
        j_lo = jnp.searchsorted(g["ZRUN"], zlo, side="right", method="compare_all")
        W = jnp.where(live, jnp.clip(j_hi - j_lo + 1, 0, None), 0)
        over = over + jnp.sum(W > zcap).astype(jnp.int32)
        j = j_lo[..., None] + jnp.arange(zcap)                                  # [n, 2, zcap]
        jv = jnp.arange(zcap) < W[..., None]
        j = jnp.clip(j, 0, g["Z0"].shape[0] - 1)
        z0, z1 = g["Z0"][j], g["Z1"][j]
        ov = jv & (z1 > zlo[..., None])
        t0 = jnp.where(nzz[:, None, None], (z0 - oz[:, None, None]) / dzs[:, None, None], -jnp.inf)
        t1 = jnp.where(nzz[:, None, None], (z1 - oz[:, None, None]) / dzs[:, None, None], jnp.inf)
        u0 = jnp.maximum(s0[..., None], jnp.minimum(t0, t1))
        u1 = jnp.minimum(s1[..., None], jnp.maximum(t0, t1))
        ln = jnp.where(ov, jnp.clip(u1 - u0, 0, None), 0.0)                    # [n, 2, zcap]
        sm = u0[..., None] + ln[..., None] * kk                                 # [n, 2, zcap, nsub]
        ph = jnp.arctan2(oy[:, None, None, None] + dy[:, None, None, None] * sm,
                         ox[:, None, None, None] + dx[:, None, None, None] * sm)
        p0, dp, full, mat = g["P0"][j], g["DP"][j], g["FULL"][j], g["MAT"][j]   # [n, 2, zcap, smax]
        ins = jnp.mod(ph[..., None, :] - p0[..., None], 2 * np.pi) <= dp[..., None]   # [n, 2, zcap, smax, nsub]
        sector = ~full & (ln[..., None] > 0)
        frac = jnp.where(sector, jnp.mean(ins, -1), 1.0)
        val = jnp.where(dp >= 0, ln[..., None] * frac, 0.0)                     # padded shells: dp = -1
        rows = jnp.broadcast_to(jnp.arange(n)[:, None, None, None], val.shape)
        out = out.at[rows.ravel(), mat.ravel()].add(val.ravel())
    return out, over


def _interp(tab, lnt0, dlnt, T, col):
    """Linear interpolation in ln T of per-material tables tab [M, NT] at T [m], material col [m]."""
    f = jnp.clip((jnp.log(jnp.maximum(T, 1e-30)) - lnt0) / dlnt, 0.0, SJ.NT - 1.000001)
    i = f.astype(jnp.int32)
    w = f - i
    return tab[col, i] * (1 - w) + tab[col, i + 1] * w


def _electron_loss(key, Tb, mc, lnt0, dlnt, p, ln, col, nd=ND, nb=NB):
    """fieldsteps.electron_loss for (track, material) pairs: p, ln [m] (ln 0: no loss), col [m] material index.
    Continuous (Urban), deltas and brem photons above E_CUT. Returns the loss [m] and the counts of delta and photon
    caps reached."""
    dt = p.dtype
    cut = fs.E_CUT_MEV
    T = jnp.sqrt(p * p + ME * ME) - ME
    on = ln > 0
    lnz = jnp.where(on, ln, 0.0)
    k = jax.random.split(key, 6)
    mean = _interp(Tb["dedx"], lnt0, dlnt, T, col) * lnz
    mcp = eloss_jax.MatConsts(*(f[col] for f in mc))
    lost = eloss_jax.sample_loss_electron(k[0], mcp, T, lnz, cut, mean, unroll=UNROLL)
    lost = jnp.where(on, lost, 0.0)
    # deltas: Poisson count of transfers above E_CUT at the start energy
    dok = on & (T > 2 * cut)
    lam = jnp.where(dok, _interp(Tb["sig_d"], lnt0, dlnt, T, col) * lnz, 0.0)
    u = jax.random.uniform(k[1], ln.shape, dt)
    pmf = jnp.exp(-lam)
    cdf = pmf
    cnt = jnp.zeros(ln.shape, jnp.int32)
    for j in range(nd):
        above = u > cdf
        cnt = cnt + above
        pmf = pmf * lam / (j + 1)
        cdf = cdf + pmf
    dcap = jnp.sum(dok & (u > cdf))
    Td = jnp.where(dok, T, 4 * cut + 1.0)
    de = SJ._sample_delta(k[2], jnp.broadcast_to(Td, (nd,) + Td.shape), cut, False, dt,
                          SJ.REJECT_ROUNDS if UNROLL else False)                  # all nd at once
    lost = lost + jnp.sum(jnp.where(cnt[None] > jnp.arange(nd)[:, None], de, 0.0), 0)
    # brem photons along the length, the energy reduced after each one (ebrem.sample_along)
    bx = Tb["brem_x"]

    def brem(j, c):
        Tc, s, live, lost = c
        kj = jax.random.split(jax.random.fold_in(k[3], j), 4)
        sig = _interp(Tb["sig_b"], lnt0, dlnt, Tc, col)
        e = jax.random.exponential(kj[0], ln.shape, dt)
        s = s + jnp.where(sig > 0, e / jnp.where(sig > 0, sig, 1.0), jnp.inf)
        hit = live & (s < lnz)
        Tp = jnp.where(hit, Tc, 1.0)
        f = jnp.clip((jnp.log(Tp) - lnt0) / dlnt, 0.0, SJ.NT - 1.000001)
        j0 = f.astype(jnp.int32)
        jj = j0 + (jax.random.uniform(kj[1], ln.shape, dt) < f - j0)
        uc = jax.random.uniform(kj[2], ln.shape, dt)
        cc = (1.0 - (1.0 - uc) ** (1.0 / SJ.C_POWER)) * (SJ.NX - 1)
        ci = jnp.minimum(cc.astype(jnp.int32), SJ.NX - 2)
        xb = bx[col, jj, ci] * (1 - (cc - ci)) + bx[col, jj, ci + 1] * (cc - ci)
        kph = jnp.where(hit, cut * jnp.exp(xb * jnp.log(jnp.maximum(Tp, cut) / cut)), 0.0)
        kph = jnp.minimum(kph, Tp)
        return Tc - kph, s, hit & (Tc - kph > cut), lost + kph

    c0 = (T, jnp.zeros_like(T), on & (T > cut), lost)
    if nb <= 4:
        c = c0
        for j in range(nb):
            c = brem(j, c)
    else:
        c = lax.fori_loop(0, nb, brem, c0, unroll=8 if UNROLL else 1)
    Tc, s, live, lost = c
    bcap = jnp.sum(live)
    return jnp.minimum(lost, T), dcap, bcap


def _highland_var(p, x_prev, x_new):
    """transport.highland_var_increment for electrons."""
    p = jnp.maximum(p, 1e-3)
    beta2 = p * p / (p * p + ME * ME)

    def gg(x):
        x = jnp.maximum(x, 0.0)
        return jnp.where(x > 0, x * (1 + 0.038 * jnp.log(jnp.maximum(x, 1e-30) / beta2)) ** 2, 0.0)

    return (13.6 ** 2 / (beta2 * p * p)) * jnp.maximum(gg(x_new) - gg(x_prev), 0.0)


def _kink_dir(d, dth):
    """steps._kink without the displacement."""
    ref = jnp.where(jnp.abs(d[:, [0]]) < 0.9, jnp.array([[1.0, 0, 0]], d.dtype), jnp.array([[0, 0, 1.0]], d.dtype))
    u = jnp.cross(d, ref)
    u = u / jnp.linalg.norm(u, axis=1)[:, None]
    v = jnp.cross(d, u)
    dn = d + u * jnp.tan(dth[:, [0]]) + v * jnp.tan(dth[:, [1]])
    return dn / jnp.linalg.norm(dn, axis=1)[:, None]


# --------------------------------------------------------------------------- the chord loop

def _state(x, u, p, t, active):
    n = x.shape[0]
    dt = x.dtype
    tk = jnp.sqrt(p * p + ME * ME) - ME
    return dict(x=x, u=u, p=p, tk=tk, t=t, path=jnp.zeros(n, dt), xcum=jnp.zeros(n, dt),
                chords=jnp.zeros(n, jnp.int32), status=jnp.where(active, LIVE, LEFT).astype(jnp.int32),
                nrec=jnp.zeros(n, jnp.int32), last_js=jnp.full(n, -1, jnp.int32), last_g=jnp.full(n, -1e30, dt))


def _records(n, dt):
    z = lambda *s: jnp.zeros((n, R_CAP) + s, dt)
    return dict(js=jnp.full((n, R_CAP), -1, jnp.int32), pos0=z(3), pos1=z(3), d0=z(3), d1=z(3), p0=z(), p1=z(),
                len=z(), t=z())


def _chord(tk_, s, rec, lane, key, it):
    """One chord for every live lane of the state s (lanes `lane` index the record buffers rec)."""
    tr_, cfg = tk_
    T, groups, env, fn, pwidth, lnt0, dlnt, mc = tr_
    n = s["x"].shape[0]
    dt = s["x"].dtype
    q = -1.0
    act = s["status"] == LIVE
    x, u, p = s["x"], s["u"], s["p"]
    key = jax.random.fold_in(key, it)
    ks = jax.random.split(key, 4)
    B0 = _field(T, fn, x)
    bm0 = jnp.linalg.norm(B0, axis=1)
    rho = jnp.where(bm0 > 0, p / jnp.maximum(KB * bm0, 1e-30), jnp.inf)
    h = jnp.minimum(jnp.minimum(cfg.chord_mm, jnp.sqrt(8.0 * rho * cfg.sagitta_mm)), cfg.max_path_mm - s["path"])
    h = jnp.maximum(h, 1e-3)
    xm, _ = _helix(x, u, p, q, B0, 0.5 * h)
    B = _field(T, fn, xm)
    xn, un = _helix(x, u, p, q, B, h)
    cv = xn - x
    L = jnp.maximum(jnp.linalg.norm(cv, axis=1), 1e-12)
    c = cv / L[:, None]
    arc = h / L
    gb = p / ME
    beta = gb / jnp.sqrt(1 + gb * gb)
    # support only where the chord reaches the smallest shell radius (the radius is largest at a chord end)
    if groups:
        rv0 = x[:, :2] - T["sc"][:2]
        rv1 = xn[:, :2] - T["sc"][:2]
        rmax = jnp.maximum(jnp.hypot(rv0[:, 0], rv0[:, 1]), jnp.hypot(rv1[:, 0], rv1[:, 1]))
        need = act & (rmax >= cfg.rflag)
        cs = max(1, int(np.ceil(SUP_FRAC * n)))
        idx = jnp.nonzero(need, size=cs, fill_value=n)[0]
        got = jnp.zeros(n + 1, bool).at[idx].set(True)[:n]
        go = act & (~need | got)
        stall = jnp.sum(act & ~go)
        sl, zover = _support(T, cfg, groups, jnp.concatenate([x, x[:1]])[idx], jnp.concatenate([c, c[:1]])[idx],
                             jnp.concatenate([L, L[:1]])[idx])
        sup = jnp.zeros((n + 1, cfg.m_sup), dt).at[idx].set(sl)[:n]
    else:
        go, stall, zover = act, jnp.int32(0), jnp.int32(0)
        sup = jnp.zeros((n, 0), dt)
    sp, pover, kover = _straws(T, cfg, pwidth, x, c, L)
    v = sp["v"] & go[:, None]
    fm = cfg.f_metal
    wall = jnp.sum(jnp.where(v, sp["wall"], 0.0), 1)
    thin = jnp.stack([wall * (1 - fm), wall * fm, jnp.sum(jnp.where(v, sp["gas"], 0.0), 1),
                      jnp.sum(jnp.where(v, sp["wire"], 0.0), 1), jnp.sum(jnp.where(v, sp["plate"], 0.0), 1)], 1)
    ln_all = jnp.concatenate([sup, thin], 1) * go[:, None]                 # [n, M]: support, then the thin parts
    M = ln_all.shape[1]
    if cfg.eloss:
        # (track, material) pairs with material, compacted: thin straw materials, then the support (larger caps)
        lost = jnp.zeros(n, dt)
        dcap = bcap = jnp.int64(0)
        ms = cfg.m_sup
        for part, cap, off, nd, nb, kk in ((ln_all[:, ms:], PAIRS * n, ms, ND, NB, 0),
                                            (ln_all[:, :ms], SUP_PAIRS * max(1, int(np.ceil(SUP_FRAC * n))), 0,
                                             ND_SUP, NB_SUP, 1)):
            Mp = part.shape[1]
            if Mp == 0:
                continue
            flat = part.ravel()
            nz = jnp.sum(flat > 0)
            pidx = jnp.nonzero(flat > 0, size=cap, fill_value=n * Mp)[0]
            plane_ = pidx // Mp
            pcol = jnp.where(pidx < n * Mp, pidx % Mp, 0) + off
            pl = jnp.concatenate([flat, jnp.zeros(1, dt)])[pidx]
            pp = jnp.maximum(jnp.concatenate([p, p[:1]])[jnp.minimum(plane_, n)], 1.0)
            lp, dc, bc = _electron_loss(jax.random.fold_in(ks[0], kk), T, mc, lnt0, dlnt, pp, pl, pcol, nd, nb)
            lost = lost.at[plane_].add(lp, mode="drop")
            dcap = dcap + dc + (nz - jnp.minimum(nz, cap))
            bcap = bcap + bc
    else:
        lost, dcap, bcap = jnp.zeros(n, dt), jnp.int32(0), jnp.int32(0)
    tk_new = s["tk"] - lost
    dead = tk_new <= 0.5
    pn = jnp.sqrt(jnp.clip((jnp.maximum(tk_new, 0.0) + ME) ** 2 - ME * ME, 1e-6, None))
    if cfg.msc:
        # Highland core kink at the chord end (the displacement is dropped, as fieldsteps: only the angle is needed)
        X = (ln_all * T["invx0"][None, :M]).sum(1)
        kk = (X > 0) & ~dead & go
        pm = 0.5 * (p + pn)
        V = _highland_var(jnp.maximum(pm, 1.0), s["xcum"], s["xcum"] + jnp.where(kk, X, 0.0))
        z = jax.random.normal(ks[1], (n, 2), dt)
        dth = z * jnp.sqrt(V)[:, None]
        th = jnp.hypot(dth[:, 0], dth[:, 1])
        dth = jnp.where((th > 0.5)[:, None], dth * (0.5 / jnp.maximum(th, 1e-30))[:, None], dth)
        un = jnp.where(kk[:, None], _kink_dir(un, dth), un)
        xcum = s["xcum"] + X
    else:
        xcum = s["xcum"]
    # records: gas passages, the first crossing of a chord continuing the track's last record
    g_in = s["path"][:, None] + sp["s_in"] * arc[:, None]
    g_out = s["path"][:, None] + sp["s_out"] * arc[:, None]
    tin = s["t"][:, None] + sp["s_in"] * arc[:, None] / (beta[:, None] * CMM)
    pos0 = x[:, None] + c[:, None] * sp["s_in"][..., None]
    pos1 = x[:, None] + c[:, None] * sp["s_out"][..., None]
    js = sp["js"]
    tol = fs.MERGE_TOL_MM if dt == jnp.float64 else MERGE_TOL_F32
    cont = v[:, 0] & (js[:, 0] == s["last_js"]) & (jnp.abs(g_in[:, 0] - s["last_g"]) < tol) & (s["nrec"] > 0)
    new = v & (~cont[:, None] | (jnp.arange(K_CHORD)[None] > 0))
    slot = s["nrec"][:, None] + jnp.cumsum(new, 1) - 1
    rover = jnp.sum(new & (slot >= R_CAP))
    slot = slot.astype(jnp.int32)
    R = R_CAP
    row = jnp.broadcast_to(lane[:, None], slot.shape)
    wslot = jnp.where(new & (slot < R), slot, R)                      # R: dropped (out of bounds)
    rec = dict(rec)
    rec["js"] = rec["js"].at[row, wslot].set(js, mode="drop")
    for f, val in (("pos0", pos0), ("pos1", pos1), ("d0", jnp.broadcast_to(c[:, None], pos0.shape)),
                   ("d1", jnp.broadcast_to(c[:, None], pos0.shape)), ("p0", jnp.broadcast_to(p[:, None], slot.shape)),
                   ("p1", jnp.broadcast_to(p[:, None], slot.shape)), ("len", sp["gas"]), ("t", tin)):
        rec[f] = rec[f].at[row, wslot].set(val, mode="drop")
    last = jnp.where(cont & (s["nrec"] - 1 < R), s["nrec"] - 1, R)
    rec["len"] = rec["len"].at[lane, last].add(sp["gas"][:, 0], mode="drop")
    rec["pos1"] = rec["pos1"].at[lane, last].set(pos1[:, 0], mode="drop")
    rec["d1"] = rec["d1"].at[lane, last].set(c, mode="drop")
    rec["p1"] = rec["p1"].at[lane, last].set(p, mode="drop")
    nv = v.sum(1)
    lastk = jnp.maximum(nv - 1, 0)
    has = nv > 0
    last_js = jnp.where(has, jnp.take_along_axis(js, lastk[:, None], 1)[:, 0], s["last_js"])
    last_g = jnp.where(has, jnp.take_along_axis(g_out, lastk[:, None], 1)[:, 0], s["last_g"])
    nrec = (s["nrec"] + new.sum(1)).astype(jnp.int32)
    # state update (lanes that wait keep their state)
    pmean = 0.5 * (p + pn)
    bmn = pmean / ME
    t = s["t"] + h / (bmn / jnp.sqrt(1 + bmn * bmn) * CMM)
    left = _leaving(T, env, xn, un)
    path = s["path"] + h
    capped = path >= cfg.max_path_mm - 1e-9
    status = jnp.where(dead, STOPPED, jnp.where(left, LEFT, jnp.where(capped, CAPPED, LIVE)))
    sel = lambda a, b: jnp.where(go.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
    ns = dict(x=sel(xn, x), u=sel(un, u), p=sel(pn, p), tk=sel(tk_new, s["tk"]), t=sel(t, s["t"]),
              path=sel(path, s["path"]), xcum=sel(xcum, s["xcum"]), chords=(s["chords"] + go).astype(jnp.int32),
              status=sel(status, s["status"]), nrec=sel(nrec, s["nrec"]), last_js=sel(last_js, s["last_js"]),
              last_g=sel(last_g, s["last_g"]))
    stats = jnp.stack([pover, kover, rover, zover, dcap, bcap, stall]).astype(jnp.int64)
    return ns, rec, stats


NSTAT = ("panel_overflow", "chord_overflow", "record_overflow", "shell_overflow", "delta_cap", "brem_cap", "waits")
# delta_cap also counts (track, material) pairs beyond the PAIRS / SUP_PAIRS caps


@partial(jax.jit, static_argnames=("cfg", "fn", "env", "pwidth", "lnt0", "dlnt", "n_it"), donate_argnames=("s", "rec"))
def _run(T, groups, mc, s, rec, lane, key, it0, cfg, fn, env, pwidth, lnt0, dlnt, n_it):
    tk_ = ((T, groups, env, fn, pwidth, lnt0, dlnt, mc), cfg)

    def cond(c):
        i, s_, _, _ = c
        return (i < n_it) & jnp.any(s_["status"] == LIVE)

    def body(c):
        i, s_, rec_, stt = c
        s2, rec2, st2 = _chord(tk_, s_, rec_, lane, key, it0 + i)
        return i + 1, s2, rec2, stt + st2

    i, s, rec, stats = lax.while_loop(cond, body, (jnp.int32(0), s, rec, jnp.zeros(len(NSTAT), jnp.int64)))
    return i, s, rec, stats


@partial(jax.jit, static_argnames=("cfg", "lnt0", "dlnt"))
def _rows(T, gas_mc, rec, nrec, key, cfg, lnt0, dlnt, tcut):
    """Per record: the fieldsteps._merge_rows columns (device). Invalid slots have len 0 and valid False."""
    n, R = rec["len"].shape
    dt = rec["len"].dtype
    valid = (jnp.arange(R)[None] < nrec[:, None]) & (rec["len"] > 0)
    js = jnp.where(valid, rec["js"], 0)
    p0, p1, d0 = rec["p0"], rec["p1"], rec["d0"]
    pvec = 0.5 * (p0[..., None] * d0 + p1[..., None] * rec["d1"])
    pmag = jnp.linalg.norm(pvec, axis=-1)
    ln = jnp.where(valid, rec["len"], 0.0)
    T0 = jnp.sqrt(p0 * p0 + ME * ME) - ME
    mean = jnp.interp(jnp.log(jnp.maximum(T0, 1e-30)), lnt0 + dlnt * jnp.arange(SJ.NT, dtype=dt), T["gas_dedx"]) * ln
    eion = eloss_jax.sample_loss_electron(key, gas_mc, jnp.where(valid, T0, 1.0), jnp.where(valid, ln, 1.0), tcut,
                                          jnp.where(valid, mean, 0.0))
    eion = jnp.where(valid, eion, 0.0)
    bg = p0 / ME
    ke = T0
    shape = jnp.where(p0 < fs.CURL_RATIO * cfg.pstraw, fs.SHAPE_CURL,
                      jnp.where(p0 < fs.LINE_RATIO * cfg.pstraw, fs.SHAPE_ARC, fs.SHAPE_LINE))
    bdir = T["bdir"]
    if cfg.bnomc > 0:
        pperp = jnp.linalg.norm(jnp.cross(d0, bdir), axis=-1)
        bendrms = 0.5 * jnp.minimum(cfg.r_in, pmag * pperp / cfg.bnomc)
        sint = jnp.linalg.norm(jnp.cross(jnp.cross(jnp.broadcast_to(bdir, d0.shape), d0), T["sdir"][js]), axis=-1)
        sagrms = fs.SAG_PRMS * sint * ln * ln * cfg.bnomc * pperp / jnp.maximum(pmag, 1e-30)
        width = jnp.minimum(sagrms, bendrms)
    else:
        width = jnp.zeros_like(ln)
    ion = jnp.where((bg > st.MINION_BG) & (ke > st.MINION_KE), 0, 1)
    return dict(valid=valid, straw_index=js, sid=T["sid"][js], plane=T["plane"][js], panel=T["panel"][js],
                straw=T["straw"][js], upanel=T["upanel"][js], shape=shape.astype(jnp.int32),
                ion=ion.astype(jnp.int32), eion=eion, len=ln, width=width, t=rec["t"],
                x0=rec["pos0"][..., 0], y0=rec["pos0"][..., 1], z0=rec["pos0"][..., 2],
                x1=rec["pos1"][..., 0], y1=rec["pos1"][..., 1], z1=rec["pos1"][..., 2],
                px=pvec[..., 0], py=pvec[..., 1], pz=pvec[..., 2])


class Result(NamedTuple):
    steps: dict      # per track and record slot [n, R_CAP] device arrays (fieldsteps sgsteps columns, `valid`)
    final: dict      # per track: x, y, z (Mu2e frame), px, py, pz, t, path, chords, status (0 left, 1 stopped, 2 cap)
    stats: dict      # cap counters (must be 0 except waits) and loop counts


def transport(trk, key, x, p, t, active=None, rounds_it=256, min_bucket=64, shrink=2, stats=None):
    """Field-on StrawGasSteps of electrons on the device. x [n, 3] (Mu2e frame), p [n, 3] momentum [MeV/c], t [n] [ns]:
    device or host arrays; active [n] bool (default all): tracks to transport (the others get no steps, status 0).
    Between launches of `rounds_it` chords the live lanes are compacted into power-of-two buckets (host reads one
    lane count; the data stay on the device). Returns Result with device arrays."""
    dt = trk.dt
    x = jnp.asarray(x, dt) - jnp.asarray(trk.origin, dt)
    pv = jnp.asarray(p, dt)
    pm = jnp.linalg.norm(pv, axis=1)
    u = pv / pm[:, None]
    t = jnp.asarray(t, dt)
    n = x.shape[0]
    active = jnp.ones(n, bool) if active is None else jnp.asarray(active, bool)
    s = _state(x, u, pm, t, active)
    # the envelope test at the start (tracks already leaving get no chord)
    s["status"] = jnp.where(active & _leaving(trk.T, trk.env, x, u), LEFT, s["status"])
    rec = _records(n, dt)
    cfg = trk.cfg
    kw = dict(cfg=cfg, fn=trk.fn, env=trk.env, pwidth=trk.pwidth, lnt0=trk.lnt0, dlnt=trk.dlnt)
    final = {k: jnp.array(s[k], copy=True) for k in ("x", "u", "p", "t", "path", "chords", "status")}
    lane = jnp.arange(n, dtype=jnp.int32)
    tot = np.zeros(len(NSTAT), np.int64)
    it = 0
    launches = 0
    nlive = int(jnp.sum(s["status"] == LIVE))
    bucket = n
    timeline = []
    while nlive:
        tl0 = time.perf_counter()
        i, s, rec, st_ = _run(trk.T, trk.groups, trk.mc, s, rec, lane, key, jnp.int32(it), n_it=int(rounds_it), **kw)
        launches += 1
        it += int(rounds_it)
        tot += np.asarray(st_)
        live = s["status"] == LIVE
        nlive = int(jnp.sum(live))
        timeline.append((bucket, round(time.perf_counter() - tl0, 3)))
        wl = jnp.where(~live & (lane < n), lane, n)                    # finished real lanes (n: dropped)
        for k in final:
            final[k] = final[k].at[wl].set(s[k], mode="drop")
        if nlive == 0:
            break
        mb = max(min_bucket, 1 << int(np.ceil(np.log2(nlive))))
        if mb * shrink <= bucket:
            idx = jnp.nonzero(live, size=mb, fill_value=0)[0]
            pad = jnp.arange(mb) >= nlive
            s = {k: v[idx] for k, v in s.items()}
            s["status"] = jnp.where(pad, LEFT, s["status"])           # padding lanes: copies, inert
            lane = jnp.where(pad, n, lane[idx])                       # n: their record writes are dropped
            bucket = mb
    # padding lanes write records to row n (out of range: dropped by mode="drop"); finals were written for real lanes
    steps = _rows(trk.T, trk.gas_mc, rec, s_nrec_of(rec), jax.random.fold_in(key, 1 << 30), cfg, trk.lnt0, trk.dlnt,
                  trk.tcut_gas)
    org = jnp.asarray(trk.origin, dt)
    fin = dict(x=final["x"][:, 0] + org[0], y=final["x"][:, 1] + org[1], z=final["x"][:, 2] + org[2],
               px=final["u"][:, 0] * final["p"], py=final["u"][:, 1] * final["p"], pz=final["u"][:, 2] * final["p"],
               t=final["t"], path=final["path"], chords=final["chords"], status=final["status"])
    out_stats = dict(zip(NSTAT, (int(v) for v in tot)), iterations=it, launches=launches, timeline=timeline)
    if stats is not None:
        stats.update(out_stats)
    return Result(steps, fin, out_stats)


def s_nrec_of(rec):
    """Records per track: the filled slots (js >= 0)."""
    return jnp.sum(rec["js"] >= 0, axis=1).astype(jnp.int32)


# --------------------------------------------------------------------------- host views (checks, writers)

STEP_COLS = ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len", "width", "t",
             "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]


def to_frame(res, keys):
    """Host DataFrame of the steps in the fieldsteps sgsteps layout (KEY columns from `keys`, row i = track i), rows
    ordered by track and passage (as fieldsteps._merge_rows)."""
    import pandas as pd
    from mu2efastsim.tables import KEY
    s = {k: np.asarray(v) for k, v in res.steps.items()}
    tr, sl = np.nonzero(s["valid"])
    out = pd.DataFrame({k: keys[k].to_numpy()[tr] for k in KEY})
    out["nsgs"] = np.bincount(tr, minlength=len(keys))[tr]
    for k in ("sid", "plane", "panel", "straw", "upanel", "shape", "ion"):
        out[k] = s[k][tr, sl]
    out["sp"] = keys["sp"].to_numpy()[tr] if "sp" in keys else 1
    for k in ("eion", "len", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"):
        out[k] = s[k][tr, sl]
    return out[KEY + STEP_COLS].reset_index(drop=True)


def final_frame(res, keys):
    import pandas as pd
    from mu2efastsim.tables import KEY
    f = {k: np.asarray(v) for k, v in res.final.items()}
    out = pd.DataFrame({k: keys[k].to_numpy() for k in KEY})
    out["sp"] = keys["sp"].to_numpy() if "sp" in keys else 1
    for k in ("x", "y", "z", "px", "py", "pz", "t", "path", "chords", "status"):
        out[k] = f[k]
    return out
