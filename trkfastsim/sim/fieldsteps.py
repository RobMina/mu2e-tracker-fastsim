"""Opt-in transport of charged primaries through the tracker in a magnetic field: StrawGasSteps with the field on (NumPy).

The default chain (`steps.make_steps`) is straight-line transport for the field-free extracted configuration; this
module is separate and changes nothing there. It starts from given states inside or at the tracker envelope (Mu2e
frame: position, momentum, time, PDG code) and follows each track as a chain of chords of its local helix
(mu2efastsim.ehelix.helix with the field at the helix midpoint of the step, as ehelix.propagate). A chord is at most
`chord_mm` long and short enough that its sagitta, L^2 / (8 rho) with rho = p / (K |q| |B|), stays below `sagitta_mm`.

On each chord (a straight segment from the step start to the helix end point):
  * straw gas crossings (crossings.straw_crossings, clipped to the chord); a crossing cut by a chord end is merged
    with its continuation on the next chord, so each row is one whole passage through a straw;
  * material: the straw walls of the crossed straws (outer-cylinder chord minus gas chord, Mylar and outer metal), the
    sense wire (W core, Au plating) at its closest approach, the gas, and the support structure (TrackerInterior
    shells, if given). The tracker volume itself is vacuum (the DS);
  * energy loss: support and wall/wire material sampled per material with the muon slice sampler
    (transport.slab_loss); the gas debits its mean unrestricted loss (with `loss` given, every material uses it), and its deposited eIon is sampled once per merged
    crossing (eloss.sample_loss with the tracker delta cut, as the default chain);
  * scattering: the chord's material (support, walls, wire, gas) as one angular kink at the chord end, Highland with
    the log term on the cumulative in-tracker x/X0 and the single-scattering tail (steps._kink_core, _group_tail);
    the lateral displacement of the kink is dropped (micrometres on chords of a few mm to tens of mm).

Rows follow MakeStrawGasSteps with the field on: entry and exit points of the passage, path = summed chord lengths
inside the gas, momentum = mean of the entry and exit momentum vectors, width = min(sagitta rms, bend rms) with the
nominal field `bnom` at the tracker centre, shape by |p| against the momentum whose bend radius is the straw radius
(curl < 1x, arc < 10x, else line), ionisation type minion if beta*gamma > 0.5 and KE > 20 MeV. Time at the gas entry
is the start time plus the arc length over beta c, beta from the momentum at each chord start. Positions are written
in the detector frame (`origin`).

Tracks stop when they leave the envelope (a padded cylinder about the tracker axis around the straws), range out,
or reach `max_path_mm` / `max_chords`. Only muons are supported with the default loss model; electrons need an
explicit `loss` callable (the muon slice sampler and tail are not electron physics).
"""
import numpy as np
import pandas as pd

from mu2efastsim import ehelix, eloss, materials
from mu2efastsim import transport as tr
from mu2efastsim.tables import KEY

from trkfastsim import crossings, geometry
from trkfastsim.sim import steps as st

C_MM_PER_NS = 299.792458
CURL_RATIO, LINE_RATIO = 1.0, 10.0        # MakeStrawGasSteps CurlRatio, LineRatio defaults
SHAPE_LINE, SHAPE_ARC, SHAPE_CURL = 0, 1, 2
MERGE_TOL_MM = 1e-6                       # continuation of a crossing on the next chord: entry within this of the exit
SAG_PRMS = 1.0 / (12.0 * np.sqrt(5.0))    # MakeStrawGasSteps parabola rms factor (includes the 1/8 of the sagitta)
CHARGE = {13: -1.0, -13: 1.0, 11: -1.0, -11: 1.0, 211: 1.0, -211: -1.0, 2212: 1.0, -2212: -1.0}
MASS = {13: eloss.MMU, 11: eloss.ME, 211: 139.57039, 2212: 938.27208816}


class Envelope:
    """Cylinder about the tracker axis around the straws, padded by pad_r / pad_z [mm]; tracks leaving it stop."""

    def __init__(self, geom, pad_r=150.0, pad_z=150.0, axis_xy=None):
        ends = np.concatenate([geom.mid - geom.dir * geom.halflen[:, None], geom.mid + geom.dir * geom.halflen[:, None]])
        self.axis = np.asarray(geometry.DETECTOR_ORIGIN_EXTRACTED[:2] if axis_xy is None else axis_xy, float)
        self.r = float(np.hypot(*(ends[:, :2] - self.axis).T).max() + geom.radius + pad_r)
        self.z0 = float(ends[:, 2].min() - geom.radius - pad_z)
        self.z1 = float(ends[:, 2].max() + geom.radius + pad_z)

    def leaving(self, x, u):
        """Outside the cylinder and moving away from it."""
        rv = x[:, :2] - self.axis
        r = np.hypot(rv[:, 0], rv[:, 1])
        ur = (rv * u[:, :2]).sum(1)
        return ((r > self.r) & (ur > 0)) | ((x[:, 2] < self.z0) & (u[:, 2] < 0)) | ((x[:, 2] > self.z1) & (u[:, 2] > 0))


def interior(gdml_path, geom, envelope, origin=geometry.DETECTOR_ORIGIN_EXTRACTED, pad=150.0):
    """Tracker support structure for the field transport: the tube/polycone pieces of the GDML near the straws
    (TrackerInterior's selection, without the calorimeter) that reach inside the envelope cylinder. Pieces wholly
    outside it (the DS coils and cryostat in a Run 1A/Run 2 world) are never crossed by a track that stops at the
    envelope, so they are left out. Returns a steps.TrackerInterior."""
    from mu2efastsim import world as world_mod
    lo, hi = crossings.panel_boxes(geom)
    tubes = world_mod.load_tubes(gdml_path, bbox=(lo.min(0) - pad, hi.max(0) + pad), skip_names=st.INTERIOR_SKIP)
    keep = []
    for tb in tubes:
        if tb[0] == "StrawGas":
            continue
        o = np.hypot(*(np.asarray(tb[2], float)[:2] - envelope.axis))
        if max(0.0, o - tb[4], tb[3] - o) < envelope.r:           # nearest point of the annulus to the axis
            keep.append(tb)
    it = st.TrackerInterior.__new__(st.TrackerInterior)
    it.shells = st.InteriorColumn(world_mod.CoaxialShells(keep, axis_point=np.asarray(origin, float)), None)
    it.nsub, it.calo_skipped = 8, {}
    it.unknown = sorted({tb[0] for tb in keep} - set(materials.BY_NAME))
    if it.unknown:
        raise KeyError(f"tracker interior materials without constants: {it.unknown}")
    return it


E_CUT_MEV = 0.1    # electron_loss: delta and photon cut [MeV]; transfers above it are sampled one by one


def electron_loss(rng, mat, p, length, mass=eloss.ME):
    """Kinetic-energy loss [MeV] of electrons over `length` [mm] of `mat` (mu2efastsim.eioni, ebrem: G4eIonisation and
    G4SeltzerBergerModel matched): continuous restricted ionisation + brem below E_CUT_MEV with Urban fluctuations,
    plus the deltas and photons above it sampled one by one (their energy leaves the track). Electrons only."""
    from mu2efastsim import ebrem, eioni
    p = np.asarray(p, float)
    T = np.sqrt(p ** 2 + eloss.ME ** 2) - eloss.ME
    length = np.asarray(length, float) * np.ones_like(T)
    lost = eioni.continuous_loss(rng, mat, T, length, E_CUT_MEV, E_CUT_MEV)
    ok = T > 2 * E_CUT_MEV
    if ok.any():
        lam = np.zeros_like(T)
        lam[ok] = eioni.xsec_per_mm(mat, T[ok], E_CUT_MEV) * length[ok]
        k = rng.poisson(lam)
        rep = np.repeat(np.arange(len(T)), k)
        if len(rep):
            np.add.at(lost, rep, eioni.sample_delta(rng, T[rep], E_CUT_MEV))
    idx, _, kph, _, _ = ebrem.sample_along(rng, mat, np.maximum(T, 0.0), length, E_CUT_MEV)
    if len(idx):
        np.add.at(lost, idx, kph)
    return np.minimum(lost, T)


def _straw_pieces(x, c, L, geom, r_out, rw, rw_in, index=None):
    """Gas crossings of chords (x, c, L): rows (lane, straw, s_in, s_out, wall, wire, plate, s_wire) with the wall chord
    (outer cylinder minus gas, both clipped to the chord and the straw length) and the wire/plating chords at the
    closest approach to the axis (assigned to the chord holding that point)."""
    xs = crossings.straw_crossings(x, c, geom, index=index)
    if not len(xs):
        return None
    lane = xs[:, 0].astype(np.int64)
    js = xs[:, 1].astype(np.int64)
    keep = xs[:, 2] < L[lane]
    xs, lane, js = xs[keep], lane[keep], js[keep]
    if not len(xs):
        return None
    s_in, s_out = xs[:, 2], np.minimum(xs[:, 3], L[lane])
    a = geom.dir[js]
    w = x[lane] - geom.mid[js]
    d = c[lane]
    da = (d * a).sum(1)
    wa = (w * a).sum(1)
    u = d - da[:, None] * a
    v = w - wa[:, None] * a
    A = np.maximum((u * u).sum(1), 1e-30)
    B = 2 * (u * v).sum(1)
    C = (v * v).sum(1) - r_out ** 2
    sq = np.sqrt(np.clip(B * B - 4 * A * C, 0.0, None))
    o0 = np.clip((-B - sq) / (2 * A), 0.0, L[lane])
    o1 = np.clip((-B + sq) / (2 * A), 0.0, L[lane])
    h = geom.halflen[js]
    with np.errstate(divide="ignore", invalid="ignore"):
        z0, z1 = (-h - wa) / da, (h - wa) / da
    zl = np.where(np.abs(da) > 1e-12, np.minimum(z0, z1), -np.inf)
    zh = np.where(np.abs(da) > 1e-12, np.maximum(z0, z1), np.inf)
    o0, o1 = np.maximum(o0, zl), np.minimum(o1, zh)
    gas = s_out - s_in
    wall = np.clip(np.clip(o1 - o0, 0.0, None) - gas, 0.0, None)
    sb = np.clip(-(u * v).sum(1) / A, -1e30, 1e30)                 # closest approach to the axis along the chord
    b = np.sqrt(np.clip((v * v).sum(1) - (u * v).sum(1) ** 2 / A, 0.0, None))
    ua = np.sqrt(A)
    own = (sb >= 0) & (sb < L[lane]) & (gas > 0)
    wire = np.where(own, 2 * np.sqrt(np.clip(rw_in ** 2 - b ** 2, 0, None)) / ua, 0.0)
    plate = np.where(own, 2 * np.sqrt(np.clip(rw ** 2 - b ** 2, 0, None)) / ua, 0.0) - wire
    return lane, js, s_in, s_out, wall, wire, plate


def _columns(n, parts):
    """Sum per lane of (material, lane index, length) triples: {material name: (material, lengths[n])}."""
    out = {}
    for m, ix, ln in parts:
        if not len(ix):
            continue
        v = out.setdefault(m.name, (m, np.zeros(n)))[1]
        np.add.at(v, ix, ln)
    return out


def transport(rng, start, geom, field, interior=None, *, chord_mm=20.0, sagitta_mm=0.02, envelope=None,
              origin=geometry.DETECTOR_ORIGIN_EXTRACTED, bnom=None, max_path_mm=30000.0, max_chords=20000,
              wire=True, tail=True, tail_k=None, msc=True, loss=None):
    """Field-on StrawGasSteps of the tracks in `start` (DataFrame: KEY columns, sp, pdg, x, y, z [mm, Mu2e frame],
    px, py, pz [MeV/c], t [ns]). field: constant 3-vector [T], a callable B(x) (ehelix.grid_field) or None (no field).
    bnom: nominal field vector at the tracker centre for the step width (default: field at the envelope centre).
    loss: None (muon slice sampler; muons only) or callable(rng, material, p, length, mass) -> MeV.
    Returns (steps DataFrame in the sgsteps layout, final DataFrame per track: x, y, z, px, py, pz, t, path, chords,
    status 0 = left the envelope, 1 = stopped, 2 = path/chord cap)."""
    n = len(start)
    pdg = start["pdg"].to_numpy().astype(np.int64)
    q = np.array([CHARGE.get(int(k), np.nan) for k in pdg])
    mass = np.array([MASS.get(abs(int(k)), np.nan) for k in pdg])
    if np.isnan(q).any() or np.isnan(mass).any():
        raise ValueError("unsupported PDG code in start states")
    if loss is None and np.any(mass != eloss.MMU):
        raise ValueError("the default loss model is the muon slice sampler: pass loss= for other particles")
    env = Envelope(geom) if envelope is None else envelope
    if field is None:
        field = np.zeros(3)
    if bnom is None:
        bnom = ehelix.field_at(field, np.array([[env.axis[0], env.axis[1], 0.5 * (env.z0 + env.z1)]]))[0]
    bnom = np.asarray(bnom, float)
    bmag = float(np.linalg.norm(bnom))
    bdir = bnom / bmag if bmag > 0 else np.array([0.0, 0.0, 1.0])
    bnomc = bmag * ehelix.K                                    # MeV/c per mm of bend radius
    x = start[["x", "y", "z"]].to_numpy(float).copy()
    pv = start[["px", "py", "pz"]].to_numpy(float)
    p = np.linalg.norm(pv, axis=1)
    u = pv / p[:, None]
    tk = np.sqrt(p ** 2 + mass ** 2) - mass
    t = start["t"].to_numpy(float).copy()
    path = np.zeros(n)
    xcum = np.zeros(n)
    chords = np.zeros(n, np.int64)
    status = np.full(n, 2, np.int64)
    r_in = geom.radius
    r_out = geom.wall_rmax if geom.wall_rmax is not None else r_in + 0.015
    f_metal = min(max(geom.metal_t / max(r_out - r_in, 1e-9), 0.0), 1.0)
    rw = geom.wire_r if wire else 0.0
    rw_in = max(geom.wire_r - geom.wire_plate_t, 0.0) if wire else 0.0
    loss_fn = loss if loss is not None else (lambda g, m, pp, ln, ms: tr.slab_loss(g, m, pp, ln))
    import time
    tm = transport.timing = {"field": 0.0, "straws": 0.0, "support": 0.0, "loss_msc": 0.0, "merge": 0.0,
                             "iterations": 0}
    pieces = []
    pidx = crossings.panel_index(geom)
    act = np.flatnonzero(~env.leaving(x, u))
    status[np.setdiff1d(np.arange(n), act)] = 0
    for _ in range(int(max_chords)):
        if not len(act):
            break
        tm["iterations"] += 1
        c0 = time.perf_counter()
        xa, ua, pa, qa, ma = x[act], u[act], p[act], q[act], mass[act]
        na = len(act)
        B0 = ehelix.field_at(field, xa)
        bm0 = np.linalg.norm(B0, axis=1)
        rho = np.where(bm0 > 0, pa / np.maximum(ehelix.K * np.abs(qa) * bm0, 1e-30), np.inf)
        h = np.minimum(np.minimum(chord_mm, np.sqrt(8.0 * rho * sagitta_mm)), max_path_mm - path[act])
        h = np.maximum(h, 1e-3)
        if callable(field):
            B = ehelix.field_at(field, ehelix.helix(xa, ua, pa, qa, B0, 0.5 * h)[0])
        else:
            B = B0
        xn, un = ehelix.helix(xa, ua, pa, qa, B, h)
        cv = xn - xa
        L = np.maximum(np.linalg.norm(cv, axis=1), 1e-12)
        c = cv / L[:, None]
        arc = h / L                                             # arc length per chord length
        gamma_b = pa / ma                                       # beta * gamma at the chord start
        beta = gamma_b / np.sqrt(1 + gamma_b ** 2)
        c1 = time.perf_counter()
        sp = _straw_pieces(xa, c, L, geom, r_out, rw, rw_in, pidx)
        parts = []
        if sp is not None:
            lane, js, s_in, s_out, wall, wr, pl = sp
            gas = s_out - s_in
            parts += [(materials.MYLAR, lane, wall * (1 - f_metal)), (materials.ALUMINIUM_6061, lane, wall * f_metal),
                      (st.STRAW_GAS, lane, gas), (materials.TUNGSTEN, lane, wr), (materials.GOLD, lane, pl)]
            s_mid = 0.5 * (s_in + s_out)
            pieces.append({"it": act[lane], "js": js, "g_in": path[act][lane] + s_in * arc[lane],
                           "g_out": path[act][lane] + s_out * arc[lane],
                           "pos0": xa[lane] + c[lane] * s_in[:, None], "pos1": xa[lane] + c[lane] * s_out[:, None],
                           "dir": c[lane], "p": pa[lane], "len": gas,
                           "t": t[act][lane] + s_in * arc[lane] / (beta[lane] * C_MM_PER_NS)})
        thin = _columns(na, parts)
        c2 = time.perf_counter()
        sup, mom = ({}, {})
        if interior is not None:
            sup, mom = interior._support(xa, c, L, True)
        c3 = time.perf_counter()
        # energy loss: support and solid straw material sampled, gas mean
        lost = np.zeros(na)
        for name, (m, ln) in thin.items():
            hh = np.flatnonzero(ln > 0)
            if not len(hh):
                continue
            if m is st.STRAW_GAS and loss is None:
                lost[hh] += eloss.dedx_restricted(m, pa[hh], 1e9, ma[hh]) * ln[hh]
            else:
                lost[hh] += loss_fn(rng, m, np.maximum(pa[hh], 1.0), ln[hh], ma[hh])
        for name, ln in sup.items():
            hh = np.flatnonzero(ln > 0)
            if len(hh):
                lost[hh] += loss_fn(rng, materials.get(name), np.maximum(pa[hh], 1.0), ln[hh], ma[hh])
        tk_new = tk[act] - lost
        dead = tk_new <= 0.5
        pn = np.sqrt(np.clip((np.maximum(tk_new, 0.0) + ma) ** 2 - ma ** 2, 1e-6, None))
        if msc:
            X = np.zeros(na)
            S1 = np.zeros(na)
            S2 = np.zeros(na)
            if sp is not None:
                for m, ln in ((materials.MYLAR, wall * (1 - f_metal)), (materials.ALUMINIUM_6061, wall * f_metal),
                              (st.STRAW_GAS, gas), (materials.TUNGSTEN, wr), (materials.GOLD, pl)):
                    xx = ln / eloss.radiation_length_mm(m)
                    hl = L[lane] - s_mid
                    np.add.at(X, lane, xx)
                    np.add.at(S1, lane, xx * hl)
                    np.add.at(S2, lane, xx * hl ** 2)
            for name, (ln, m1, m2) in mom.items():
                x0m = eloss.radiation_length_mm(materials.get(name))
                X += ln / x0m
                S1 += (ln * L - m1) / x0m
                S2 += (ln * L ** 2 - 2 * L * m1 + m2) / x0m
            kk = np.flatnonzero((X > 0) & ~dead)
            if len(kk):
                pm = 0.5 * (pa[kk] + pn[kk])
                V = tr.highland_var_increment(np.maximum(pm, 1.0), xcum[act][kk], xcum[act][kk] + X[kk], ma[kk])
                w1 = S1[kk] / X[kk]
                w2 = np.maximum(S2[kk] / X[kk], w1 ** 2)
                dth, _, a = st._kink_core(rng, V, w1, w2)
                if tail:
                    tparts = [(m, ln[kk]) for m, ln in thin.values()]
                    tparts += [(materials.get(nm), ln[kk]) for nm, ln in sup.items()]
                    dth = dth + st._group_tail(rng, tparts, np.arange(len(kk)), np.maximum(pm, 1.0),
                                               np.maximum(pm, 1.0), a, tail_k)
                th = np.hypot(dth[:, 0], dth[:, 1])
                big = th > 0.5
                dth[big] *= (0.5 / th[big])[:, None]
                _, dn = st._kink(xn[kk], un[kk], dth, np.zeros_like(dth))
                un[kk] = dn
            xcum[act] += X
        pmean = 0.5 * (pa + pn)
        bm = pmean / ma
        t[act] += h / (bm / np.sqrt(1 + bm ** 2) * C_MM_PER_NS)
        x[act], u[act], p[act], tk[act] = xn, un, pn, tk_new
        path[act] += h
        chords[act] += 1
        left = env.leaving(xn, un)
        capped = path[act] >= max_path_mm - 1e-9
        status[act[dead]] = 1
        status[act[left & ~dead]] = 0
        act = act[~dead & ~left & ~capped]
        c4 = time.perf_counter()
        tm["field"] += c1 - c0
        tm["straws"] += c2 - c1
        tm["support"] += c3 - c2
        tm["loss_msc"] += c4 - c3
    final = pd.DataFrame({k: start[k].to_numpy() for k in KEY})
    final["sp"] = start["sp"].to_numpy()
    final["x"], final["y"], final["z"] = x[:, 0], x[:, 1], x[:, 2]
    final["px"], final["py"], final["pz"] = (u * p[:, None]).T
    final["t"], final["path"], final["chords"], final["status"] = t, path, chords, status
    c5 = time.perf_counter()
    steps = _merge_rows(rng, pieces, start, geom, mass, q, origin, bdir, bnomc)
    tm["merge"] = time.perf_counter() - c5
    return steps, final


def _merge_rows(rng, pieces, start, geom, mass, q, origin, bdir, bnomc):
    cols = ["sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len", "width", "t",
            "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]
    if not pieces:
        return pd.DataFrame({**{k: [] for k in KEY}, "nsgs": [], **{k: [] for k in cols}})
    r = {k: np.concatenate([pc[k] for pc in pieces]) for k in pieces[0]}
    o = np.lexsort((r["g_in"], r["it"]))
    r = {k: v[o] for k, v in r.items()}
    m = len(r["it"])
    cont = np.zeros(m, bool)
    if m > 1:
        cont[1:] = ((r["it"][1:] == r["it"][:-1]) & (r["js"][1:] == r["js"][:-1])
                    & (np.abs(r["g_in"][1:] - r["g_out"][:-1]) < MERGE_TOL_MM))
    grp = np.cumsum(~cont) - 1
    first = np.flatnonzero(~cont)
    last = np.r_[first[1:], m] - 1
    it, js = r["it"][first], r["js"][first]
    ln = np.bincount(grp, r["len"])
    keep = ln > 0
    first, last, it, js, ln = first[keep], last[keep], it[keep], js[keep], ln[keep]
    p0 = r["p"][first]
    d0 = r["dir"][first]
    p1 = r["p"][last]
    pvec = 0.5 * (p0[:, None] * d0 + p1[:, None] * r["dir"][last])
    pmag = np.linalg.norm(pvec, axis=1)
    ms = mass[it]
    eion = eloss.sample_loss(rng, st.STRAW_GAS, p0, ln, st.delta_cut_mev(), ms)
    bg = p0 / ms
    ke = np.sqrt(p0 ** 2 + ms ** 2) - ms
    pstraw = bnomc * geom.radius
    shape = np.where(p0 < CURL_RATIO * pstraw, SHAPE_CURL, np.where(p0 < LINE_RATIO * pstraw, SHAPE_ARC, SHAPE_LINE))
    if bnomc > 0:
        pperp = np.linalg.norm(np.cross(d0, bdir), axis=1)
        bendrms = 0.5 * np.minimum(geom.radius, pmag * pperp / bnomc)
        sint = np.linalg.norm(np.cross(np.cross(bdir[None], d0), geom.dir[js]), axis=1)
        sagrms = SAG_PRMS * sint * ln * ln * bnomc * pperp / pmag
        width = np.minimum(sagrms, bendrms)
    else:
        width = np.zeros(len(it))
    pos0 = geometry.to_detector(r["pos0"][first], origin)
    pos1 = geometry.to_detector(r["pos1"][last], origin)
    steps = pd.DataFrame({k: start[k].to_numpy()[it] for k in KEY})
    steps["nsgs"] = np.bincount(it, minlength=len(start))[it]
    steps["sid"] = geom.sid[js]
    steps["plane"] = geom.plane[js]
    steps["panel"] = geom.panel[js]
    steps["straw"] = geom.straw[js]
    steps["upanel"] = geom.upanel[js]
    steps["shape"] = shape
    steps["ion"] = np.where((bg > st.MINION_BG) & (ke > st.MINION_KE), 0, 1)
    steps["sp"] = start["sp"].to_numpy()[it]
    steps["eion"] = eion
    steps["len"] = ln
    steps["width"] = width
    steps["t"] = r["t"][first]
    for j, cc in enumerate("xyz"):
        steps[f"{cc}0"] = pos0[:, j]
        steps[f"{cc}1"] = pos1[:, j]
        steps[f"p{cc}"] = pvec[:, j]
    return steps.reset_index(drop=True)
