"""F2 line fit: batched Gauss-Newton on drift-distance residuals, vectorised over tracks (numpy reference for JAX).

Track model (KinKal KinematicLine, no field): position p0 + u * v (t - t0), v = beta c for a muon at the fixed momentum
(KKLine keeps mom = 2 GeV/c). Fitted parameters x = (a0, b0, a1, b1, t0): the line through (a0, 0, b0) with direction
(a1, -1, b1), the LineFinder convention, which covers downward cosmics without the theta = pi/2 singularity.

Per hit (KKStrawHit, KinKal SensorLine): the signal travels along the wire from the far end to the early end at
2*halfPropV, so the wire time at a point s from the wire midpoint is t_meas - (hlen - esign s) / vsig. At the POCA of the
track and the wire, deltaT = t_wire(POCA) - t_track(POCA) and the drift residual is lr * rDrift(deltaT) - DOCA (signed).
Drift side lr and activity are inputs (from the stock fit for F2-on-stock-states, or from a fixed rule); the drift
error comes from driftInfo. Fixed iteration count, no annealing: F2 of the fastreco plan.
"""
import numpy as np

C = 299.792458                     # mm/ns
MMU = 105.6583755                  # MeV
MOM = 2000.0                       # KKLine fixed momentum, MeV/c


def _interp(grid, vals, x):
    return np.interp(x, grid, vals)


def residuals(x, H, drift):
    """Signed drift residuals r [T, N] and their Jacobian J [T, N, 5] for parameters x [T, 5]; H holds per-hit arrays."""
    a0, b0, a1, b1, t0 = (x[:, i:i + 1] for i in range(5))
    p0 = np.stack([a0, np.zeros_like(a0), b0], -1)                    # [T, 1, 3]
    d = np.stack([a1, -np.ones_like(a1), b1], -1)                     # [T, 1, 3]
    dn = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / dn
    m, w = H["mid"], H["wdir"]                                        # [T, N, 3]
    c = np.sum(u * w, -1)
    s2 = np.maximum(1.0 - c * c, 1e-12)
    dl = p0 - m
    du, dw = np.sum(dl * u, -1), np.sum(dl * w, -1)
    st = (dw * c - du) / s2                                           # along track from p0
    sw = (dw - du * c) / s2                                           # along wire from midpoint
    ptrk = p0 + u * st[..., None]
    pw = m + w * sw[..., None]
    delta = pw - ptrk                                                 # sensor - particle (KinKal convention)
    sdir = w * H["esign"][..., None]                                  # signal direction: towards the early end
    lsign = np.sign(np.sum(np.cross(sdir, u) * delta, -1))
    lsign = np.where(lsign == 0, 1.0, lsign)
    doca = np.linalg.norm(delta, axis=-1) * lsign                     # DOCA signed by angular momentum (KinKal)
    beta = MOM / np.hypot(MOM, MMU)
    v = beta * C
    ttrk = t0 + st / v
    twire = H["tmeas"] - (H["hlen"] - H["esign"] * sw) / H["vsig"]
    dT = twire - ttrk
    rdrift = _interp(drift["dt"], drift["rdrift"], dT)
    vdrift = _interp(drift["dt"], drift["dvel"], dT)
    rd_ = np.where(H["lr"] != 0, H["lr"] * rdrift - doca, -doca)    # drift: lr*rDrift - DOCA; null: DOCA vs the wire
    rt_ = dT - H["totdt"]                                             # TOT drift-time constraint (KKSHFlag tot)
    calong = sw * H["esign"]                                          # sensor POCA along sensorDirection (to the early end)
    rl_ = calong - H["wdist"] * np.where(H["esign"] * H["udir_dot_w"] < 0, -1.0, 1.0)  # longval (time division)
    r = np.stack([rd_, rt_, rl_], -1)
    J = None
    return r, doca, dT, rdrift, vdrift, J


def _model(x, H, drift):
    r, doca, dT, rd, vd, _ = residuals(x, H, drift)
    return r, doca, dT, rd


def _weights(H, drift, dT, use_tot=True, use_long=True):
    """1/variance per residual block [T, N, 3]: drift (signed drift error, or DriftInfo::nullHitVar for null hits),
    TOT (ComboHit timeVar), longitudinal (ComboHit uVar); inactive hits 0."""
    sd = _interp(drift["dt"], drift["sderr"], dT)
    ud = _interp(drift["dt"], drift["uderr"], dT)
    rd = np.maximum(0.0, _interp(drift["dt"], drift["rdrift"], dT))
    var = np.where(H["lr"] != 0, sd * sd, rd * rd + ud * ud)
    w = np.stack([1.0 / np.maximum(var, 1e-6), (1.0 / np.maximum(H["tvar"], 1e-6)) * use_tot,
                  (1.0 / np.maximum(H["uvar"], 1e-6)) * use_long], -1)
    return np.where(H["active"][..., None], w, 0.0)


def fit(x0, H, drift, niter=8, eps=(1e-3, 1e-3, 1e-6, 1e-6, 1e-3), use_tot=True, use_long=True):
    """Gauss-Newton with a numerical Jacobian; weights 1/sigma^2 from driftInfo's signed drift error (masked hits 0)."""
    x = x0.copy()
    eps = np.asarray(eps)
    for _ in range(niter):
        r, _, dT, _ = _model(x, H, drift)
        wgt = _weights(H, drift, dT, use_tot, use_long)
        J = np.empty(r.shape + (5,))
        for k in range(5):
            dx = np.zeros(5); dx[k] = eps[k]
            rp = _model(x + dx, H, drift)[0]
            rm = _model(x - dx, H, drift)[0]
            J[..., k] = (rp - rm) / (2 * eps[k])
        JW = J * wgt[..., None]
        A = np.einsum("tnki,tnkj->tij", JW, J)
        g = np.einsum("tnki,tnk->ti", JW, r)
        A += np.eye(5)[None] * 1e-12
        step = np.linalg.solve(A, -g[..., None])[..., 0]
        x = x + step
    r, doca, dT, rd = _model(x, H, drift)
    wgt = _weights(H, drift, dT, use_tot, use_long)
    J = np.empty(r.shape + (5,))
    for k in range(5):
        dx = np.zeros(5); dx[k] = eps[k]
        J[..., k] = (_model(x + dx, H, drift)[0] - _model(x - dx, H, drift)[0]) / (2 * eps[k])
    A = np.einsum("tnki,tnkj->tij", J * wgt[..., None], J) + np.eye(5)[None] * 1e-12
    cov = np.linalg.inv(A)
    chi2 = np.sum(wgt * r * r, (-1, -2))
    nres = (wgt > 0).sum((-1, -2))
    return {"x": x, "cov": cov, "chi2": chi2, "ndof": nres - 5, "r": r, "doca": doca, "dT": dT, "rdrift": rd}


def to_kinematic_line(x, cov=None):
    """(a0, b0, a1, b1, t0) -> KinKal KinematicLine (d0, phi0, z0, theta, t0) at the POCA to the z axis (+ errors)."""
    a0, b0, a1, b1, t0 = x.T
    p = np.stack([a0, np.zeros_like(a0), b0], 1)
    d = np.stack([a1, -np.ones_like(a1), b1], 1)
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    uz = u[:, 2]
    s2 = np.maximum(1.0 - uz * uz, 1e-12)
    s1 = (p[:, 2] * uz - np.sum(p * u, 1)) / s2                       # POCA to the z axis along the track
    pca = p + u * s1[:, None]
    z0 = pca[:, 2]
    dca = np.hypot(pca[:, 0], pca[:, 1])
    zc = np.cross(np.array([0.0, 0.0, 1.0])[None], np.stack([pca[:, 0], pca[:, 1], np.zeros_like(z0)], 1))
    sign = np.copysign(1.0, -np.sum(zc * u, 1))
    beta = MOM / np.hypot(MOM, MMU)
    out = {"d0": sign * dca, "phi0": np.arctan2(u[:, 1], u[:, 0]), "z0": z0, "theta": np.arccos(uz),
           "t0": t0 + s1 / (beta * C)}
    return out


def seed_params(sd_a0, sd_b0, sd_a1, sd_b1, sd_t0):
    """LineFinder seed -> fit parameters; the seed t0 refers to the y = 0 plane point (a0, 0, b0) only approximately."""
    return np.stack([sd_a0, sd_b0, sd_a1, sd_b1, sd_t0], 1).astype(np.float64)


def build_hits(trk_of_hit, sh_index, lr, active, sh, cond, ntrk, nmax=None):
    """Per-track padded hit tensors from flat (track, makeSH index, lr, active) lists and the makeSH columns."""
    from .hitreco import unique_straw
    st = cond["straw"]
    o = np.lexsort((np.arange(len(trk_of_hit)), trk_of_hit))
    trk_of_hit, sh_index, lr, active = trk_of_hit[o], sh_index[o], lr[o], active[o]
    cnt = np.bincount(trk_of_hit, minlength=ntrk)
    N = nmax or max(int(cnt.max()) if len(cnt) else 1, 1)
    pos = np.arange(len(trk_of_hit)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    us = unique_straw(sh["sid"][sh_index].astype(np.int64))
    H = {"active": np.zeros((ntrk, N), bool), "lr": np.zeros((ntrk, N)), "tmeas": np.zeros((ntrk, N)),
         "hlen": np.ones((ntrk, N)), "esign": np.ones((ntrk, N)), "vsig": np.ones((ntrk, N)),
         "mid": np.zeros((ntrk, N, 3)), "wdir": np.zeros((ntrk, N, 3)), "sh": np.full((ntrk, N), -1)}
    H["wdir"][..., 0] = 1.0
    H["active"][trk_of_hit, pos] = active.astype(bool)
    H["lr"][trk_of_hit, pos] = lr
    H["sh"][trk_of_hit, pos] = sh_index
    eend = sh["eend"][sh_index]
    H["tmeas"][trk_of_hit, pos] = np.where(eend == 1, sh["et1"][sh_index], sh["et0"][sh_index]).astype(np.float64)
    H["hlen"][trk_of_hit, pos] = st["hlen"][us]
    # wire direction points from the hv end to the cal end (Straw::wireEnd: cal = mid + hlen*dir); the early end is
    # cal (eend 0) or hv (eend 1); s along +dir from the midpoint, so the far end distance is hlen - esign*s with esign
    # = +1 for cal early, -1 for hv early
    H["esign"][trk_of_hit, pos] = np.where(eend == 1, -1.0, 1.0)
    H["mid"][trk_of_hit, pos] = np.stack([st["midx"][us], st["midy"][us], st["midz"][us]], 1)
    H["wdir"][trk_of_hit, pos] = np.stack([st["dirx"][us], st["diry"][us], st["dirz"][us]], 1)
    H["vsig"][trk_of_hit, pos] = 2.0 * sh["halfpv"][sh_index]          # 2*halfPropV(edep), as Mu2eKinKal::hitLine
    for k, col in (("totdt", "dtime"), ("tvar", "timevar"), ("uvar", "uvar"), ("wdist", "wdist")):
        H[k] = np.zeros((ntrk, N)) if k != "tvar" and k != "uvar" else np.ones((ntrk, N))
        H[k][trk_of_hit, pos] = sh[col][sh_index].astype(np.float64)
    H["udir_dot_w"] = np.ones((ntrk, N))   # ComboHit uDir is the straw direction for single-straw hits
    return H


def from_kinematic_line(d0, phi0, z0, theta, t0):
    """KinKal KinematicLine -> fit parameters (a0, b0, a1, b1, t0 at the y = 0 point)."""
    p0 = np.stack([-d0 * np.sin(phi0), d0 * np.cos(phi0), z0], 1)
    u = np.stack([np.sin(theta) * np.cos(phi0), np.sin(theta) * np.sin(phi0), np.cos(theta)], 1)
    s = -p0[:, 1] / u[:, 1]
    p = p0 + u * s[:, None]
    d = u / (-u[:, 1:2])
    beta = MOM / np.hypot(MOM, MMU)
    return np.stack([p[:, 0], p[:, 2], d[:, 0], d[:, 2], t0 + s / (beta * C)], 1)
