"""JAX twin of linefit.fit (F2): the same residual model per hit, exact Jacobian by forward-mode autodiff, vmapped over
tracks, fixed iteration count, float64. Drift radii are frozen in the Jacobian (stop_gradient) unless drift_dt, as
stock KKLine (no DriftDt flag). The numpy module (numerical Jacobian) is the reference; agreement is within its
finite-difference error, not bitwise. Hits are padded to a bucketed length per batch (inactive pad hits weigh 0).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .linefit import C, MMU, MOM

jax.config.update("jax_enable_x64", True)

HIT_KEYS = ("active", "lr", "tmeas", "hlen", "esign", "vsig", "mid", "wdir", "totdt", "tvar", "uvar", "wdist", "udir_dot_w",
            "edep")
_ANN = {}   # stage index -> (sign layers, sign cut, cluster layers, cluster cut), filled by ann_schedule()


def stock_schedule(seed=True):
    """CHSEEDFIT (if seed) then CHDRIFTEXT, meta-iteration by meta-iteration: temperature, CADSHU settings
    (maxdoca, maxderr, minrdrift, maxrdrift), BkgANN and DriftANN stages, as in Mu2eKinKal/fcl/prolog.fcl."""
    from . import driftann
    _ANN["bseed"] = (driftann.load(driftann.find("TrainBkgSeed.dat")), 0.2)
    _ANN["bfinal"] = (driftann.load(driftann.find("TrainBkgFinal.dat")), 0.2)
    for i, (sw, sc, cw, cc) in enumerate(driftann.SCHEDULE):
        _ANN[i] = (driftann.load(driftann.find(sw)), sc, driftann.load(driftann.find(cw)), cc)
    st = []
    if seed:
        for T, cad in zip((5.0, 5.0, 2.0, 1.0), ((400.0, 800000.0, 50.0, 50.0), (40.0, 80.0, 5.0, 5.0),
                                                 (20.0, 40.0, 5.0, 4.0), (10.0, 20.0, 5.0, 3.0))):
            st.append(("stock", T, cad, -1, -1))
    temps = (2.0, 1.0, 0.3, 0.0, 1.0, 0.5, 0.2, 0.0, 0.0)
    cads = ((10.0, 20.0, 5.0, 5.0), (10.0, 20.0, 5.0, 4.0), (8.0, 16.0, 5.0, 3.0), (8.0, 16.0, 5.0, 3.0),
            (6.0, 10.0, 5.0, 3.0), (5.0, 10.0, 5.0, 3.0), (5.0, 10.0, 5.0, 3.0), (5.0, 10.0, 5.0, 3.0), (5.0, 10.0, 5.0, 3.0))
    bkg = (-1, -1, -1, "bseed", -1, -1, -1, "bfinal", -1)
    ann = (-1, -1, -1, -1, 0, 1, 2, 3, 4)
    for T, cad, b, a in zip(temps, cads, bkg, ann):
        st.append(("stock", T, cad, b, a))
    return tuple(st)


def ann_schedule():
    """The stock CHDRIFTEXT extension: three CADSHU meta-iterations (TOT:LongVal, then NullDriftVar), then the five
    DriftANN meta-iterations, as (kind, ...) stages for fit_own_states. CADSHU (maxdoca, maxrdrift) gates come from
    CHDRIFTEXT's CADSHUSettings; the BkgANN and annealing temperatures are not modelled."""
    from . import driftann
    stages = [(10.0, 1.0, 3.0), (10.0, 1.0, 3.0), (8.0, 1.0, 3.0)]
    gates = [(8.0, 3.0), (6.0, 3.0), (5.0, 3.0), (5.0, 3.0), (5.0, 3.0)]
    for i, ((sw, sc, cw, cc), (md, mr)) in enumerate(zip(driftann.SCHEDULE, gates)):
        _ANN[i] = (driftann.load(driftann.find(sw)), sc, driftann.load(driftann.find(cw)), cc)
        stages.append(("ann", i, md, mr))
    return tuple(stages)
DRIFT_KEYS = ("dt", "rdrift", "sderr", "uderr")


def _resid_one(x, h, d, rd_frozen, drift_dt):
    """Residual blocks [N, 3] for one track; rd_frozen is used for the drift radius when not drift_dt."""
    a0, b0, a1, b1, t0 = x[0], x[1], x[2], x[3], x[4]
    p0 = jnp.array([a0, 0.0, b0])
    dvec = jnp.array([a1, -1.0, b1])
    u = dvec / jnp.linalg.norm(dvec)
    m, w = h["mid"], h["wdir"]
    c = w @ u
    s2 = jnp.maximum(1.0 - c * c, 1e-12)
    dl = p0[None, :] - m
    du, dw = dl @ u, jnp.sum(dl * w, -1)
    st = (dw * c - du) / s2
    sw = (dw - du * c) / s2
    delta = (m + w * sw[:, None]) - (p0[None, :] + u[None, :] * st[:, None])
    sdir = w * h["esign"][:, None]
    lsign = jnp.sign(jnp.sum(jnp.cross(sdir, jnp.broadcast_to(u, sdir.shape)) * delta, -1))
    lsign = jnp.where(lsign == 0, 1.0, lsign)
    doca = jnp.sqrt(jnp.sum(delta * delta, -1) + 1e-30) * lsign
    v = (x[5] / jnp.hypot(x[5], MMU) if x.shape[0] > 5 else MOM / jnp.hypot(MOM, MMU)) * C   # F1: momentum is x[5]
    dT = h["tmeas"] - (h["hlen"] - h["esign"] * sw) / h["vsig"] - (t0 + st / v)
    rd_live = jnp.interp(dT, d["dt"], d["rdrift"])
    rdrift = rd_live if drift_dt else rd_frozen
    rdr = jnp.where(h["lr"] != 0, h["lr"] * rdrift - doca, -doca)
    rt = dT - h["totdt"]
    rl = sw * h["esign"] - h["wdist"] * jnp.where(h["esign"] * h["udir_dot_w"] < 0, -1.0, 1.0)
    return jnp.stack([rdr, rt, rl], -1), dT, rd_live, doca


def _weights_one(h, d, dT, use_tot, use_long):
    sd = jnp.interp(dT, d["dt"], d["sderr"])
    ud = jnp.interp(dT, d["dt"], d["uderr"])
    rd = jnp.maximum(0.0, jnp.interp(dT, d["dt"], d["rdrift"]))
    var = jnp.where(h["lr"] != 0, sd * sd, rd * rd + ud * ud)
    wt = jnp.stack([1.0 / jnp.maximum(var, 1e-6), (1.0 / jnp.maximum(h["tvar"], 1e-6)) * use_tot,
                    (1.0 / jnp.maximum(h["uvar"], 1e-6)) * use_long], -1)
    return jnp.where(h["active"][:, None], wt, 0.0)


def _fit_one(x0, h, d, niter, drift_dt, use_tot, use_long, vscale=1.0):
    def step(x, _):
        r, dT, rd_live, _ = _resid_one(x, h, d, None, True)
        wt = _weights_one(h, d, dT, use_tot, use_long) / vscale
        rd_f = jax.lax.stop_gradient(rd_live)
        J = jax.jacfwd(lambda xx: _resid_one(xx, h, d, rd_f, drift_dt)[0])(x)       # [N, 3, 5]
        A = jnp.einsum("nki,nk,nkj->ij", J, wt, J) + 1e-12 * jnp.eye(5)
        g = jnp.einsum("nki,nk,nk->i", J, wt, r)
        return x - jnp.linalg.solve(A, g), None

    x, _ = jax.lax.scan(step, x0, None, length=niter)
    r, dT, rd_live, doca = _resid_one(x, h, d, None, True)
    wt = _weights_one(h, d, dT, use_tot, use_long) / vscale
    J = jax.jacfwd(lambda xx: _resid_one(xx, h, d, jax.lax.stop_gradient(rd_live), drift_dt)[0])(x)
    A = jnp.einsum("nki,nk,nkj->ij", J, wt, J) + 1e-12 * jnp.eye(5)
    chi2 = jnp.sum(wt * r * r)
    return x, jnp.linalg.inv(A), chi2, jnp.sum(wt > 0) - 5, doca, dT


import functools


@functools.partial(jax.jit, static_argnames=("niter", "drift_dt", "use_tot", "use_long"))
def _fit_batch(x0, h, d, niter, drift_dt, use_tot, use_long):
    f = functools.partial(_fit_one, niter=niter, drift_dt=drift_dt, use_tot=use_tot, use_long=use_long)
    return jax.vmap(f, in_axes=(0, 0, None))(x0, h, d)


def _pad_hits(H, n_pad):
    out = {}
    for k in HIT_KEYS:
        if k not in H:
            continue
        a = np.asarray(H[k])
        pad = [(0, 0), (0, n_pad - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
        fill = 1.0 if k in ("tvar", "uvar", "hlen", "vsig", "esign", "udir_dot_w") else 0.0
        b = np.pad(a, pad, constant_values=fill)
        if k == "wdir" and n_pad > a.shape[1]:
            b[:, a.shape[1]:, 0] = 1.0
        out[k] = b
    return out


def fit(x0, H, drift, niter=8, drift_dt=False, use_tot=True, use_long=True, n_pad=None, chunk=65536):
    """numpy-in, numpy-out wrapper with the linefit.fit keys (x, cov, chi2, ndof, doca, dT)."""
    N = H["active"].shape[1]
    n_pad = n_pad or max(16, 1 << int(np.ceil(np.log2(max(N, 1)))))
    Hp = _pad_hits(H, n_pad)
    d = {k: jnp.asarray(np.asarray(drift[k], np.float64)) for k in DRIFT_KEYS}
    outs = []
    T = x0.shape[0]
    for t0 in range(0, T, chunk):
        sl = slice(t0, t0 + chunk)
        n = min(chunk, T - t0)
        cpad = chunk if T > chunk else n
        hx = {k: np.concatenate([v[sl], np.repeat(v[sl][-1:], cpad - n, 0)]) if cpad > n else v[sl] for k, v in Hp.items()}
        xx = np.concatenate([x0[sl], np.repeat(x0[sl][-1:], cpad - n, 0)]) if cpad > n else x0[sl]
        o = _fit_batch(jnp.asarray(xx), {k: jnp.asarray(v) for k, v in hx.items()}, d,
                       niter=niter, drift_dt=drift_dt, use_tot=use_tot, use_long=use_long)
        outs.append([np.asarray(a)[:n] for a in o])
    x, cov, chi2, ndof, doca, dT = (np.concatenate([o[i] for o in outs]) for i in range(6))
    return {"x": x, "cov": cov, "chi2": chi2, "ndof": ndof, "doca": doca[:, :N], "dT": dT[:, :N]}


def _states_one(x, h, d, maxdoca, nullsig, cons):
    """Hit states from the current line (a CADSHU-like rule standing in for the drift ANN): active if |DOCA| < maxdoca;
    left/right from the DOCA sign where |DOCA| > nullsig x drift error and the drift radius agrees with |DOCA| within
    cons x drift error, else null."""
    _, dT, rd_live, doca = _resid_one(x, h, d, None, True)
    sd = jnp.interp(dT, d["dt"], d["sderr"])
    act = h["valid"] & (jnp.abs(doca) < maxdoca)
    drift = (jnp.abs(doca) > nullsig * sd) & (jnp.abs(jnp.abs(doca) - rd_live) < cons * sd)
    lr = jnp.where(drift, jnp.sign(doca), 0.0)
    return act, lr


def _hitgeom(x, h, d, cov):
    """DOCA, its variance from the parameter covariance, rDrift, upos (signal-side position along the wire), particle
    POCA rho and the wire-track direction cosine, at parameters x."""
    _, dT, rd_live, doca = _resid_one(x, h, d, None, True)
    J = jax.jacfwd(lambda xx: _resid_one(xx, h, d, None, True)[3])(x)
    docavar = jnp.einsum("ni,ij,nj->n", J, cov, J)
    a0, b0, a1, b1 = x[0], x[1], x[2], x[3]
    p0 = jnp.array([a0, 0.0, b0]); u = jnp.array([a1, -1.0, b1]); u = u / jnp.linalg.norm(u)
    m, w = h["mid"], h["wdir"]
    c = w @ u
    s2 = jnp.maximum(1.0 - c * c, 1e-12)
    dl = p0[None, :] - m
    du, dw = dl @ u, jnp.sum(dl * w, -1)
    st, sw = (dw * c - du) / s2, (dw - du * c) / s2
    pp = p0[None, :] + u[None, :] * st[:, None]
    return doca, docavar, rd_live, sw, jnp.hypot(pp[:, 0], pp[:, 1]), c


def _stage_cad(h, geo, maxdoca, maxderr, minrdrift, maxrdrift):
    """CADSHU::wireHitState: usable if rDrift < maxrdrift, |DOCA| < maxdoca, 0 < docaVar < maxderr^2; then drift if
    rDrift > minrdrift (sign of DOCA), else null; otherwise inactive. Frozen-inactive hits stay inactive."""
    doca, docavar, rd, _, _, _ = geo
    use = (rd < maxrdrift) & (jnp.abs(doca) < maxdoca) & (docavar > 0) & (docavar < maxderr * maxderr)
    act = h["valid"] & ~h["frozen"] & use
    lr = jnp.where(use & (rd > minrdrift), jnp.sign(doca), 0.0)
    return act, jnp.where(act, lr, 0.0)


def _stage_bkg(h, geo, layers, cut, act, lr):
    """BkgANNSHU: inputs |DOCA|, cDrift, sigma(DOCA), TOT drift time, |wireDist - upos|, POCA rho; output < cut ->
    inactive (and frozen: 'Inactive' freeze); an inactive hit above the cut becomes null."""
    from . import driftann
    doca, docavar, rd, sw, rho, _ = geo
    x = jnp.stack([jnp.abs(doca), rd, jnp.sqrt(jnp.maximum(docavar, 0.0)), h["totdt"], jnp.abs(h["wdist"] - sw), rho], -1)
    q = driftann.mlp(layers, x)
    bad = h["valid"] & (q < cut)
    act2 = jnp.where(bad, False, jnp.where(h["valid"] & ~act & ~h["frozen"], True, act))
    lr2 = jnp.where(bad, 0.0, jnp.where(act, lr, 0.0))
    return act2, lr2, h["frozen"] | bad


def _states_ann(x, h, d, ann, maxdoca, maxrdrift, cov):
    """CADSHU gate (active if |DOCA| < maxdoca and rDrift < maxrdrift) then the drift ANNs (driftann) decide drift vs
    null: setLR = cluster > cut and sign > cut. DOCA variance from the current parameter covariance."""
    from . import driftann
    _, dT, rd_live, doca = _resid_one(x, h, d, None, True)
    J = jax.jacfwd(lambda xx: _resid_one(xx, h, d, None, True)[3])(x)            # dDOCA/dx [N, 5]
    docavar = jnp.einsum("ni,ij,nj->n", J, cov, J)
    u = jnp.array([x[2], -1.0, x[3]]); u = u / jnp.linalg.norm(u)
    dirdot = h["wdir"] @ u
    sp, cp = driftann.inputs(doca, docavar, rd_live, rd_live, h["totdt"], h["edep"], dirdot)
    sq = driftann.mlp(ann[0], sp)
    cq = driftann.mlp(ann[2], cp)
    act = h["valid"] & (jnp.abs(doca) < maxdoca) & (rd_live < maxrdrift)
    setlr = (cq > ann[3]) & (sq > ann[1])
    return act, jnp.where(setlr, jnp.sign(doca), 0.0), sq, cq


@functools.partial(jax.jit, static_argnames=("niter", "use_tot", "use_long", "schedule"))
def _fit_own_states(x0, h, d, niter, use_tot, use_long, schedule):
    """Alternate state updates and fixed-iteration fits; schedule = ((maxdoca, nullsig), ...) per pass."""
    def one(x0_, h_):
        x = x0_
        hh = dict(h_)
        cov = jnp.eye(5) * 1e4
        hh["frozen"] = jnp.zeros_like(hh["valid"])
        vs = 1.0
        failed = jnp.array(False)
        prev = jnp.array(jnp.inf)
        for stage in schedule:
            if stage[0] == "ann":
                act, lr, _, _ = _states_ann(x, hh, d, _ANN[stage[1]], stage[2], stage[3], cov)
                act = act & hh["active"]
            elif stage[0] == "stock":                     # ("stock", T, cad settings, bkg index or -1, ann index or -1)
                _, T, cad, ib, ia = stage
                geo = _hitgeom(x, hh, d, cov)
                act, lr = _stage_cad(hh, geo, *cad)
                if ib != -1:
                    act, lr, hh["frozen"] = _stage_bkg(hh, geo, _ANN[ib][0], _ANN[ib][1], act, lr)
                if ia != -1:
                    hh["active"] = act
                    _, lra, _, _ = _states_ann(x, hh, d, _ANN[ia], 1e9, 1e9, cov)
                    lr = jnp.where(act, lra, 0.0)
                vs = (1.0 + T) ** 2
            else:
                act, lr = _states_one(x, hh, d, *stage)
            hh["active"], hh["lr"] = act, lr
            xn, covn, chi2, ndof = _fit_one(x, hh, d, niter, False, use_tot, use_long, vs)[:4]
            if stage[0] == "stock":
                # KinKal Track status after the meta-iteration: lowNDOF (ndof < MinNDOF 1) or chisqdiverged (chi2/ndof
                # rose by more than DivergenceDeltaChisq: 20 in CHSEEDFIT, 10 in CHDRIFTEXT); a failed fit keeps its
                # last usable state and is not extended further
                cpn = chi2 / jnp.maximum(ndof, 1)
                div = 20.0 if stage[1] >= 1.0 and len(stage) and stage[2][0] > 10.0 else 10.0
                bad = (ndof < 1) | (jnp.isfinite(prev) & (cpn - prev > div)) | ~jnp.isfinite(cpn)
                failed = failed | bad
                prev = jnp.where(failed, prev, cpn)
                xn = jnp.where(failed, x, xn)
                covn = jnp.where(failed, cov, covn)
            x, cov = xn, covn
        out = _fit_one(x, hh, d, niter, False, use_tot, use_long, vs)
        return out + (hh["active"], hh["lr"], failed)
    return jax.vmap(one)(x0, h)


SCHEDULE = ((20.0, 1.0, 3.0), (10.0, 1.0, 3.0), (5.0, 1.0, 3.0), (5.0, 1.0, 3.0))   # (maxdoca, nullsig, cons) per pass


def fit_own_states(x0, H, drift, niter=4, use_tot=True, use_long=True, schedule=SCHEDULE, n_pad=None, chunk=32768):
    """Fit with hit states decided by the fast reco itself (starting from all hits of the seed); returns linefit keys +
    active, lr."""
    N = H["active"].shape[1]
    n_pad = n_pad or max(16, 1 << int(np.ceil(np.log2(max(N, 1)))))
    Hp = _pad_hits(H, n_pad)
    Hp["valid"] = np.pad(np.asarray(H["sh"]) >= 0, [(0, 0), (0, n_pad - N)])
    d = {k: jnp.asarray(np.asarray(drift[k], np.float64)) for k in DRIFT_KEYS}
    outs = []
    T = x0.shape[0]
    for t0 in range(0, T, chunk):
        n = min(chunk, T - t0)
        cpad = chunk if T > chunk else n
        sl = slice(t0, t0 + n)
        hx = {k: np.concatenate([v[sl], np.repeat(v[sl][-1:], cpad - n, 0)]) if cpad > n else v[sl] for k, v in Hp.items()}
        xx = np.concatenate([x0[sl], np.repeat(x0[sl][-1:], cpad - n, 0)]) if cpad > n else x0[sl]
        o = _fit_own_states(jnp.asarray(xx), {k: jnp.asarray(v) for k, v in hx.items()}, d, niter=niter,
                            use_tot=use_tot, use_long=use_long, schedule=tuple(schedule))
        outs.append([np.asarray(a)[:n] for a in o])
    keys = ("x", "cov", "chi2", "ndof", "doca", "dT", "active", "lr", "failed")
    res = {k: np.concatenate([o[i] for o in outs]) for i, k in enumerate(keys)}
    for k in ("doca", "dT", "active", "lr"):
        res[k] = res[k][:, :N]
    return res
