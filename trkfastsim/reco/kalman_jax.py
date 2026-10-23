"""F1: KinKal-like Kalman fit of the extracted (no field) KinematicLine, batched over tracks in JAX.

Parameters x = (a0, b0, a1, b1, t0) as F2 (linefit): line through (a0, 0, b0), direction (a1, -1, b1), t0 at that point.
Effects, ordered by their track length s along the direction of motion (time order):
  hits      linearised residual rows of linefit_jax._resid_one (drift or null DOCA, TOT time, longitudinal), weight
            1 / (variance x (1+T)^2), as KinKal ResidualHit::updateWeight;
  crossings each straw the track crosses (KKStrawXing), placed just after its hit (toff = wire radius / speed); its
            effect is process noise: two projected scattering angles of variance kkmaterial.straw_scatter_var(...)
            mapped to the parameters of the downstream piece (exact Jacobian of the re-parametrisation at the
            crossing point), as ElementXing::parameterChange. Energy loss is not modelled (momentum fixed, as KKLine
            with its 1e-4 momentum constraint).
Part 1 (this section): geometry and effect construction. Filter, smoother and iteration below.
"""
import jax
import jax.numpy as jnp

from . import kkmaterial
from .linefit import C, MMU, MOM

jax.config.update("jax_enable_x64", True)

V = MOM / (MOM ** 2 + MMU ** 2) ** 0.5 * C           # muon speed at the KKLine momentum, mm/ns


def point_dir(x):
    """Reference point (a0, 0, b0) and unit direction of motion."""
    d = jnp.array([x[2], -1.0, x[3]])
    return jnp.array([x[0], 0.0, x[1]]), d / jnp.linalg.norm(d)


def params_from(p, u, t):
    """Line through point p with unit direction u (u_y < 0), time t at p -> (a0, b0, a1, b1, t0)."""
    s = -p[1] / u[1]                                   # to the y = 0 plane
    q = p + u * s
    return jnp.array([q[0], q[2], -u[0] / u[1], -u[2] / u[1], t + s / V])


def transverse_basis(u):
    """Two unit vectors perpendicular to u (KinKal MomBasis perpdir/phidir analogue; the projected variance is
    isotropic, so only orthonormality matters)."""
    e1 = jnp.cross(u, jnp.array([0.0, 0.0, 1.0]))
    e1 = jnp.where(jnp.linalg.norm(e1) < 1e-9, jnp.array([1.0, 0.0, 0.0]), e1)
    e1 = e1 / jnp.linalg.norm(e1)
    return e1, jnp.cross(u, e1)


def kink(x, s, th):
    """Parameters of the line after a kink th = (th1, th2) [rad] at track length s from the reference point."""
    p0, u = point_dir(x)
    e1, e2 = transverse_basis(u)
    un = u + th[0] * e1 + th[1] * e2
    un = un / jnp.linalg.norm(un)
    return params_from(p0 + u * s, un, x[4] + s / V)


def kink_jacobian(x, s):
    """d(params after kink)/d(th1, th2) at th = 0: [5, 2], closed form. The kink point q = p0 + u s stays; the
    direction moves by du = e_k. With t = q_y / u_y (track length from y = 0 to q): a = q_x - u_x t, b = q_z - u_z t,
    a1 = -u_x / u_y, b1 = -u_z / u_y, t0 = t_q - t / v. Differentiating in u at fixed q:
    dt = -t du_y / u_y, da = -du_x t - u_x dt, db1 = -(du_z - b1' ...) as below."""
    p0, u = point_dir(x)
    e1, e2 = transverse_basis(u)
    q = p0 + u * s
    t = q[1] / u[1]
    E = jnp.stack([e1, e2], 1)                          # [3, 2]
    dt = -t * E[1] / u[1]
    da0 = -E[0] * t - u[0] * dt
    db0 = -E[2] * t - u[2] * dt
    da1 = -E[0] / u[1] + u[0] * E[1] / u[1] ** 2
    db1 = -E[2] / u[1] + u[2] * E[1] / u[1] ** 2
    dt0 = -dt / V
    return jnp.stack([da0, db0, da1, db1, dt0], 0)


def kink_jacobian_ad(x, s):
    """Reference for kink_jacobian (forward-mode autodiff of kink)."""
    return jax.jacfwd(lambda th: kink(x, s, th))(jnp.zeros(2))


def crossing_noise(x, s, scatvar):
    """Process noise covariance [5, 5] of one crossing with projected angle variance scatvar."""
    D = kink_jacobian(x, s)
    return scatvar * D @ D.T


def straw_poca(x, mid, wdir):
    """Track length s of the POCA to each straw axis, signed DOCA magnitude, and track-straw direction cosine.
    mid, wdir [M, 3]."""
    p0, u = point_dir(x)
    c = wdir @ u
    s2 = jnp.maximum(1.0 - c * c, 1e-12)
    dl = p0[None, :] - mid
    du, dw = dl @ u, jnp.sum(dl * wdir, -1)
    st = (dw * c - du) / s2
    sw = (dw - du * c) / s2
    delta = (mid + wdir * sw[:, None]) - (p0[None, :] + u[None, :] * st[:, None])
    return st, jnp.sqrt(jnp.sum(delta * delta, -1) + 1e-30), c


def crossing_effects(x, cov, mid, wdir, valid, nsig, varscale, mom=MOM):
    """Per crossing: track length s (+ toff so it follows its hit), process noise Q [M, 5, 5] and the path/scatter
    values. DOCA variance from the parameter covariance (ClosestApproachData::docaVar); nsig and varscale from the
    StrawXingUpdater of the meta-iteration (scalevar: varscale = (1+T)^2, else 1)."""
    st, doca, c = straw_poca(x, mid, wdir)
    J = jax.jacfwd(lambda xx: straw_poca(xx, mid, wdir)[1])(x)          # [M, 5]
    docavar = jnp.einsum("mi,ij,mj->m", J, cov, J)
    wall, gas = kkmaterial.path_lengths(doca, docavar, c, nsig)
    sv = kkmaterial.scatter_angle_var("straw-wall", mom, wall) + kkmaterial.scatter_angle_var("straw-gas", mom, gas)
    sv = jnp.where(valid, sv * varscale, 0.0)
    D = jax.vmap(lambda s: kink_jacobian(x, s))(st)
    toff = kkmaterial.STRAW["wrad"]
    return {"s": st + toff, "D": D, "scatvar": sv, "doca": doca, "docavar": docavar, "wall": wall, "gas": gas,
            "dirdot": c}


# ---------------------------------------------------------------------------------------------------------------------
# Part 2: forward/backward filter and smoother over time-ordered effects (Track::iterate, Measurement/Material).
# With no field and no energy loss the parameters are global (every piece has the same parametrisation and a
# crossing shifts no mean), so a crossing only adds Q to the covariance: a random walk between pieces.
# Effect arrays per track, padded to E and sorted by s (pad effects have w = 0 and Q = 0):
#   xref [E, 5]  reference parameters of the effect's piece (linearisation point)
#   H [E, 3, 5]  d(residual)/d(params) at xref;  r [E, 3] residual at xref
#   w [E, 3]     weight 1 / (mvar x varscale) per residual row, 0 if inactive;  mvar [E, 3] unscaled variance
#   Q [E, 5, 5]  process noise (crossings only)
# Information state (W, b) with x = W^-1 b. A hit adds W += H^T w H, b += H^T w (H xref - r), i.e. the KinKal
# Residual::weight with dRdP = -H and value = r.

def _solve(W, b):
    return jnp.linalg.solve(W, b)


def _add_noise(W, b, D, sv):
    """(W, b) after process noise Q = sv D D^T (D [5, 2], rank 2), parameters unchanged (Material::process). Woodbury:
    W' = W - W D (sv D^T W D + I)^-1 sv D^T W, b' = W' W^-1 b = b - W D (sv D^T W D + I)^-1 sv D^T b; only a 2x2
    solve, exact at sv = 0."""
    WD = W @ D
    S = sv * (D.T @ WD) + jnp.eye(2)
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    Si = jnp.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det     # explicit 2x2 inverse: no solver call
    K_ = Si @ (sv * WD.T)                                                  # [2, 5]
    return W - WD @ K_, b - WD @ (Si @ (sv * (D.T @ b)))


INV = __import__("os").environ.get("F1_INV", "lu")             # lu (jnp.linalg.inv) or chol


def spd_inv(M):
    """Inverse of symmetric positive-definite matrices [..., 5, 5]: LU (default) or Cholesky (F1_INV=chol)."""
    if INV != "chol":
        return jnp.linalg.inv(M)
    L = jnp.linalg.cholesky(0.5 * (M + jnp.swapaxes(M, -1, -2)))
    I = jnp.broadcast_to(jnp.eye(M.shape[-1]), M.shape)
    Li = jax.scipy.linalg.solve_triangular(L, I, lower=True)
    return jnp.swapaxes(Li, -1, -2) @ Li


def _hit_info(e):
    Hw = e["H"] * e["w"][:, None]
    return jnp.einsum("ki,kj->ij", Hw, e["H"]), Hw.T @ (e["H"] @ e["xref"] - e["r"])


def _predicted_chi2(W, b, e):
    """Chisq of the hit's rows against state (W, b) (ResidualHit::chisq): predicted residual r + H (x - xref), variance
    mvar + H C H^T (measurement variance not temperature-scaled; rows treated separately, as KinKal)."""
    C = spd_inv(W)
    x = C @ b
    res = e["r"] + e["H"] @ (x - e["xref"])
    var = e["mvar"] + jnp.einsum("ki,ij,kj->k", e["H"], C, e["H"])
    act = e["w"] > 0
    return jnp.sum(jnp.where(act, res * res / jnp.where(act, var, 1.0), 0.0)), jnp.sum(act)


def filter_smooth(eff, W0f, b0f, W0b, b0b):
    """One algebraic iteration for one track. eff holds D [E, 5, 2] and sv [E] (crossing noise Q = sv D D^T) instead
    of Q. The scans only accumulate information; the forward chi2 is evaluated afterwards from the stored
    pre-effect forward states, in parallel over effects."""
    def fwd(carry, e):
        W, b = carry
        pre = (W, b)
        dW, db = _hit_info(e)
        W, b = _add_noise(W + dW, b + db, e["D"], e["sv"])
        return (W, b), (pre, (W, b))

    def bwd(carry, e):
        W, b = carry
        out = (W, b)
        W, b = _add_noise(W, b, e["D"], e["sv"])
        dW, db = _hit_info(e)
        return (W + dW, b + db), out

    _, ((Wp, bp), (Wf, bf)) = jax.lax.scan(fwd, (W0f, b0f), eff, unroll=UNROLL)
    (Wbf, bbf), (Wb, bb) = jax.lax.scan(bwd, (W0b, b0b), eff, reverse=True, unroll=UNROLL)
    c, n = jax.vmap(_predicted_chi2)(Wp, bp, eff)
    return {"W": Wf + Wb, "b": bf + bb, "chi2": jnp.sum(c), "ndof": jnp.sum(n) - 5, "Wfront": Wbf, "bfront": bbf}


# Parallel-in-time version (Sarkka and Garcia-Fernandez, IEEE TAC 66 (2021) 299): each effect is an element
# (A, b, C, eta, J) of the conditional p(x_{k+1} | x_k, y_k) = N(A x_k + b, C) and the likelihood p(y_k | x_k) in
# information form; the combination is associative, so lax.associative_scan gives every filtered state in log2(E)
# levels instead of E sequential steps. Same results as filter_smooth (tests).

def _sym(M):
    return 0.5 * (M + jnp.swapaxes(M, -1, -2))


def _combine(ei, ej):
    """Associative operator, ei before ej (batched over leading axes)."""
    Ai, bi, Ci, ei_, Ji = ei
    Aj, bj, Cj, ej_, Jj = ej
    I = jnp.eye(5)
    M = I + Ci @ Jj
    Xa = jnp.linalg.solve(M, jnp.concatenate([Ai, (bi + (Ci @ ej_[..., None])[..., 0])[..., None], Ci], -1))
    A = Aj @ Xa[..., :5]
    b = (Aj @ Xa[..., 5:6])[..., 0] + bj
    C = _sym(Aj @ Xa[..., 6:] @ jnp.swapaxes(Aj, -1, -2) + Cj)
    Nm = I + Jj @ Ci
    Ya = jnp.linalg.solve(Nm, jnp.concatenate([(ej_ - (Jj @ bi[..., None])[..., 0])[..., None], Jj], -1))
    AiT = jnp.swapaxes(Ai, -1, -2)
    eta = (AiT @ Ya[..., :1])[..., 0] + ei_
    J = _sym(AiT @ Ya[..., 1:] @ Ai + Ji)
    return A, b, C, eta, J


def _elements(Jm, eta, Q, P0, m0, J0, eta0, Q0):
    """Elements for effects 1..E-1 (measurement then noise) and element 0 (prior + measurement 0 + noise 0)."""
    E = Jm.shape[0]
    A = jnp.broadcast_to(jnp.eye(5), (E, 5, 5))
    b = jnp.zeros((E, 5))
    Pp = jnp.linalg.inv(jnp.linalg.inv(P0) + J0)
    mp = Pp @ (jnp.linalg.solve(P0, m0) + eta0)
    A = A.at[0].set(jnp.zeros((5, 5)))
    b = b.at[0].set(mp)
    C = Q.at[0].set(_sym(Pp + Q0))
    eta = eta.at[0].set(jnp.zeros(5))
    Jm = Jm.at[0].set(jnp.zeros((5, 5)))
    return A, b, C, eta, Jm


def filter_smooth_par(eff, Cf0, xf0, Cb0, xb0):
    """As filter_smooth, from covariance-form priors (front Cf0/xf0, back Cb0/xb0), parallel in time."""
    E = eff["H"].shape[0]
    Jm, eta = jax.vmap(_hit_info)(eff)
    Q = eff["sv"][:, None, None] * jnp.einsum("eia,eja->eij", eff["D"], eff["D"])
    # forward: element k maps x_k -> x_{k+1}; output k = filtered state after effect k (after its noise)
    fe = _elements(Jm, eta, Q, Cf0, xf0, Jm[0], eta[0], Q[0])
    _, bF, CF, _, _ = jax.lax.associative_scan(_combine, fe)
    # predicted chi2: state before effect k = output k-1 (prior for k = 0)
    bpre = jnp.concatenate([xf0[None], bF[:-1]])
    Cpre = jnp.concatenate([Cf0[None], CF[:-1]])

    def chi(x, C, e):
        res = e["r"] + e["H"] @ (x - e["xref"])
        var = e["mvar"] + jnp.einsum("ki,ij,kj->k", e["H"], C, e["H"])
        act = e["w"] > 0
        return jnp.sum(jnp.where(act, res * res / jnp.where(act, var, 1.0), 0.0)), jnp.sum(act)
    c, n = jax.vmap(chi)(bpre, Cpre, eff)
    # backward on z_m = x_{E-m}: element m measures z_m (effect E-m; prior for m = 0), then noise Q_{E-m-1};
    # output m = predictive at z_{m+1} = x_{E-m-1}; the downstream information of effect k (on x_{k+1}) is output
    # E-k-2, and the back prior itself for k = E-1
    Jr, er, Qr = Jm[::-1], eta[::-1], Q[::-1]                      # index m -> effect E-1-m
    Jb = jnp.concatenate([jnp.zeros((1, 5, 5)), Jr[:-1]])          # element m measures effect E-m
    eb = jnp.concatenate([jnp.zeros((1, 5)), er[:-1]])
    be = _elements(Jb, eb, Qr, Cb0, xb0, jnp.zeros((5, 5)), jnp.zeros(5), Qr[0])
    _, bB, CB, _, _ = jax.lax.associative_scan(_combine, be)
    xdown = jnp.concatenate([bB[:E - 1][::-1], xb0[None]])
    Cdown = jnp.concatenate([CB[:E - 1][::-1], Cb0[None]])
    # output E-k-2 is the prediction after Q_{k+1}, before hit k+1: add hit k+1 (information form)
    WF = jnp.linalg.inv(CF)
    WP = jnp.linalg.inv(Cdown)
    Jn = jnp.concatenate([Jm[1:], jnp.zeros((1, 5, 5))])
    en = jnp.concatenate([eta[1:], jnp.zeros((1, 5))])
    W = WF + WP + Jn
    bvec = jnp.einsum("eij,ej->ei", WF, bF) + jnp.einsum("eij,ej->ei", WP, xdown) + en
    return {"W": W, "b": bvec, "chi2": jnp.sum(c), "ndof": jnp.sum(n) - 5}


def smoothed(fs, eff):
    """Parameters and covariance of each effect's piece, and the unbiased parameters of each hit (Hit::
    unbiasedParameters: the piece weight minus the hit's own weight)."""
    C = spd_inv(fs["W"])
    x = jnp.einsum("eij,ej->ei", C, fs["b"])
    dW, db = jax.vmap(_hit_info)(eff)
    Cu = spd_inv(fs["W"] - dW)
    xu = jnp.einsum("eij,ej->ei", Cu, fs["b"] - db)
    return x, C, xu, Cu




# ---------------------------------------------------------------------------------------------------------------------
# Part 3: the KinKal fit loop for one track (KinematicLineFit::produce -> Track(fit with CHSEEDFIT) -> goodFit ->
# extend with CHDRIFTEXT -> goodFit), vmapped over tracks.
# Effects in natural order: N hits then M crossings; each carries the parameters/covariance of the piece it was in
# after the last accepted iteration (its reference). A meta-iteration (stage) first updates the hit and crossing
# states from the current fit (Effect::updateState(first)), then iterates (Track::iterate + setStatus) until the
# status is not 'unconverged' or MaxNIter is reached.
from . import driftann  # noqa: E402
from .linefit_jax import _resid_one  # noqa: E402

MAXDVAR = 2.4 * 2.4 / 3.0              # DriftInfo::maxdvar_
# sequential lax.scan filter by default (measured faster on CPU and on an A100 slice); F1_SEQUENTIAL=0 selects
# the associative-scan version
PARALLEL = __import__("os").environ.get("F1_SEQUENTIAL", "1") != "1"
UNROLL = int(__import__("os").environ.get("F1_UNROLL", "8"))      # scan steps fused per loop iteration
CONVERGED, UNCONVERGED, LOWNDOF, GAPDIV, PARDIV, CHIDIV, FAILED = 0, 1, 2, 3, 4, 5, 7


def kl_params(x):
    """(a0, b0, a1, b1, t0) -> KinematicLine (d0, phi0, z0, theta, t0) at the POCA to the z axis (jnp)."""
    p, u = point_dir(x)
    uz = u[2]
    s1 = (p[2] * uz - p @ u) / jnp.maximum(1.0 - uz * uz, 1e-12)
    pca = p + u * s1
    sign = jnp.sign(-(pca[0] * u[1] - pca[1] * u[0]))
    sign = jnp.where(sign == 0, 1.0, sign)
    return jnp.array([sign * jnp.hypot(pca[0], pca[1]), jnp.arctan2(u[1], u[0]), pca[2], jnp.arccos(uz), x[4] + s1 / V])


def seed_covariance(x0, errors=(15.0, 0.05, 10.0, 0.02, 2.0)):
    """KINEMATICLINE SeedErrors (d0, phi0, z0, theta, t0; mom is not a fit parameter here) mapped to (a0,b0,a1,b1,t0)."""
    J = jax.jacfwd(kl_params)(x0)
    Ji = jnp.linalg.inv(J)
    return Ji @ jnp.diag(jnp.asarray(errors) ** 2) @ Ji.T


def hit_rows(xref_h, h, d):
    """Per hit at its own reference: residual rows r [N, 3], Jacobian H [N, 3, 5] with the drift radius frozen (no
    DriftDt: dDdT = 0), deltaT, live rDrift, DOCA and the track length s of the POCA to the wire."""
    def one(x, hi):
        r, dT, rd, doca = _resid_one(x, hi, d, None, True)
        rf = jax.lax.stop_gradient(rd)
        J = jax.jacfwd(lambda xx: _resid_one(xx, hi, d, rf, False)[0])(x)
        s = straw_poca(x, hi["mid"], hi["wdir"])[0]
        return r[0], J[0], dT[0], rd[0], doca[0], s[0]
    return jax.vmap(one)(xref_h, {k: v[:, None] for k, v in h.items()})


def hit_geometry(xu, Cu, h, d):
    """Per hit at its unbiased parameters (KKStrawHit::unbiasedClosestApproach): DOCA, DOCA variance, live rDrift,
    deltaT, position along the wire, POCA rho, wire-track direction cosine."""
    def one(x, C, hi):
        _, dT, rd, doca = _resid_one(x, hi, d, None, True)
        J = jax.jacfwd(lambda xx: _resid_one(xx, hi, d, None, True)[3])(x)[0]
        p, u = point_dir(x)
        c = hi["wdir"][0] @ u
        s2 = jnp.maximum(1.0 - c * c, 1e-12)
        dl = p - hi["mid"][0]
        du, dw = dl @ u, dl @ hi["wdir"][0]
        st, sw = (dw * c - du) / s2, (dw - du * c) / s2
        pp = p + u * st
        return doca[0], J @ C @ J, rd[0], dT[0], sw, jnp.hypot(pp[0], pp[1]), c, pp
    return jax.vmap(one)(xu, Cu, {k: v[:, None] for k, v in h.items()})


def update_states(st, xu, Cu, h, d, stage, ann):
    """Hit states for a meta-iteration, as KKStrawHit::updateWHS: CADSHU, then BkgANNSHU, then DriftANNSHU, on the
    unbiased closest approach. st: dict active, lr, frozen, nulldv [N]. Returns the new st and dvar [N]."""
    doca, dvar_, rd, dT, sw, rho, c, _ = hit_geometry(xu, Cu, h, d)
    valid = h["valid"]
    maxdoca, maxderr, minrd, maxrd = (stage["cad"][i] for i in range(4))
    # CADSHU (allowed Null:Inactive; updateable unless frozen by another updater)
    upd = valid & ~st["frozen"]
    inwin = (rd < maxrd) & (jnp.abs(doca) < maxdoca) & (dvar_ > 0) & (dvar_ < maxderr * maxderr)
    to_null = inwin & ~(rd > minrd)
    to_inact = ~inwin
    changed = upd & (to_null | to_inact)
    act = jnp.where(changed, to_null, st["active"])
    lr = jnp.where(changed, 0.0, st["lr"])
    nulldv = jnp.where(changed, stage["cad_nulldv"], st["nulldv"])
    frozen = jnp.where(changed, False, st["frozen"])
    # BkgANNSHU (selected network; inactive below the cut and frozen; an inactive hit above it becomes null)
    x6 = jnp.stack([jnp.abs(doca), rd, jnp.sqrt(jnp.maximum(dvar_, 0.0)), h["totdt"], jnp.abs(h["wdist"] - sw), rho], -1)
    qb = jnp.where(stage["bkg"] == 1, driftann.mlp(ann["bseed"], x6), driftann.mlp(ann["bfinal"], x6))
    use_b = valid & (stage["bkg"] > 0)
    bad = use_b & (qb < 0.2)
    react = use_b & ~bad & ~act
    act = jnp.where(bad, False, jnp.where(react, True, act))
    lr = jnp.where(bad | react, 0.0, lr)
    frozen = jnp.where(use_b, bad, frozen)
    # DriftANNSHU on active hits (stage 0 or stage 1 networks, per-stage cuts)
    sp, cp = driftann.inputs(doca, dvar_, rd, rd, h["totdt"], h["edep"], c)
    s0, c0 = driftann.mlp(ann["s0"], sp), driftann.mlp(ann["c0"], cp)
    s1, c1 = driftann.mlp(ann["s1"], sp), driftann.mlp(ann["c1"], cp)
    sq, cq = jnp.where(stage["ann"] == 0, s0, s1), jnp.where(stage["ann"] == 0, c0, c1)
    use_a = valid & act & (stage["ann"] >= 0)
    setlr = (cq > stage["ccut"]) & (sq > stage["scut"])
    lr = jnp.where(use_a, jnp.where(setlr, jnp.where(doca > 0, 1.0, -1.0), 0.0), lr)
    nulldv = jnp.where(use_a, True, nulldv)
    # dVar_ from the drift info at the unbiased deltaT
    sd = jnp.interp(dT, d["dt"], d["sderr"])
    ud = jnp.interp(dT, d["dt"], d["uderr"])
    rdp = jnp.maximum(rd, 0.0)
    dvar = jnp.where(lr != 0, sd * sd, jnp.where(nulldv, rdp * rdp + ud * ud, MAXDVAR))
    return {"active": act, "lr": lr, "frozen": frozen, "nulldv": nulldv}, dvar, {"doca": doca, "docavar": dvar_,
                                                                                 "rdrift": rd, "signq": sq, "clusq": cq}


def crossing_rows(xref_c, cref_c, crs, nsig, varscale):
    def one(x, C, mid, wdir):
        e = crossing_effects(x, C, mid[None], wdir[None], jnp.array([True]), nsig, varscale)
        return {k: v[0] for k, v in e.items()}
    return jax.vmap(one)(xref_c, cref_c, crs["mid"], crs["wdir"])


def _pos(x, t):
    p, u = point_dir(x)
    return p + u * V * (t - x[4])


def iterate_once(xe, Ce, h, crs, d, hs, xact, varscale, nsig, sxscale, prior, dwt):
    """One Track::iterate: rows at each effect's reference, effects between the first and last active hit, forward
    chi2, smoothed pieces. Returns new pieces (natural order) and the quantities setStatus needs."""
    N = h["valid"].shape[0]
    hh = dict(h, active=hs["active"], lr=hs["lr"])
    r, Hj, dT, rd, doca, s_h = hit_rows(xe[:N], hh, d)
    act = hs["active"] & h["valid"]
    mvar = jnp.stack([hs["dvar"], h["tvar"], h["uvar"]], -1)
    w = jnp.where(act[:, None], 1.0 / (mvar * varscale), 0.0)
    ce = crossing_rows(xe[N:], Ce[N:], crs, nsig, sxscale)
    big = 1e12
    smin, smax = jnp.min(jnp.where(act, s_h, big)), jnp.max(jnp.where(act, s_h, -big))
    inb = (ce["s"] > smin) & (ce["s"] < smax) & xact & crs["valid"]
    M = ce["s"].shape[0]
    s_all = jnp.concatenate([jnp.where(h["valid"], s_h, big), jnp.where(inb, ce["s"], big)])
    order = jnp.argsort(s_all)
    eff = {"xref": xe, "H": jnp.concatenate([Hj, jnp.zeros((M, 3, 5))]),
           "r": jnp.concatenate([r, jnp.zeros((M, 3))]), "w": jnp.concatenate([w, jnp.zeros((M, 3))]),
           "mvar": jnp.concatenate([mvar, jnp.ones((M, 3))]),
           "D": jnp.concatenate([jnp.zeros((N, 5, 2)), ce["D"]]),
           "sv": jnp.concatenate([jnp.zeros(N), jnp.where(inb, ce["scatvar"], 0.0)])}
    effs = {k: v[order] for k, v in eff.items()}
    dw = dwt / varscale
    if PARALLEL:
        fs = filter_smooth_par(effs, prior["Cf"] * dw, prior["xf"], prior["Cb"] * dw, prior["xb"])
    else:
        Wf, Wb = jnp.linalg.inv(prior["Cf"] * dw), jnp.linalg.inv(prior["Cb"] * dw)
        fs = filter_smooth(effs, Wf, Wf @ prior["xf"], Wb, Wb @ prior["xb"])
    x, C, xu, Cu = smoothed(fs, effs)
    # pieces: a new piece starts at each in-bounds crossing; gap = position jump at the crossing time
    isx = jnp.concatenate([jnp.zeros(N, bool), inb])[order]
    tx = effs["xref"][:, 4] + s_all[order] / V
    xprev = jnp.concatenate([x[:1], x[:-1]])
    gaps = jax.vmap(lambda a, b, t: jnp.linalg.norm(_pos(a, t) - _pos(b, t)))(x, xprev, tx)
    ngap = jnp.sum(isx)
    avggap = jnp.where(ngap > 0, jnp.sum(jnp.where(isx, gaps, 0.0)) / jnp.maximum(ngap, 1), 0.0)
    first = jnp.argmax(effs["w"].sum(-1) > 0)
    last = (effs["w"].shape[0] - 1) - jnp.argmax((effs["w"].sum(-1) > 0)[::-1])
    inv = jnp.argsort(order)
    return {"x": x[inv], "C": C[inv], "xu": xu[inv][:N], "Cu": Cu[inv][:N], "chi2": fs["chi2"], "ndof": fs["ndof"],
            "xfront": x[first], "Cfront": C[first], "xback": x[last], "Cback": C[last], "avggap": avggap,
            "s_hit": s_h, "inbounds": inb, "xing": ce, "nact": jnp.sum(act)}


def set_status(o, ref, prevcpn, it, cfg):
    """Track::setStatus: covariance sanity, avg gap, parameter change vs the previous fit at front and back, chi2/ndof
    change (only from the 2nd iteration of a meta-iteration), ndof, convergence."""
    bad = jnp.any(jnp.diagonal(o["Cfront"]) < 0) | jnp.any(jnp.diagonal(o["Cback"]) < 0) | \
        ~jnp.all(jnp.isfinite(o["x"]))
    dpf = o["xfront"] - ref["xf"]
    dpb = o["xback"] - ref["xb"]
    chif = dpf @ jnp.linalg.solve(ref["Cf"], dpf)
    chib = dpb @ jnp.linalg.solve(ref["Cb"], dpb)
    cpn = jnp.where(o["ndof"] > 0, o["chi2"] / jnp.maximum(o["ndof"], 1), -1.0)
    dchi = jnp.where(it > 0, cpn - prevcpn, cfg["conv"] + 1e-4)
    st = jnp.where(o["avggap"] > cfg["divgap"], GAPDIV,
         jnp.where((chif > cfg["pdchisq"]) | (chib > cfg["pdchisq"]), PARDIV,
         jnp.where(dchi > cfg["divchi"], CHIDIV,
         jnp.where(o["ndof"] < cfg["minndof"], LOWNDOF,
         jnp.where(jnp.abs(dchi) < cfg["conv"], CONVERGED, UNCONVERGED)))))
    st = jnp.where(o["nact"] == 0, LOWNDOF, st)
    return jnp.where(bad, FAILED, st), cpn


def in_detector(xe, s_hit, active, N):
    """KinematicLineFit::goodFit: every active hit's fit position inside rho < 900, |z| < 1800 mm."""
    def pos(x, s):
        p, u = point_dir(x)
        return p + u * s
    P = jax.vmap(pos)(xe[:N], s_hit)
    ok = (jnp.hypot(P[:, 0], P[:, 1]) < 900.0) & (jnp.abs(P[:, 2]) < 1800.0)
    return jnp.all(ok | ~active)


def fit_track(x0, h, crs, d, stages, ann, dwt=1e6, maxniter=10, fixed=None):
    """Full schedule for one track. h: padded hit dict with 'valid'; crs: mid, wdir, valid, hit (index or -1).
    stages: dict of per-stage arrays (see stock_stages). fixed: optional dict active, lr, nulldv [N] that replaces the
    updaters in every stage (fit-only comparisons on stock hit states). Returns final pieces, states, status per stage."""
    N, M = h["valid"].shape[0], crs["valid"].shape[0]
    C0 = seed_covariance(x0)
    xe = jnp.tile(x0, (N + M, 1))
    Ce = jnp.tile(C0, (N + M, 1, 1))
    st = {"active": h["valid"], "lr": jnp.zeros(N), "frozen"  # KKStrawHit starts null (active)
          : jnp.zeros(N, bool), "nulldv": jnp.zeros(N, bool)}
    carry0 = {"xe": xe, "Ce": Ce, "xu": jnp.tile(x0, (N, 1)), "Cu": jnp.tile(C0, (N, 1, 1)), "st": st,
              "dvar": jnp.full(N, MAXDVAR), "ref": {"xf": x0, "Cf": C0, "xb": x0, "Cb": C0},
              "status": jnp.array(UNCONVERGED), "stop": jnp.array(False), "good_seed": jnp.array(True),
              "chi2": jnp.array(0.0), "ndof": jnp.array(0), "cpn": jnp.array(-1.0), "niter": jnp.array(0),
              "s_hit": jnp.zeros(N), "xact": jnp.zeros(M, bool)}

    def stage_fn(c, sg):
        # goodFit gate before the extension (and the schedule stops after any unusable status)
        gate_fail = sg["gate"] & ~in_detector(c["xe"], c["s_hit"], c["st"]["active"], N)
        stop = c["stop"] | gate_fail
        good_seed = c["good_seed"] & ~(sg["gate"] & gate_fail)
        st2, dvar, _ = update_states(c["st"], c["xu"], c["Cu"], h, d, sg, ann)
        if fixed is not None:
            st2 = dict(st2, active=fixed["active"] & h["valid"], lr=fixed["lr"], nulldv=fixed["nulldv"])
            _, _, rdu, dTu, _, _, _, _ = hit_geometry(c["xu"], c["Cu"], h, d)
            sd, ud = jnp.interp(dTu, d["dt"], d["sderr"]), jnp.interp(dTu, d["dt"], d["uderr"])
            rdp = jnp.maximum(rdu, 0.0)
            dvar = jnp.where(st2["lr"] != 0, sd * sd, jnp.where(st2["nulldv"], rdp * rdp + ud * ud, MAXDVAR))
        # KKStrawXing::updateState(first): active_ = |DOCA at the reference| < SXU maxdoca, or its hit is active
        _, dref, _ = straw_poca_batch(c["xe"][N:], crs)
        hact = jnp.where(crs["hit"] >= 0, st2["active"][jnp.maximum(crs["hit"], 0)], False)
        xact = (jnp.abs(dref) < sg["sx_maxdoca"]) | hact
        vs = (1.0 + sg["T"]) ** 2
        sxs = jnp.where(sg["sx_scale"], vs, 1.0)
        hs = {"active": st2["active"], "lr": st2["lr"], "dvar": dvar}
        cfg = {k: sg[k] for k in ("conv", "divchi", "pdchisq", "divgap", "minndof")}

        def body(ic):
            it, cc = ic
            o = iterate_once(cc["xe"], cc["Ce"], h, crs, d, hs, xact, vs, sg["sx_nsig"], sxs, cc["ref"], dwt)
            status, cpn = set_status(o, cc["ref"], cc["cpn"], it, cfg)
            ok = status < LOWNDOF
            take = lambda a, b: jnp.where(ok, a, b)
            nref = {"xf": take(o["xfront"], cc["ref"]["xf"]), "Cf": take(o["Cfront"], cc["ref"]["Cf"]),
                    "xb": take(o["xback"], cc["ref"]["xb"]), "Cb": take(o["Cback"], cc["ref"]["Cb"])}
            cn = dict(cc, xe=take(o["x"], cc["xe"]), Ce=take(o["C"], cc["Ce"]), xu=take(o["xu"], cc["xu"]),
                      Cu=take(o["Cu"], cc["Cu"]), ref=nref, status=status, chi2=o["chi2"], ndof=o["ndof"], cpn=cpn,
                      niter=cc["niter"] + 1, s_hit=take(o["s_hit"], cc["s_hit"]))
            return it + 1, cn

        def cond(ic):
            it, cc = ic
            return (it == 0) | ((cc["status"] == UNCONVERGED) & (it < maxniter))      # Track::canIterate: iter_ < maxniter_

        cs = dict(c, st=st2, dvar=dvar, xact=xact)
        _, cn = jax.lax.while_loop(cond, body, (jnp.array(0), cs))
        cn["stop"] = stop | (cn["status"] >= LOWNDOF)
        cn["good_seed"] = good_seed
        keep = lambda a, b: jax.tree_util.tree_map(lambda u, v: jnp.where(stop, v, u), a, b)
        out = keep(cn, dict(c, stop=stop, good_seed=good_seed))
        return out, (out["status"], out["chi2"], out["ndof"], out["niter"])

    cf, hist = jax.lax.scan(stage_fn, carry0, stages)
    good = (cf["status"] < LOWNDOF) & ~cf["stop"] & in_detector(cf["xe"], cf["s_hit"], cf["st"]["active"], N)
    return cf, hist, good


def straw_poca_batch(xe_c, crs):
    return jax.vmap(lambda x, m, w: tuple(a[0] for a in straw_poca(x, m[None], w[None])))(xe_c, crs["mid"], crs["wdir"])


def load_anns():
    return {"bseed": driftann.load(driftann.find("TrainBkgSeed.dat")),
            "bfinal": driftann.load(driftann.find("TrainBkgFinal.dat")),
            "s0": driftann.load(driftann.find("TrainSign_Stage0.dat")),
            "c0": driftann.load(driftann.find("TrainCluster_Stage0.dat")),
            "s1": driftann.load(driftann.find("TrainSign_Stage1.dat")),
            "c1": driftann.load(driftann.find("TrainCluster_Stage1.dat"))}


def stock_stages(seed=True, ext=True):
    """CHSEEDFIT then CHDRIFTEXT (Offline/Mu2eKinKal/fcl/prolog.fcl) as per-stage arrays for fit_track."""
    rows = []
    seedcfg = dict(conv=0.5, divchi=20.0, pdchisq=1e8, divgap=20.0, minndof=1.0)
    extcfg = dict(conv=0.1, divchi=10.0, pdchisq=1e6, divgap=10.0, minndof=1.0)
    if seed:
        for T, cad in zip((5.0, 5.0, 2.0, 1.0), ((400.0, 800000.0, 50.0, 50.0), (40.0, 80.0, 5.0, 5.0),
                                                 (20.0, 40.0, 5.0, 4.0), (10.0, 20.0, 5.0, 3.0))):
            rows.append(dict(T=T, cad=cad, cad_nulldv=False, bkg=0, ann=-1, scut=0.0, ccut=0.0, sx_maxdoca=2.5,
                             sx_nsig=100.0, sx_scale=True, gate=False, **seedcfg))
    if ext:
        temps = (2.0, 1.0, 0.3, 0.0, 1.0, 0.5, 0.2, 0.0, 0.0)
        cads = ((10.0, 20.0, 5.0, 5.0), (10.0, 20.0, 5.0, 4.0), (8.0, 16.0, 5.0, 3.0), (8.0, 16.0, 5.0, 3.0),
                (6.0, 10.0, 5.0, 3.0), (5.0, 10.0, 5.0, 3.0), (5.0, 10.0, 5.0, 3.0), (5.0, 10.0, 5.0, 3.0),
                (5.0, 10.0, 5.0, 3.0))
        bkg = (0, 0, 0, 1, 0, 0, 0, 2, 0)
        ann = (-1, -1, -1, -1, 0, 1, 1, 1, 1)
        cuts = (0, 0, 0, 0, 0.5, 0.4, 0.3, 0.2, 0.1)
        nsig = (10.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        for i in range(9):
            rows.append(dict(T=temps[i], cad=cads[i], cad_nulldv=i >= 2, bkg=bkg[i], ann=ann[i], scut=cuts[i],
                             ccut=cuts[i], sx_maxdoca=2.5, sx_nsig=nsig[i], sx_scale=i < 3, gate=seed and i == 0,
                             **extcfg))
    return {k: jnp.asarray([r[k] for r in rows]) for k in rows[0]}
