"""F1: KinKal-like Kalman fit of the extracted (no field) KinematicLine, batched over tracks in JAX.

Parameters x = (a0, b0, a1, b1, t0, mom) (NP = 6): the F2 line through (a0, 0, b0), direction (a1, -1, b1), t0 at that
point, plus the momentum [MeV/c] as KinematicLine's 6th parameter (the speed in the time terms follows it). A straw
crossing kinks the direction (scattering) and lowers the momentum by its energy loss (Moyal mean, kkmaterial.straw_eloss),
as ElementXing::parameterChange; a ParameterHit constrains the momentum to the seed value (KINEMATICLINE
ParameterConstraints 1e-4 MeV/c).
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

NP = 6
V = MOM / (MOM ** 2 + MMU ** 2) ** 0.5 * C           # muon speed at the KKLine seed momentum, mm/ns


def speed(mom):
    return mom / jnp.sqrt(mom * mom + MMU * MMU) * C


def point_dir(x):
    """Reference point (a0, 0, b0) and unit direction of motion."""
    d = jnp.array([x[2], -1.0, x[3]])
    return jnp.array([x[0], 0.0, x[1]]), d / jnp.linalg.norm(d)


def params_from(p, u, t, mom):
    """Line through point p with unit direction u (u_y < 0), time t at p, momentum mom -> parameters."""
    s = -p[1] / u[1]                                   # to the y = 0 plane
    q = p + u * s
    return jnp.array([q[0], q[2], -u[0] / u[1], -u[2] / u[1], t + s / speed(mom), mom])


def transverse_basis(u):
    """Two unit vectors perpendicular to u (KinKal MomBasis perpdir/phidir analogue; the projected variance is
    isotropic, so only orthonormality matters)."""
    e1 = jnp.cross(u, jnp.array([0.0, 0.0, 1.0]))
    e1 = jnp.where(jnp.linalg.norm(e1) < 1e-9, jnp.array([1.0, 0.0, 0.0]), e1)
    e1 = e1 / jnp.linalg.norm(e1)
    return e1, jnp.cross(u, e1)


def kink(x, s, th, dmom=0.0):
    """Parameters after a kink th = (th1, th2) [rad] and a momentum change dmom at track length s (position and time
    of the crossing point unchanged)."""
    p0, u = point_dir(x)
    e1, e2 = transverse_basis(u)
    un = u + th[0] * e1 + th[1] * e2
    un = un / jnp.linalg.norm(un)
    return params_from(p0 + u * s, un, x[4] + s / speed(x[5]), x[5] + dmom)


def kink_jacobian(x, s):
    """d(params after kink)/d(th1, th2, dmom) at 0: [6, 3], closed form. The kink point q = p0 + u s stays; the
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
    v = speed(x[5])
    dt0 = -dt / v
    E3 = jnp.sqrt(x[5] ** 2 + MMU ** 2)
    dvdp = C * MMU ** 2 / E3 ** 3
    tcol = jnp.array([0.0, 0.0, 0.0, 0.0, t * dvdp / (v * v), 1.0])      # dmom: t0 = t_q - t / v(mom)
    ang = jnp.stack([da0, db0, da1, db1, dt0, jnp.zeros(2)], 0)
    return jnp.concatenate([ang, tcol[:, None]], 1)


def kink_jacobian_ad(x, s):
    """Reference for kink_jacobian (forward-mode autodiff of kink)."""
    return jax.jacfwd(lambda z: kink(x, s, z[:2], z[2]))(jnp.zeros(3))


def crossing_noise(x, s, scatvar, elvar=0.0):
    """Process noise covariance [6, 6] of one crossing: projected angle variance scatvar, momentum variance elvar."""
    D = kink_jacobian(x, s)
    return D @ jnp.diag(jnp.array([scatvar, scatvar, elvar])) @ D.T


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


def crossing_effects(x, cov, mid, wdir, valid, nsig, varscale, mom=None):
    """Per crossing: track length s (+ toff so it follows its hit), process noise Q [M, 5, 5] and the path/scatter
    values. DOCA variance from the parameter covariance (ClosestApproachData::docaVar); nsig and varscale from the
    StrawXingUpdater of the meta-iteration (scalevar: varscale = (1+T)^2, else 1)."""
    mom = x[5] if mom is None else mom
    st, doca, c = straw_poca(x, mid, wdir)
    J = jax.jacfwd(lambda xx: straw_poca(xx, mid, wdir)[1])(x)          # [M, NP]
    docavar = jnp.einsum("mi,ij,mj->m", J, cov, J)
    wall, gas = kkmaterial.path_lengths(doca, docavar, c, nsig)
    sv = kkmaterial.scatter_angle_var("straw-wall", mom, wall) + kkmaterial.scatter_angle_var("straw-gas", mom, gas,
                                                                                   scatterfrac=kkmaterial.GAS_SCATTERFRAC)
    sv = jnp.where(valid, sv * varscale * SCATSCALE, 0.0)
    dm, dmv = kkmaterial.straw_eloss(wall, gas, mom)
    dm = jnp.where(valid, dm * ELOSS, 0.0)
    dmv = jnp.where(valid, dmv * varscale * ELOSS, 0.0)
    D = jax.vmap(lambda s: kink_jacobian(x, s))(st)
    toff = kkmaterial.STRAW["wrad"]
    return {"s": st + toff, "D": D, "scatvar": sv, "dmom": dm, "dmomvar": dmv, "doca": doca, "docavar": docavar,
            "wall": wall, "gas": gas, "dirdot": c}


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



def _add_noise(W, b, D, sv, dmv=0.0, dmean=None):
    """(W, b) after the crossing: mean shift dmean [NP] (energy loss) and process noise Q = D diag(sv, sv, dmv) D^T
    (rank 3), as Material::process. Woodbury with S = I + Qd D^T W D (3x3, closed form)."""
    if dmean is not None:
        b = b + W @ dmean                                                  # x -> x + dmean
    qd = jnp.stack([sv, sv, dmv])
    WD = W @ D                                                             # [NP, 3]
    S = jnp.eye(3) + qd[:, None] * (D.T @ WD)
    Si = _inv3(S)
    K_ = Si @ (qd[:, None] * WD.T)                                         # [3, NP]
    return W - WD @ K_, b - WD @ (Si @ (qd * (D.T @ b)))


INV = __import__("os").environ.get("F1_INV", "lu")             # lu (jnp.linalg.inv) or chol


def gj_inv(M):
    """Unrolled Gauss-Jordan inverse of [..., n, n] matrices without pivoting (SPD inputs), elementwise jnp ops only,
    so XLA fuses it with the surrounding code instead of calling a batched LU kernel."""
    n = M.shape[-1]
    A = jnp.concatenate([M, jnp.broadcast_to(jnp.eye(n, dtype=M.dtype), M.shape)], -1)
    for k in range(n):
        piv = A[..., k:k + 1, :] / A[..., k:k + 1, k:k + 1]
        A = A - A[..., :, k:k + 1] * piv
        A = jnp.concatenate([A[..., :k, :], piv, A[..., k + 1:, :]], -2)
    return A[..., n:]


def spd_inv(M):
    """Inverse of symmetric positive-definite matrices [..., 5, 5]: LU (default), unrolled Gauss-Jordan (F1_INV=gj) or
    Cholesky (F1_INV=chol)."""
    if INV == "gj":
        return gj_inv(M)
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
        W, b = _add_noise(W + dW, b + db, e["D"], e["sv"], e["dmv"], e["D"][:, 2] * e["dm"])
        return (W, b), (pre, (W, b))

    def bwd(carry, e):
        W, b = carry
        out = (W, b)
        W, b = _add_noise(W, b, e["D"], e["sv"], e["dmv"], -e["D"][:, 2] * e["dm"])
        dW, db = _hit_info(e)
        return (W + dW, b + db), out

    _, ((Wp, bp), (Wf, bf)) = jax.lax.scan(fwd, (W0f, b0f), eff, unroll=UNROLL)
    (Wbf, bbf), (Wb, bb) = jax.lax.scan(bwd, (W0b, b0b), eff, reverse=True, unroll=UNROLL)
    c, n = jax.vmap(_predicted_chi2)(Wp, bp, eff)
    return {"W": Wf + Wb, "b": bf + bb, "chi2": jnp.sum(c), "ndof": jnp.sum(n) - NP, "Wfront": Wbf, "bfront": bbf}


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
    I = jnp.eye(NP)
    M = I + Ci @ Jj
    Xa = jnp.linalg.solve(M, jnp.concatenate([Ai, (bi + (Ci @ ej_[..., None])[..., 0])[..., None], Ci], -1))
    A = Aj @ Xa[..., :NP]
    b = (Aj @ Xa[..., NP:NP + 1])[..., 0] + bj
    C = _sym(Aj @ Xa[..., NP + 1:] @ jnp.swapaxes(Aj, -1, -2) + Cj)
    Nm = I + Jj @ Ci
    Ya = jnp.linalg.solve(Nm, jnp.concatenate([(ej_ - (Jj @ bi[..., None])[..., 0])[..., None], Jj], -1))
    AiT = jnp.swapaxes(Ai, -1, -2)
    eta = (AiT @ Ya[..., :1])[..., 0] + ei_
    J = _sym(AiT @ Ya[..., 1:] @ Ai + Ji)
    return A, b, C, eta, J


def _elements(Jm, eta, Q, P0, m0, J0, eta0, Q0):
    """Elements for effects 1..E-1 (measurement then noise) and element 0 (prior + measurement 0 + noise 0)."""
    E = Jm.shape[0]
    A = jnp.broadcast_to(jnp.eye(NP), (E, NP, NP))
    b = jnp.zeros((E, NP))
    Pp = jnp.linalg.inv(jnp.linalg.inv(P0) + J0)
    mp = Pp @ (jnp.linalg.solve(P0, m0) + eta0)
    A = A.at[0].set(jnp.zeros((NP, NP)))
    b = b.at[0].set(mp)
    C = Q.at[0].set(_sym(Pp + Q0))
    eta = eta.at[0].set(jnp.zeros(NP))
    Jm = Jm.at[0].set(jnp.zeros((NP, NP)))
    return A, b, C, eta, Jm


def filter_smooth_par(eff, Cf0, xf0, Cb0, xb0):
    """As filter_smooth, from covariance-form priors (front Cf0/xf0, back Cb0/xb0), parallel in time."""
    E = eff["H"].shape[0]
    Jm, eta = jax.vmap(_hit_info)(eff)
    qd = jnp.stack([eff["sv"], eff["sv"], eff["dmv"]], -1)
    Q = jnp.einsum("eia,ea,eja->eij", eff["D"], qd, eff["D"])            # mean shift not modelled in this variant
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
    Jb = jnp.concatenate([jnp.zeros((1, NP, NP)), Jr[:-1]])          # element m measures effect E-m
    eb = jnp.concatenate([jnp.zeros((1, NP)), er[:-1]])
    be = _elements(Jb, eb, Qr, Cb0, xb0, jnp.zeros((NP, NP)), jnp.zeros(NP), Qr[0])
    _, bB, CB, _, _ = jax.lax.associative_scan(_combine, be)
    xdown = jnp.concatenate([bB[:E - 1][::-1], xb0[None]])
    Cdown = jnp.concatenate([CB[:E - 1][::-1], Cb0[None]])
    # output E-k-2 is the prediction after Q_{k+1}, before hit k+1: add hit k+1 (information form)
    WF = spd_inv(CF)
    WP = spd_inv(Cdown)
    Jn = jnp.concatenate([Jm[1:], jnp.zeros((1, NP, NP))])
    en = jnp.concatenate([eta[1:], jnp.zeros((1, NP))])
    W = WF + WP + Jn
    bvec = jnp.einsum("eij,ej->ei", WF, bF) + jnp.einsum("eij,ej->ei", WP, xdown) + en
    return {"W": W, "b": bvec, "chi2": jnp.sum(c), "ndof": jnp.sum(n) - NP}


# Covariance-form filter with a crossing-only RTS smoother (default, F1_FILTER=cov). With no field the parameters are
# global, so the smoothed line is constant between crossings: piece p = effects after crossing p-1 up to crossing p.
# Forward: hits are 3-row updates (closed-form 3x3 inverse), crossings add Q (no inverse). The back prior enters as a
# pseudo-measurement of the last piece, which makes the result equal to KinKal's forward + backward combination.
# Backward: Rauch-Tung-Striebel over the crossing slots only. Unbiased hit parameters: 3x3 downdate of the piece.

def _inv3(S):
    """Closed-form inverse of [..., 3, 3] matrices (adjugate / determinant)."""
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 0, 2]
    d, e, f = S[..., 1, 0], S[..., 1, 1], S[..., 1, 2]
    g, h, i = S[..., 2, 0], S[..., 2, 1], S[..., 2, 2]
    A = e * i - f * h
    B = -(d * i - f * g)
    Cc = d * h - e * g
    det = a * A + b * B + c * Cc
    adj = jnp.stack([jnp.stack([A, -(b * i - c * h), b * f - c * e], -1),
                     jnp.stack([B, a * i - c * g, -(a * f - c * d)], -1),
                     jnp.stack([Cc, -(a * h - b * g), a * e - b * d], -1)], -2)
    return adj / det[..., None, None]


def _hit_update(x, C, e):
    """Gauss-Newton update of (x, C) by one hit's rows: x' = x - C A^T (I + A C A^T)^-1 sqrt(w) res(x), A = sqrt(w) H;
    exact for w = 0 rows (inactive)."""
    sw = jnp.sqrt(e["w"])
    A = sw[:, None] * e["H"]
    CA = C @ A.T                                         # [NP, 3]
    Sinv = _inv3(jnp.eye(3) + A @ CA)
    res = e["r"] + e["H"] @ (x - e["xref"])
    xn = x - CA @ (Sinv @ (sw * res))
    Cn = C - CA @ Sinv @ CA.T
    return xn, 0.5 * (Cn + Cn.T)


def filter_smooth_cov(eff, isx, Cf0, xf0, Cb0, xb0):
    """One algebraic iteration in covariance form. isx [E]: effect is a crossing slot (sorted order). Returns the
    piece of each effect (x, C), forward chi2/ndof, and the pre-crossing states needed for gaps."""
    E = isx.shape[0]

    def fwd(carry, e):
        x, C = carry
        res = e["r"] + e["H"] @ (x - e["xref"])
        var = e["mvar"] + jnp.einsum("ki,ij,kj->k", e["H"], C, e["H"])
        act = e["w"] > 0
        c = jnp.sum(jnp.where(act, res * res / jnp.where(act, var, 1.0), 0.0))
        xh, Ch = _hit_update(x, C, e)
        Qe = e["D"] @ jnp.diag(jnp.stack([e["sv"], e["sv"], e["dmv"]])) @ e["D"].T
        return (xh + e["D"][:, 2] * e["dm"], Ch + Qe), (x, C, c, jnp.sum(act))

    (xe_, Ce_), (xpre, Cpre, c, n) = jax.lax.scan(fwd, (xf0, Cf0), eff)
    # back prior as a pseudo-measurement of the last piece
    Wsum = spd_inv(Ce_) + spd_inv(Cb0)
    Cl = spd_inv(Wsum)
    xl = Cl @ (spd_inv(Ce_) @ xe_ + spd_inv(Cb0) @ xb0)
    return xpre, Cpre, jnp.sum(c), jnp.sum(n) - NP, xl, Cl


def _rts(xpre_x, Cpre_x, Q_x, xl, Cl, dmean_x=None):
    """Backward RTS over crossing slots (pre-crossing forward states, their Q and mean shift), from the last piece
    (xl, Cl). Returns pieces [M + 1]: piece j is before crossing j, piece M is the last."""
    if dmean_x is None:
        dmean_x = jnp.zeros_like(xpre_x)

    def bwd(carry, z):
        xs, Cs = carry
        xf, Cf, Q, dm = z
        Cp = Cf + Q
        G = jnp.linalg.solve(Cp, Cf).T                   # Cf Cp^-1 (symmetric)
        xn = xf + G @ (xs - xf - dm)
        Cn = Cf + G @ (Cs - Cp) @ G.T
        Cn = 0.5 * (Cn + Cn.T)
        return (xn, Cn), (xn, Cn)

    _, (xs, Cs) = jax.lax.scan(bwd, (xl, Cl), (xpre_x, Cpre_x, Q_x, dmean_x), reverse=True)
    return jnp.concatenate([xs, xl[None]]), jnp.concatenate([Cs, Cl[None]])


def unbiased_hits(x, C, H, r, xref, w):
    """Per hit: parameters without the hit's own rows, from its piece (x, C): C_u = C + C A^T (I - A C A^T)^-1 A C,
    x_u = x + C_u H^T w res(x)."""
    sw = jnp.sqrt(w)
    A = sw[..., :, None] * H
    CA = C @ jnp.swapaxes(A, -1, -2)
    Si = _inv3(jnp.eye(3) - A @ CA)
    Cu = C + CA @ Si @ jnp.swapaxes(CA, -1, -2)
    res = r + (H @ (x - xref)[..., None])[..., 0]
    xu = x + (Cu @ (jnp.swapaxes(H, -1, -2) @ (w * res)[..., None]))[..., 0]
    return xu, 0.5 * (Cu + jnp.swapaxes(Cu, -1, -2))


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
FILTER = __import__("os").environ.get("F1_FILTER", "cov")       # cov (RTS over crossings) or info (two-filter)
SCATSCALE = float(__import__("os").environ.get("F1_SCATSCALE", "1"))   # diagnostic only: scale on the scattering variance
FIXED_NULLDV_ALL = __import__("os").environ.get("F1_FIXED_NULLDV", "0") == "1"
NOHIT_IN_SEED = __import__("os").environ.get("F1_NOHIT_IN_SEED", "0") == "1"   # diagnostic: old behaviour
REF_AT_TIME = __import__("os").environ.get("F1_REFFRONT", "0") != "1"   # 1: old behaviour (old front piece)
KL_TEST = __import__("os").environ.get("F1_KLTEST", "1") == "1"   # parameter-change test in KinematicLine parameters
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
    return jnp.array([sign * jnp.hypot(pca[0], pca[1]), jnp.arctan2(u[1], u[0]), pca[2], jnp.arccos(uz),
                      x[4] + s1 / speed(x[5]), x[5]])


MOM_CONSTRAINT = float(__import__("os").environ.get("F1_MOMCONS", "1e-4"))   # KINEMATICLINE ParameterConstraints (mom)
ELOSS = float(__import__("os").environ.get("F1_ELOSS", "1"))                  # 0: no energy loss (diagnostic)


def seed_covariance(x0, errors=(15.0, 0.05, 10.0, 0.02, 2.0, 1.0)):
    """KINEMATICLINE SeedErrors (d0, phi0, z0, theta, t0, mom) mapped to (a0, b0, a1, b1, t0, mom)."""
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
    out = jax.vmap(one)(xref_c, cref_c, crs["mid"], crs["wdir"])
    if "kst_wall" in crs:        # diagnostic: scattering from the stock KalSeed path lengths
        sv = (kkmaterial.scatter_angle_var("straw-wall", MOM, crs["kst_wall"]) +
              kkmaterial.scatter_angle_var("straw-gas", MOM, crs["kst_gas"], scatterfrac=kkmaterial.GAS_SCATTERFRAC)) \
            * varscale * SCATSCALE
        out = dict(out, scatvar=sv)
    return out


def _pos(x, t):
    p, u = point_dir(x)
    return p + u * speed(x[5]) * (t - x[4])


def nearest_index(ranges, count, t):
    """PiecewiseTrajectory::nearestIndex on owned pieces 0..count: clamp at the ends, else the first piece whose
    range end is at or after t (earlier piece at an exact boundary)."""
    valid = jnp.arange(ranges.shape[0]) <= count
    idx = jnp.sum(valid[:, None] & (ranges[:, 1, None] < t[None]), axis=0)
    idx = jnp.minimum(idx, count)
    idx = jnp.where(t <= ranges[0, 0], 0, idx)
    return jnp.where(t >= ranges[count, 1], count, idx).astype(jnp.int32)


def reference_search_state(owner, mid, wdir, hint, max_moves=10):
    """Analytic search state with distinct pointer and solved-piece identities, not native CA arithmetic."""
    if max_moves < 0:
        raise ValueError("max_moves must be nonnegative")
    def nearest(t):
        idx = nearest_index(owner["ranges"], owner["count"], t)
        return jnp.where(t <= owner["ranges"][0, 0], 0, idx)

    index = nearest(hint)

    def solve(idx):
        xx = owner["P"][idx]
        s, _, c = straw_poca_batch(xx, {"mid": mid, "wdir": wdir})
        t = xx[:, 4] + s / speed(xx[:, 5])
        return t, jnp.isfinite(t) & (1 - c * c >= 1e-5)

    time, usable = solve(index)
    initial = (index, time, usable, jnp.ones_like(usable), jnp.zeros_like(usable), index, jnp.zeros_like(index))

    def step(attempt, carry):
        idx, t, ok, active, exhausted, solved, moves = carry
        target = nearest(t)
        changed = active & (target != idx)
        idx = jnp.where(active, target, idx)
        move = changed & ok
        exhausted = move & (attempt == max_moves)
        active = move & ~exhausted
        nt, nu = solve(idx)
        return (idx, jnp.where(active, nt, t), jnp.where(active, nu, ok), active, exhausted,
                jnp.where(active, idx, solved), moves + active.astype(moves.dtype))

    state = jax.lax.fori_loop(0, max_moves + 1, step, initial) if max_moves else initial
    index, time, usable, _, exhausted, solved, moves = state
    return {"index": index, "solution_index": solved, "time": time, "usable": usable,
            "exhausted": exhausted, "moves": moves, "unsupported": ~usable | exhausted}


def reference_search(owner, mid, wdir, hint, max_moves=10):
    """Compatibility view of the analytic search; no hint fallback, hit retry or native arithmetic."""
    state = reference_search_state(owner, mid, wdir, hint, max_moves)
    return state["index"], state["time"], state["unsupported"]


def seed_owner(x0, C0, t_begin, t_end, M):
    """Seed trajectory as an owned piecewise trajectory: one piece over the seed time range (KKFit::range)."""
    ranges = jnp.zeros((M + 1, 2)).at[0].set(jnp.stack([t_begin, t_end]))
    return {"P": jnp.tile(x0, (M + 1, 1)), "C": jnp.tile(C0, (M + 1, 1, 1)), "ranges": ranges,
            "count": jnp.array(0, jnp.int32)}


def _transit_time(ce, mom):
    """Straw transit time of each crossing (ElementXing::transitTime: crossing length / speed)."""
    orad, irad = kkmaterial.STRAW["orad"], kkmaterial.STRAW["irad"]
    length = 2 * jnp.sqrt(orad ** 2 - jnp.minimum(jnp.abs(ce["doca"]), irad) ** 2)
    return length / jnp.sqrt(jnp.maximum(1 - ce["dirdot"] ** 2, 0.01)) / speed(mom)


TIME_PARTS = ("cons", "prior", "bounds", "statusref", "seedx", "transit")


def _iterate_time(xe, Ce, h, crs, d, hs, xact, varscale, nsig, sxscale, dwt, tm, prior=None):
    """Track::iterate with the stock time semantics (time_model='stock'): effects sorted by their time on the
    reference (KKEFFComp), the momentum ParameterHit at its immutable seed time, fit bounds from the first and last
    active measurement including that constraint (Track::setBounds), priors from the old trajectory's pieces at the
    bounds (initFitState), candidate piece ranges as Material::append. Returns what iterate_once returns plus the
    owned candidate trajectory and the setStatus query times (candidate front/back piece range midpoints).
    tm["legacy"]: parts reverted to the legacy rule (leave-one-out arms, TIME_PARTS): cons (constraint at the hit-only
    seed midpoint in local s, lifted to time on the first hit's reference), prior (retained endpoint priors), bounds
    (diagnostic: crossing inclusion, prior and range bounds from the active hits only, constraint excluded), statusref (setStatus reference at the candidate bounds, not the end-piece
    midpoints), transit (no transit extension of the last range)."""
    N = h["valid"].shape[0]
    hh = dict(h, active=hs["active"], lr=hs["lr"])
    r, Hj, dT, rd, doca, s_h = hit_rows(xe[:N], hh, d)
    act = hs["active"] & h["valid"]
    mvar = jnp.stack([hs["dvar"], h["tvar"], h["uvar"]], -1)
    w = jnp.where(act[:, None], 1.0 / (mvar * varscale), 0.0)
    ce = crossing_rows(xe[N:], Ce[N:], crs, nsig, sxscale)
    M = ce["s"].shape[0]
    big = 1e12
    t_h = xe[:N, 4] + s_h / speed(xe[:N, 5])
    t_x = xe[N:, 4] + ce["s"] / speed(xe[N:, 5])
    if "xing_offset" in tm:
        t_x = xe[N:, 4] + (ce["s"] - kkmaterial.STRAW["wrad"]) / speed(xe[N:, 5]) + tm["xing_offset"]
    leg = tm.get("legacy", ())
    t_c = xe[0, 4] + hs["s_cons"] / speed(xe[0, 5]) if "cons" in leg else tm["t_cons"]
    cons = MOM_CONSTRAINT > 0
    cb = cons and "bounds" not in leg
    tmin = jnp.minimum(jnp.min(jnp.where(act, t_h, big)), jnp.where(cb, t_c, big))
    tmax = jnp.maximum(jnp.max(jnp.where(act, t_h, -big)), jnp.where(cb, t_c, -big))
    inb = (t_x > tmin) & (t_x < tmax) & xact & crs["valid"]
    owner = tm["owner"]
    pi = nearest_index(owner["ranges"], owner["count"], jnp.stack([tmin, tmax]))
    if "prior" not in leg:
        prior = {"xf": owner["P"][pi[0]], "Cf": owner["C"][pi[0]], "xb": owner["P"][pi[1]], "Cb": owner["C"][pi[1]]}
    eff = {"xref": xe, "H": jnp.concatenate([Hj, jnp.zeros((M, 3, NP))]),
           "r": jnp.concatenate([r, jnp.zeros((M, 3))]), "w": jnp.concatenate([w, jnp.zeros((M, 3))]),
           "mvar": jnp.concatenate([mvar, jnp.ones((M, 3))]),
           "D": jnp.concatenate([jnp.zeros((N, NP, 3)), ce["D"]]),
           "sv": jnp.concatenate([jnp.zeros(N), jnp.where(inb, ce["scatvar"], 0.0)]),
           "dm": jnp.concatenate([jnp.zeros(N), jnp.where(inb, ce["dmom"], 0.0)]),
           "dmv": jnp.concatenate([jnp.zeros(N), jnp.where(inb, ce["dmomvar"], 0.0)])}
    hc = jnp.zeros((1, 3, NP)).at[0, 0, 5].set(1.0)
    rc = jnp.zeros((1, 3)).at[0, 0].set(xe[0, 5] - hs["mom_seed"])
    mvc = jnp.ones((1, 3)).at[0, 0].set(MOM_CONSTRAINT ** 2)
    wc = jnp.zeros((1, 3)).at[0, 0].set(jnp.where(cons, 1.0 / (MOM_CONSTRAINT ** 2 * varscale), 0.0))
    eff = {"xref": jnp.concatenate([xe, xe[:1]]), "H": jnp.concatenate([eff["H"], hc]),
           "r": jnp.concatenate([eff["r"], rc]), "w": jnp.concatenate([eff["w"], wc]),
           "mvar": jnp.concatenate([eff["mvar"], mvc]), "D": jnp.concatenate([eff["D"], jnp.zeros((1, NP, 3))]),
           "sv": jnp.concatenate([eff["sv"], jnp.zeros(1)]), "dm": jnp.concatenate([eff["dm"], jnp.zeros(1)]),
           "dmv": jnp.concatenate([eff["dmv"], jnp.zeros(1)])}
    t_all = jnp.concatenate([jnp.where(h["valid"], t_h, big), jnp.where(inb, t_x, big), t_c[None]])
    order = jnp.argsort(t_all)
    effs = {k: v[order] for k, v in eff.items()}
    dw = dwt / varscale
    isx = jnp.concatenate([jnp.zeros(N, bool), inb, jnp.zeros(1, bool)])[order]
    E = N + M + 1
    xpre, Cpre, chi2, ndof, xl, Cl = filter_smooth_cov(effs, isx, prior["Cf"] * dw, prior["xf"], prior["Cb"] * dw,
                                                       prior["xb"])
    ar = jnp.arange(E)
    xpos = jnp.argsort(jnp.where(isx, ar, E + ar))[:M]
    vx = isx[xpos]
    Dx = effs["D"][xpos]
    qd = jnp.stack([effs["sv"][xpos], effs["sv"][xpos], effs["dmv"][xpos]], -1)
    Qx = jnp.where(vx[:, None, None], jnp.einsum("mia,ma,mja->mij", Dx, qd, Dx), 0.0)
    dmx = jnp.where(vx[:, None], Dx[:, :, 2] * effs["dm"][xpos][:, None], 0.0)
    Px, PC = _rts(xpre[xpos], Cpre[xpos], Qx, xl, Cl, dmx)
    # piece ownership in time: a crossing's boundary is its time (TOCA + wire radius / speed); the crossing itself
    # and every other effect belong to the piece holding their TOCA
    bnd = jnp.sort(jnp.where(inb, t_x, big))
    toff = tm.get("xing_offset", kkmaterial.STRAW["wrad"] / speed(xe[N:, 5]))
    t_true = jnp.concatenate([t_h, t_x - toff, t_c[None]])
    pid_nat = jnp.searchsorted(bnd, t_true, side="right")
    xs_, Cs_ = Px[pid_nat[order]], PC[pid_nat[order]]
    ngap = jnp.sum(vx)
    tx = jnp.where(vx, bnd, 0.0)
    gaps = jax.vmap(lambda a, b, t: jnp.linalg.norm(_pos(a, t) - _pos(b, t)))(Px[1:], Px[:-1], tx)
    avggap = jnp.where(ngap > 0, jnp.sum(jnp.where(vx, gaps, 0.0)) / jnp.maximum(ngap, 1), 0.0)
    # candidate ranges (Material::append): front [tmin, first boundary], ..., back [last boundary, end], the end
    # extended to every crossing's time + transit time
    transit = _transit_time(ce, xe[N:, 5])
    tend = tmax if "transit" in leg else jnp.maximum(tmax, jnp.max(jnp.where(inb, t_x + transit, -big)))
    count = ngap.astype(jnp.int32)
    starts = jnp.concatenate([tmin[None], bnd])
    ends = jnp.concatenate([bnd, tend[None]])
    ends = jnp.where(jnp.arange(M + 1) == count, tend, ends)
    ranges = jnp.stack([starts, ends], -1)
    query = jnp.stack([0.5 * (ranges[0, 0] + ranges[0, 1]), 0.5 * (ranges[count, 0] + ranges[count, 1])])
    if "statusref" in leg:
        query = jnp.stack([tmin, tmax])
    inv = jnp.argsort(order)
    x, C = xs_[inv][:N + M], Cs_[inv][:N + M]
    candidate = {"P": Px, "C": PC, "ranges": ranges, "count": count}
    ref_unsupported = jnp.zeros(N + M, bool)
    if tm.get("reference_model", "previous") != "previous":
        model = tm["reference_model"]
        mids = jnp.concatenate([h["mid"], crs["mid"]])
        dirs = jnp.concatenate([h["wdir"], crs["wdir"]])
        pi, _, ref_unsupported = reference_search(candidate, mids, dirs, t_true[:N + M],
                                                 max_moves=10 if model == "search" else 0)
        pid_nat = pid_nat.at[:N + M].set(pi)
        x, C = Px[pi], PC[pi]
    xu, Cu = unbiased_hits(x[:N], C[:N], Hj, r, xe[:N], w)
    trace = {}
    if tm.get("trace", False):
        trace = {"order": order, "ordering_time": t_all, "physical_time": jnp.concatenate([t_h, t_x, t_c[None]]),
                 "reference_query_time": t_true, "is_material": isx, **{"eff_" + k: v for k, v in effs.items()},
                 "bounds": jnp.stack([tmin, tmax]), "deweight": dw,
                 "prior_parameters": jnp.stack([prior["xf"], prior["xb"]]),
                 "prior_covariance": jnp.stack([prior["Cf"], prior["Cb"]]),
                 "crossing_offset": toff, "crossing_transit": transit,
                 "crossing_inbounds": inb,
                 **{"crossing_" + k: ce[k] for k in ("doca", "docavar", "wall", "gas", "dirdot")}}
    return {"x": x, "C": C, "xu": xu, "Cu": Cu, "chi2": chi2, "ndof": ndof, "s_front": tmin, "s_back": tmax,
            **({"trace": trace} if tm.get("trace", False) else {}),
            "reference_unsupported": ref_unsupported,
            "xfront": Px[0], "Cfront": PC[0], "xback": Px[count], "Cback": PC[count], "avggap": avggap,
            "s_hit": s_h, "inbounds": inb, "xing": ce, "nact": jnp.sum(act), "pid": pid_nat[:N + M], "bnd": bnd,
            "owner": {"P": Px, "C": PC, "ranges": ranges, "count": count}, "query": query}


def iterate_once(xe, Ce, h, crs, d, hs, xact, varscale, nsig, sxscale, prior, dwt, tm=None):
    """One Track::iterate: rows at each effect's reference, effects between the first and last active hit, forward
    chi2, smoothed pieces. Returns new pieces (natural order) and the quantities setStatus needs. tm (time model
    'stock'): see _iterate_time."""
    if tm is not None:
        if FILTER != "cov":
            raise ValueError("time_model='stock' requires F1_FILTER=cov")
        return _iterate_time(xe, Ce, h, crs, d, hs, xact, varscale, nsig, sxscale, dwt, tm, prior)
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
    eff = {"xref": xe, "H": jnp.concatenate([Hj, jnp.zeros((M, 3, NP))]),
           "r": jnp.concatenate([r, jnp.zeros((M, 3))]), "w": jnp.concatenate([w, jnp.zeros((M, 3))]),
           "mvar": jnp.concatenate([mvar, jnp.ones((M, 3))]),
           "D": jnp.concatenate([jnp.zeros((N, NP, 3)), ce["D"]]),
           "sv": jnp.concatenate([jnp.zeros(N), jnp.where(inb, ce["scatvar"], 0.0)]),
           "dm": jnp.concatenate([jnp.zeros(N), jnp.where(inb, ce["dmom"], 0.0)]),
           "dmv": jnp.concatenate([jnp.zeros(N), jnp.where(inb, ce["dmomvar"], 0.0)])}
    # momentum ParameterHit (KKFit::makeSeedParamHit at the seed time range mid): residual mom - seed mom, weight
    # 1 / (sigma^2 (1+T)^2) (ParameterHit::updateState), 1 DOF; linear, so its reference is irrelevant
    hc = jnp.zeros((1, 3, NP)).at[0, 0, 5].set(1.0)
    rc = jnp.zeros((1, 3)).at[0, 0].set(xe[0, 5] - hs["mom_seed"])
    mvc = jnp.ones((1, 3)).at[0, 0].set(MOM_CONSTRAINT ** 2)
    wc = jnp.zeros((1, 3)).at[0, 0].set(jnp.where(MOM_CONSTRAINT > 0, 1.0 / (MOM_CONSTRAINT ** 2 * varscale), 0.0))
    eff = {"xref": jnp.concatenate([xe, xe[:1]]), "H": jnp.concatenate([eff["H"], hc]),
           "r": jnp.concatenate([eff["r"], rc]), "w": jnp.concatenate([eff["w"], wc]),
           "mvar": jnp.concatenate([eff["mvar"], mvc]), "D": jnp.concatenate([eff["D"], jnp.zeros((1, NP, 3))]),
           "sv": jnp.concatenate([eff["sv"], jnp.zeros(1)]), "dm": jnp.concatenate([eff["dm"], jnp.zeros(1)]),
           "dmv": jnp.concatenate([eff["dmv"], jnp.zeros(1)])}
    s_all = jnp.concatenate([s_all, hs["s_cons"][None]])
    order = jnp.argsort(s_all)
    effs = {k: v[order] for k, v in eff.items()}
    dw = dwt / varscale
    isx = jnp.concatenate([jnp.zeros(N, bool), inb, jnp.zeros(1, bool)])[order]
    if FILTER == "cov":
        E = N + M + 1
        xpre, Cpre, chi2, ndof, xl, Cl = filter_smooth_cov(effs, isx, prior["Cf"] * dw, prior["xf"], prior["Cb"] * dw,
                                                           prior["xb"])
        ar = jnp.arange(E)
        xpos = jnp.argsort(jnp.where(isx, ar, E + ar))[:M]          # crossing slots in order, then padding
        vx = isx[xpos]
        Dx = effs["D"][xpos]
        qd = jnp.stack([effs["sv"][xpos], effs["sv"][xpos], effs["dmv"][xpos]], -1)
        Qx = jnp.where(vx[:, None, None], jnp.einsum("mia,ma,mja->mij", Dx, qd, Dx), 0.0)
        dmx = jnp.where(vx[:, None], Dx[:, :, 2] * effs["dm"][xpos][:, None], 0.0)
        # forward states before each crossing slot are taken AFTER that slot's own (empty) hit rows, i.e. xpre is the
        # state entering the crossing; the RTS prediction across it is x + dmean, C + Q
        Px, PC = _rts(xpre[xpos], Cpre[xpos], Qx, xl, Cl, dmx)       # pieces [M + 1]
        # piece of every effect by its position along the track (KinKal PCA on the piecewise trajectory): the number
        # of in-bounds crossing boundaries (TOCA + wire radius) at or before it; a crossing sits at its TOCA, i.e. on
        # the piece before its own boundary; crossings outside the bounds or inactive get their local piece too
        toff = kkmaterial.STRAW["wrad"]
        bnd = jnp.sort(jnp.where(inb, ce["s"], big))
        s_true = jnp.concatenate([s_h, ce["s"] - toff, hs["s_cons"][None]])
        pid_nat = jnp.searchsorted(bnd, s_true, side="right")
        pidx = pid_nat[order]
        xs_, Cs_ = Px[pidx], PC[pidx]
        tx = effs["xref"][xpos, 4] + s_all[order][xpos] / speed(effs["xref"][xpos, 5])
        gaps = jax.vmap(lambda a, b, t: jnp.linalg.norm(_pos(a, t) - _pos(b, t)))(Px[1:], Px[:-1], tx)
        ngap = jnp.sum(vx)
        avggap = jnp.where(ngap > 0, jnp.sum(jnp.where(vx, gaps, 0.0)) / jnp.maximum(ngap, 1), 0.0)
        inv = jnp.argsort(order)
        x, C = xs_[inv][:N + M], Cs_[inv][:N + M]
        pid_nat = pid_nat[:N + M]
        xu, Cu = unbiased_hits(x[:N], C[:N], Hj, r, xe[:N], w)
        acts = effs["w"].sum(-1) > 0
        first = jnp.argmax(acts)
        last = (E - 1) - jnp.argmax(acts[::-1])
        s_sorted = s_all[order]
        return {"x": x, "C": C, "xu": xu, "Cu": Cu, "chi2": chi2, "ndof": ndof, "s_front": s_sorted[first],
                "s_back": s_sorted[last],
                "xfront": xs_[first], "Cfront": Cs_[first], "xback": xs_[last], "Cback": Cs_[last], "avggap": avggap,
                "s_hit": s_h, "inbounds": inb, "xing": ce, "nact": jnp.sum(act), "pid": pid_nat, "bnd": bnd}
    if PARALLEL:
        fs = filter_smooth_par(effs, prior["Cf"] * dw, prior["xf"], prior["Cb"] * dw, prior["xb"])
    else:
        Wf, Wb = spd_inv(prior["Cf"] * dw), spd_inv(prior["Cb"] * dw)
        fs = filter_smooth(effs, Wf, Wf @ prior["xf"], Wb, Wb @ prior["xb"])
    x, C, xu, Cu = smoothed(fs, effs)
    # pieces: a new piece starts at each in-bounds crossing; gap = position jump at the crossing time
    tx = effs["xref"][:, 4] + s_all[order] / speed(effs["xref"][:, 5])
    xprev = jnp.concatenate([x[:1], x[:-1]])
    gaps = jax.vmap(lambda a, b, t: jnp.linalg.norm(_pos(a, t) - _pos(b, t)))(x, xprev, tx)
    ngap = jnp.sum(isx)
    avggap = jnp.where(ngap > 0, jnp.sum(jnp.where(isx, gaps, 0.0)) / jnp.maximum(ngap, 1), 0.0)
    first = jnp.argmax(effs["w"].sum(-1) > 0)
    last = (effs["w"].shape[0] - 1) - jnp.argmax((effs["w"].sum(-1) > 0)[::-1])
    inv = jnp.argsort(order)
    pid = jnp.cumsum(isx)[inv][:N + M]
    bnd = jnp.sort(jnp.where(inb, ce["s"], big))
    return {"s_front": s_all[order][first], "s_back": s_all[order][last],
            "x": x[inv][:N + M], "C": C[inv][:N + M], "xu": xu[inv][:N], "Cu": Cu[inv][:N], "chi2": fs["chi2"], "ndof": fs["ndof"],
            "xfront": x[first], "Cfront": C[first], "xback": x[last], "Cback": C[last], "avggap": avggap,
            "s_hit": s_h, "inbounds": inb, "xing": ce, "nact": jnp.sum(act), "pid": pid, "bnd": bnd}


def piece_at(xe, Ce, pid, bnd, s):
    """Parameters and covariance of the trajectory piece holding track length s (PiecewiseTrajectory::nearestPiece):
    piece index = number of crossing boundaries at or before s; any effect of that piece carries its parameters."""
    p = jnp.searchsorted(bnd, s, side="right")
    cand = pid == p
    e = jnp.where(jnp.any(cand), jnp.argmax(cand), jnp.argmin(jnp.abs(pid - p)))
    return xe[e], Ce[e]


def set_status(o, ref, prevcpn, it, cfg):
    """Track::setStatus: covariance sanity, avg gap, parameter change vs the previous fit at front and back, chi2/ndof
    change (only from the 2nd iteration of a meta-iteration), ndof, convergence."""
    bad = jnp.any(jnp.diagonal(o["Cfront"]) < 0) | jnp.any(jnp.diagonal(o["Cback"]) < 0) | \
        ~jnp.all(jnp.isfinite(o["x"]))
    if KL_TEST:
        # KinKal evaluates the parameter change in the KinematicLine parameters (d0, phi0, z0, theta, t0, mom); the
        # map from ours is non-linear (d0 and phi0 at the POCA to the z axis), and the reference covariance of nearly
        # vertical cosmics is close to singular, so the test is evaluated there, not in (a0, b0, a1, b1, t0, mom)
        def kl_chi2(xn, xr, Cr):
            Jr = jax.jacfwd(kl_params)(xr)
            dk = kl_params(xn) - kl_params(xr)
            dk = dk.at[1].set((dk[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
            return dk @ jnp.linalg.solve(Jr @ Cr @ Jr.T, dk)
        chif = kl_chi2(o["xfront"], ref["xf"], ref["Cf"])
        chib = kl_chi2(o["xback"], ref["xb"], ref["Cb"])
    else:
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
    set_status.last = (chif, chib)
    set_status.momterm = ((o["xfront"][5] - ref["xf"][5]) ** 2 / ref["Cf"][5, 5], jnp.sqrt(ref["Cf"][5, 5]))
    return jnp.where(bad, FAILED, st), cpn


def in_detector(xe, s_hit, active, N):
    """KinematicLineFit::goodFit: every active hit's fit position inside rho < 900, |z| < 1800 mm."""
    def pos(x, s):
        p, u = point_dir(x)
        return p + u * s
    P = jax.vmap(pos)(xe[:N], s_hit)
    ok = (jnp.hypot(P[:, 0], P[:, 1]) < 900.0) & (jnp.abs(P[:, 2]) < 1800.0)
    return jnp.all(ok | ~active)


def construct_crossings(owner, xe, crs, offsets, constructed, unsupported, enabled):
    """Freeze diagnostic extension offsets once, retaining unsupported construction searches."""
    hint_s = straw_poca_batch(xe, crs)[0]
    hint_t = xe[:, 4] + hint_s / speed(xe[:, 5])
    pi, _, failed = reference_search(owner, crs["mid"], crs["wdir"], hint_t)
    take = enabled & crs["valid"] & ~constructed
    value = kkmaterial.STRAW["wrad"] / speed(owner["P"][pi, 5])
    return (jnp.where(take, value, offsets), constructed | take, unsupported | (take & failed))


def fit_track(x0, h, crs, d, stages, ann, dwt=1e6, maxniter=10, fixed=None, time_model="legacy",
              xing_activity="hit_or_doca", time_legacy=(), constraint_time=None, reference_model="previous",
              crossing_time="current", seed_input=None, trace=False):
    """Full schedule for one track. h: padded hit dict with 'valid'; crs: mid, wdir, valid, hit (index or -1).
    stages: dict of per-stage arrays (see stock_stages). fixed: optional dict active, lr, nulldv [N] that replaces the
    updaters in every stage (fit-only comparisons on stock hit states). Returns final pieces, states, status per stage.
    time_model: 'legacy' (effects ordered by track length, constraint at the seed-hit midpoint position, retained
    endpoint priors, reference pieces at the endpoint track lengths) or 'stock' (KinKal time semantics, see
    _iterate_time; the constraint at the immutable seed time, the midpoint of the seed hit and crossing times, and
    the setStatus reference at the candidate end-piece range midpoints on the owned old trajectory).
    xing_activity: 'hit_or_doca' (p109 KKStrawXing::active: |DOCA| < maxdoca at the meta-iteration start, or its hit
    active) or 'doca' (diagnostic: the DOCA test alone). time_legacy: parts of the stock time model reverted to the
    legacy rule (TIME_PARTS; 'seedx': seed-fit crossings of extension-added hits kept), for leave-one-out arms.
    constraint_time: optional certified native ParameterHit time; changes only this immutable scalar in the stock
    time model, not the reconstructed seed state or owner range. Full stock-exactness is not implied.
    reference_model: previous (unchanged), boundary (earlier-piece hint selection), or search (analytic piece
    recomputation, at most ten moves); the last two are isolated stock-time diagnostics, not native-exact fits.
    trace: default-off single-track operand history in cf['trace']; mask records by valid. The dedicated
    single-target driver enforces the memory bound; vmapped capture requires an outer batch-size guard."""
    if crossing_time not in ("current", "constructed"):
        raise ValueError("unknown crossing_time: " + str(crossing_time))
    if crossing_time != "current" and time_model != "stock":
        raise ValueError("constructed crossing offsets require stock time model")
    if seed_input is not None and (time_model != "stock" or constraint_time is None or "cons" in time_legacy):
        raise ValueError("native seed requires stock time model and explicit constraint time")
    if reference_model not in ("previous", "boundary", "search"):
        raise ValueError("unknown reference_model: " + str(reference_model))
    if reference_model != "previous" and time_model != "stock":
        raise ValueError("reference diagnostics require stock time model")
    if time_model not in ("legacy", "stock"):
        raise ValueError("unknown time_model: " + str(time_model))
    if xing_activity not in ("hit_or_doca", "doca"):
        raise ValueError("unknown xing_activity: " + str(xing_activity))
    if set(time_legacy) - set(TIME_PARTS) or (time_legacy and time_model != "stock"):
        raise ValueError("time_legacy needs time_model='stock' and parts from " + ",".join(TIME_PARTS))
    time_legacy = tuple(time_legacy)
    if constraint_time is not None and (time_model != "stock" or "cons" in time_legacy):
        raise ValueError("native constraint time requires stock time model without cons reversion")
    if trace and (time_model != "stock" or FILTER != "cov"):
        raise ValueError("iteration trace requires stock time model and covariance filter")
    hit_or = xing_activity == "hit_or_doca"
    stock_time = time_model == "stock"
    N, M = h["valid"].shape[0], crs["valid"].shape[0]
    if x0.shape[0] == 5:
        x0 = jnp.concatenate([x0, jnp.array([MOM])])
    if seed_input is not None:
        x0 = seed_input["parameters"]
    C0 = seed_covariance(x0) if seed_input is None else seed_input["covariance"]
    s_seed = straw_poca(x0, h["mid"], h["wdir"])[0]
    sv_ = h["valid"] & (h["inseed"] if "inseed" in h else True)
    s_cons = 0.5 * (jnp.min(jnp.where(sv_, s_seed, 1e12)) + jnp.max(jnp.where(sv_, s_seed, -1e12)))
    xe = jnp.tile(x0, (N + M, 1))
    Ce = jnp.tile(C0, (N + M, 1, 1))
    st = {"active": h["valid"], "lr": jnp.zeros(N), "frozen"  # KKStrawHit starts null (active)
          : jnp.zeros(N, bool), "nulldv": jnp.zeros(N, bool)}
    carry0 = {"xe": xe, "Ce": Ce, "xu": jnp.tile(x0, (N, 1)), "Cu": jnp.tile(C0, (N, 1, 1)), "st": st,
              "dvar": jnp.full(N, MAXDVAR), "ref": {"xf": x0, "Cf": C0, "xb": x0, "Cb": C0},
              "status": jnp.array(UNCONVERGED), "stop": jnp.array(False), "good_seed": jnp.array(True),
              "chi2": jnp.array(0.0), "ndof": jnp.array(0), "cpn": jnp.array(-1.0), "niter": jnp.array(0),
              "s_hit": jnp.zeros(N), "xact": jnp.zeros(M, bool), "pid": jnp.zeros(N + M, jnp.int32),
              "bnd": jnp.full(M, 1e12), "dpmax": jnp.array(0.0), "dmomf": jnp.array(0.0),
              "smref": jnp.array(0.0), "mterm": jnp.array(0.0),
              "mhist": jnp.zeros(16), "shist": jnp.full(16, -1),
              "reference_unsupported": jnp.zeros(N + M, bool)}
    if stock_time:
        # KKFit::range: extrema of the initial seed hit and crossing times on the seed; ParameterHit at its mid
        v0 = speed(x0[5])
        t_sh = jnp.where(sv_, x0[4] + s_seed / v0, jnp.nan)
        s_sx = straw_poca(x0, crs["mid"], crs["wdir"])[0] + kkmaterial.STRAW["wrad"]
        sx_ = crs["valid"] & (crs["hit"] >= 0) & sv_[jnp.maximum(crs["hit"], 0)]
        t_sx = jnp.where(sx_, x0[4] + s_sx / v0, jnp.nan)
        t_seed = jnp.concatenate([t_sh, t_sx])
        tb, te = jnp.nanmin(t_seed), jnp.nanmax(t_seed)
        t_cons = 0.5 * (tb + te) if constraint_time is None else constraint_time
        if seed_input is not None:
            tb, te = seed_input["range"][0], seed_input["range"][1]
        carry0["owner"] = seed_owner(x0, C0, tb, te, M)
        if crossing_time == "constructed":
            carry0["xing_offset"] = jnp.full(M, kkmaterial.STRAW["wrad"] / v0)
            carry0["xing_constructed"] = sx_
            carry0["construction_unsupported"] = jnp.zeros(M, bool)

    if trace:
        from . import fittrace
        trace_empty = fittrace.empty(N, M, stages["T"].shape[0], maxniter)
        carry0["trace_buffer"] = trace_empty

    def stage_fn(c, sg):
        # goodFit gate before the extension (and the schedule stops after any unusable status)
        gate_fail = sg["gate"] & ~in_detector(c["xe"], c["s_hit"], c["st"]["active"], N)
        stop = c["stop"] | gate_fail
        good_seed = c["good_seed"] & ~(sg["gate"] & gate_fail)
        if crossing_time == "constructed":
            offsets, constructed, unsupported = construct_crossings(
                c["owner"], c["xe"][N:], crs, c["xing_offset"], c["xing_constructed"],
                c["construction_unsupported"], sg["gate"] & ~stop)
            c = dict(c, xing_offset=offsets, xing_constructed=constructed, construction_unsupported=unsupported)
        st2, dvar, _ = update_states(c["st"], c["xu"], c["Cu"], h, d, sg, ann)
        if fixed is not None:
            # null hits take the variance of the stage's flag (NullDriftVar only from the 3rd CHDRIFTEXT stage), unless
            # F1_FIXED_NULLDV=1 (NullDriftVar in every stage, the earlier diagnostic)
            ndv = fixed["nulldv"] & (sg["cad_nulldv"] | FIXED_NULLDV_ALL)
            st2 = dict(st2, active=fixed["active"] & h["valid"], lr=fixed["lr"], nulldv=ndv)
            _, _, rdu, dTu, _, _, _, _ = hit_geometry(c["xu"], c["Cu"], h, d)
            sd, ud = jnp.interp(dTu, d["dt"], d["sderr"]), jnp.interp(dTu, d["dt"], d["uderr"])
            rdp = jnp.maximum(rdu, 0.0)
            dvar = jnp.where(st2["lr"] != 0, sd * sd, jnp.where(st2["nulldv"], rdp * rdp + ud * ud, MAXDVAR))
        # hits added by the extension (KKFit::addStrawHits) do not exist in the seed-fit stages
        if "inseed" in h:
            st2 = dict(st2, active=st2["active"] & (h["inseed"] | ~sg["seedfit"]),
                       lr=jnp.where(h["inseed"] | ~sg["seedfit"], st2["lr"], 0.0))
        # p109 KKStrawXing: updateState(first) sets active_ = |DOCA at the reference| < SXU maxdoca; active() is
        # active_ or its hit active
        _, dref, _ = straw_poca_batch(c["xe"][N:], crs)
        hact = jnp.where(crs["hit"] >= 0, st2["active"][jnp.maximum(crs["hit"], 0)], False)
        xact = (jnp.abs(dref) < sg["sx_maxdoca"]) | (hact & hit_or)
        # crossings without a hit exist only from the extension on (KKFit::addStraws); the seed fit has the
        # crossings made with the straw hits (KKFit::makeStrawHits)
        xact = xact & ((crs["hit"] >= 0) | ~sg["seedfit"] | NOHIT_IN_SEED)
        if stock_time and "inseed" in h and "seedx" not in time_legacy:
            # nor crossings of hits the extension adds (KKFit::addStrawHits makes those with their hits)
            xact = xact & (sx_ | ~sg["seedfit"] | NOHIT_IN_SEED)
        if "stock_active" in crs:
            xact = crs["stock_active"]
        vs = (1.0 + sg["T"]) ** 2
        sxs = jnp.where(sg["sx_scale"], vs, 1.0)
        hs = {"active": st2["active"], "lr": st2["lr"], "dvar": dvar, "s_cons": s_cons, "mom_seed": x0[5]}
        cfg = {k: sg[k] for k in ("conv", "divchi", "pdchisq", "divgap", "minndof")}

        def body(ic):
            it, cc = ic
            tm = {"t_cons": t_cons, "owner": cc["owner"], "legacy": time_legacy,
                  "reference_model": reference_model, "trace": trace} if stock_time else None
            if crossing_time == "constructed":
                tm["xing_offset"] = cc["xing_offset"]
            o = iterate_once(cc["xe"], cc["Ce"], h, crs, d, hs, xact, vs, sg["sx_nsig"], sxs, cc["ref"], dwt, tm=tm)
            if stock_time:
                # Track::setStatus: old pieces nearest the candidate front/back piece range midpoints
                ow = cc["owner"]
                ri = nearest_index(ow["ranges"], ow["count"], o["query"])
                sref = {"xf": ow["P"][ri[0]], "Cf": ow["C"][ri[0]], "xb": ow["P"][ri[1]], "Cb": ow["C"][ri[1]]}
            else:
                # Track::setStatus compares the new front/back pieces with the OLD trajectory's piece at their times
                xfo, Cfo = piece_at(cc["xe"], cc["Ce"], cc["pid"], cc["bnd"], o["s_front"])
                xbo, Cbo = piece_at(cc["xe"], cc["Ce"], cc["pid"], cc["bnd"], o["s_back"])
                sref = {"xf": xfo, "Cf": Cfo, "xb": xbo, "Cb": Cbo} if REF_AT_TIME else cc["ref"]
            status, cpn = set_status(o, sref, cc["cpn"], it, cfg)
            chif, chib = set_status.last
            mterm, smref = set_status.momterm
            ok = status < LOWNDOF
            take = lambda a, b: jnp.where(ok, a, b)
            nref = {"xf": take(o["xfront"], cc["ref"]["xf"]), "Cf": take(o["Cfront"], cc["ref"]["Cf"]),
                    "xb": take(o["xback"], cc["ref"]["xb"]), "Cb": take(o["Cback"], cc["ref"]["Cb"])}
            cn = dict(cc, xe=take(o["x"], cc["xe"]), Ce=take(o["C"], cc["Ce"]), xu=take(o["xu"], cc["xu"]),
                      Cu=take(o["Cu"], cc["Cu"]), ref=nref, status=status, chi2=o["chi2"], ndof=o["ndof"], cpn=cpn,
                      niter=cc["niter"] + 1, s_hit=take(o["s_hit"], cc["s_hit"]),
                      pid=take(o["pid"].astype(jnp.int32), cc["pid"]), bnd=take(o["bnd"], cc["bnd"]),
                      dpmax=jnp.maximum(cc["dpmax"], jnp.maximum(chif, chib)),
                      dmomf=jnp.where(jnp.maximum(chif, chib) >= cc["dpmax"], o["xfront"][5] - sref["xf"][5], cc["dmomf"]),
                      smref=jnp.where(jnp.maximum(chif, chib) >= cc["dpmax"], smref, cc["smref"]),
                      mterm=jnp.where(jnp.maximum(chif, chib) >= cc["dpmax"], mterm, cc["mterm"]),
                      mhist=cc["mhist"].at[jnp.minimum(it, 15)].set(o["xfront"][5]),
                      shist=cc["shist"].at[jnp.minimum(it, 15)].set(status))
            if stock_time:
                cn["owner"] = jax.tree_util.tree_map(take, o["owner"], cc["owner"])
                valid_effect = jnp.concatenate([h["valid"], crs["valid"]])
                cn["reference_unsupported"] = cc["reference_unsupported"] | (ok & o["reference_unsupported"] & valid_effect)
            if trace:
                record = dict(o["trace"], valid=~stop, iteration=it, status=status, swapped=ok & ~stop,
                    chi2=o["chi2"], ndof=o["ndof"], previous_cpn=cc["cpn"], cpn=cpn,
                    endpoint_chi2=jnp.stack([chif, chib]), avggap=o["avggap"],
                    reference_parameters=cc["xe"], reference_covariance=cc["Ce"], reference_index=cc["pid"],
                    owner_parameters=ow["P"], owner_covariance=ow["C"], owner_ranges=ow["ranges"], owner_count=ow["count"],
                    candidate_parameters=o["owner"]["P"], candidate_covariance=o["owner"]["C"],
                    candidate_ranges=o["owner"]["ranges"], candidate_count=o["owner"]["count"],
                    next_reference_index=o["pid"], query=o["query"], status_reference_index=ri,
                    status_reference_parameters=jnp.stack([sref["xf"], sref["xb"]]),
                    status_reference_covariance=jnp.stack([sref["Cf"], sref["Cb"]]),
                    hit_active=hs["active"], hit_lr=hs["lr"], crossing_active=xact)
                if set(record) != set(cc["trace_buffer"]):
                    raise ValueError("iteration trace schema mismatch")
                cn["trace_buffer"] = {k: v.at[it].set(record[k].astype(v.dtype)) for k, v in cc["trace_buffer"].items()}
            return it + 1, cn

        def cond(ic):
            it, cc = ic
            return (it == 0) | ((cc["status"] == UNCONVERGED) & (it < maxniter))      # Track::canIterate: iter_ < maxniter_

        cs = dict(c, st=st2, dvar=dvar, xact=xact, dpmax=jnp.array(0.0), dmomf=jnp.array(0.0), smref=jnp.array(0.0),
                  mterm=jnp.array(0.0), mhist=jnp.zeros(16), shist=jnp.full(16, -1))
        if trace:
            cs["trace_buffer"] = trace_empty
        _, cn = jax.lax.while_loop(cond, body, (jnp.array(0), cs))
        cn["stop"] = stop | (cn["status"] >= LOWNDOF)
        cn["good_seed"] = good_seed
        keep = lambda a, b: jax.tree_util.tree_map(lambda u, v: jnp.where(stop, v, u), a, b)
        out = keep(cn, dict(c, stop=stop, good_seed=good_seed))
        history = (out["status"], out["chi2"], out["ndof"], out["niter"], cn["dpmax"], cn["dmomf"],
                   out["st"]["active"], out["st"]["lr"], cn["smref"], cn["mterm"],
                   jnp.sqrt(out["ref"]["Cf"][5, 5]), out["ref"]["xf"][5] - out["ref"]["xb"][5], cn["mhist"], cn["shist"])
        return out, (history, cn["trace_buffer"]) if trace else history

    cf, hist = jax.lax.scan(stage_fn, carry0, stages)
    if trace:
        hist, records = hist
        cf.pop("trace_buffer")
        cf["trace"] = records
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
            rows.append(dict(T=T, cad=cad, cad_nulldv=False, bkg=0, ann=-1, scut=0.0, ccut=0.0, sx_maxdoca=2.5, seedfit=True,
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
            rows.append(dict(T=temps[i], cad=cads[i], cad_nulldv=i >= 2, bkg=bkg[i], ann=ann[i], scut=cuts[i], seedfit=False,
                             ccut=cuts[i], sx_maxdoca=2.5, sx_nsig=nsig[i], sx_scale=i < 3, gate=seed and i == 0,
                             **extcfg))
    out = {k: jnp.asarray([r[k] for r in rows]) for k in rows[0]}
    cs = float(__import__("os").environ.get("F1_CONVSCALE", "1"))     # diagnostic: tighter convergence test
    if cs != 1.0:
        out["conv"] = out["conv"] * cs
    return out


def t0_piece(cf, nself=10):
    """Parameters and covariance of the piece holding the POCA to the z axis, found self-consistently as
    PiecewiseTrajectory::t0 (KalSeed::t0Segment): start from the piece of the first effect, compute that piece's
    z-axis POCA track length s1, move to the piece whose range contains s1 (number of boundaries <= s1)."""
    if "owner" in cf:
        # time_model='stock': PiecewiseTrajectory::t0 on the owned pieces (front piece t0, then the piece holding it)
        ow = cf["owner"]
        t0_of = lambda i: kl_params(ow["P"][i])[4]
        step_t = lambda i, _: (nearest_index(ow["ranges"], ow["count"], t0_of(i)[None])[0], None)
        i, _ = jax.lax.scan(step_t, jnp.array(0, ow["count"].dtype), None, length=nself)
        return ow["P"][i], ow["C"][i], i
    xe, Ce, pid, bnd = cf["xe"], cf["Ce"], cf["pid"], cf["bnd"]

    def s1_of(x):
        p, u = point_dir(x)
        uz = u[2]
        return (p[2] * uz - p @ u) / jnp.maximum(1.0 - uz * uz, 1e-12)

    def step(e, _):
        pnew = jnp.searchsorted(bnd, s1_of(xe[e]), side="right")
        cand = pid == pnew
        return jnp.where(jnp.any(cand), jnp.argmax(cand), e), None

    e, _ = jax.lax.scan(step, jnp.array(0), None, length=nself)
    return xe[e], Ce[e], pid[e]
