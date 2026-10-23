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
    """d(params after kink)/d(th1, th2) at th = 0: [5, 2]."""
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
    Q = jax.vmap(lambda s, v: crossing_noise(x, s, v))(st, sv)
    toff = kkmaterial.STRAW["wrad"]
    return {"s": st + toff, "Q": Q, "scatvar": sv, "doca": doca, "docavar": docavar, "wall": wall, "gas": gas,
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


def _add_noise(W, b, Q):
    """(W, b) -> covariance form, C += Q, back (Material::process; parameters unchanged)."""
    C = jnp.linalg.inv(W)
    x = C @ b
    Wn = jnp.linalg.inv(C + Q)
    return Wn, Wn @ x


def _hit_info(e):
    Hw = e["H"] * e["w"][:, None]
    return jnp.einsum("ki,kj->ij", Hw, e["H"]), Hw.T @ (e["H"] @ e["xref"] - e["r"])


def _predicted_chi2(W, b, e):
    """Chisq of the hit's rows against state (W, b) (ResidualHit::chisq): predicted residual r + H (x - xref), variance
    mvar + H C H^T (measurement variance not temperature-scaled)."""
    C = jnp.linalg.inv(W)
    x = C @ b
    res = e["r"] + e["H"] @ (x - e["xref"])
    var = e["mvar"] + jnp.einsum("ki,ij,kj->k", e["H"], C, e["H"])
    act = e["w"] > 0
    return jnp.sum(jnp.where(act, res * res / jnp.where(act, var, 1.0), 0.0)), jnp.sum(act)


def filter_smooth(eff, W0f, b0f, W0b, b0b):
    """One algebraic iteration for one track. W0f/b0f and W0b/b0b: deweighted priors at the front and back
    (Track::initFitState). Returns per effect the smoothed information (forward through the effect + backward
    from the next effect: the piece holding the effect), the forward chi2 and ndof (ndof - 5 as Chisq(NParams))."""
    def fwd(carry, e):
        W, b, chi2, nd = carry
        c, n = _predicted_chi2(W, b, e)
        dW, db = _hit_info(e)
        W, b = W + dW, b + db
        W, b = _add_noise(W, b, e["Q"])
        return (W, b, chi2 + c, nd + n), (W, b)

    def bwd(carry, e):
        W, b = carry
        out = (W, b)                                   # downstream information, before this effect
        W, b = _add_noise(W, b, e["Q"])
        dW, db = _hit_info(e)
        return (W + dW, b + db), out

    (_, _, chi2, nd), (Wf, bf) = jax.lax.scan(fwd, (W0f, b0f, 0.0, 0), eff)
    (Wbf, bbf), (Wb, bb) = jax.lax.scan(bwd, (W0b, b0b), eff, reverse=True)
    Ws, bs = Wf + Wb, bf + bb
    return {"W": Ws, "b": bs, "chi2": chi2, "ndof": nd - 5, "Wfront": Wbf, "bfront": bbf}


def smoothed(fs, eff):
    """Parameters and covariance of each effect's piece, and the unbiased parameters of each hit (Hit::
    unbiasedParameters: the piece weight minus the hit's own weight)."""
    C = jnp.linalg.inv(fs["W"])
    x = jnp.einsum("eij,ej->ei", C, fs["b"])
    dW, db = jax.vmap(_hit_info)(eff)
    Cu = jnp.linalg.inv(fs["W"] - dW)
    xu = jnp.einsum("eij,ej->ei", Cu, fs["b"] - db)
    return x, C, xu, Cu
