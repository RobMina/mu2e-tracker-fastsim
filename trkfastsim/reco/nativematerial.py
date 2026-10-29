"""Saved-operand material-noise diagnostics, NOT a fitter material implementation.

The frozen p109 DetMaterial::scatterAngleVar uses per-material (nonstatic)
vfactor and sig2factor. Fractions here are explicit and never read/set F1_GASFRAC.
Only material constants are shared with kkmaterial; the scalar arithmetic below
is NumPy, independent of its JAX implementation. The separate kink-time
diagnostic re-evaluates the existing local D only, with no path or scale fitting.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import kalman_jax as K

from .kkmaterial import material_constants, MMU

SOLID_FRACTION = 0.999999
DEFAULT_GAS_FRACTION = 0.999999
P109_GAS_FRACTION = 0.9999999


def _finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite " + name)
    return value


def scatter_angle_var(name, momentum, path, *, fraction, mass=MMU):
    """Projected Lynch-Dahl angular variance (rad^2), no temperature scaling.

    Momentum/path broadcast normally. Signed paths use abs(path), as native.
    Zero path returns exactly zero; no floor is applied to a positive variance.
    """
    p, path = np.broadcast_arrays(_finite(momentum, "momentum"), _finite(path, "path"))
    fraction, mass = _finite(fraction, "fraction"), _finite(mass, "mass")
    if fraction.shape != () or not 0 <= fraction < 1:
        raise ValueError("fraction must be scalar in [0, 1)")
    if mass.shape != () or mass <= 0 or np.any(p <= 0):
        raise ValueError("mass and momentum must be positive (scalar mass)")
    c = material_constants(name)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        beta = p / np.sqrt(p*p + mass*mass)
        chic2 = c["chic2"] * np.abs(path) * c["density"] / (beta*beta*p*p)
        chia2 = c["chia2_1"] * (1 + c["chia2_2"]/(beta*beta)) / (p*p)
        v = (0.5/(1-fraction)) * chic2 / chia2
        vs = np.where(v > 0, v, 1.)
        result = chic2/(1+fraction*fraction) * ((1+vs)*np.log1p(vs)/vs - 1)
    _finite(v, "Lynch-Dahl v")
    _finite(result, "scattering variance")
    return np.where(v > 0, np.maximum(result, 0.), 0.)


def scattering_variants(momentum, wall, gas, varscale):
    """Two explicit policies on the same saved paths/momentum, including scale.

    Returns scaled wall, gas_default, gas_p109, default and p109 scalars. The
    scale is supplied from the saved first-stage temperature, never optimized.
    """
    p, w, g = np.broadcast_arrays(_finite(momentum, "momentum"), _finite(wall, "wall path"),
                                  _finite(gas, "gas path"))
    scale = _finite(varscale, "variance scale")
    if scale.shape != () or scale <= 0:
        raise ValueError("variance scale must be positive scalar")
    if np.any(w < 0) or np.any(g < 0):
        raise ValueError("saved material paths must be nonnegative")
    wallvar = scatter_angle_var("straw-wall", p, w, fraction=SOLID_FRACTION) * scale
    gas_default = scatter_angle_var("straw-gas", p, g, fraction=DEFAULT_GAS_FRACTION) * scale
    gas_p109 = scatter_angle_var("straw-gas", p, g, fraction=P109_GAS_FRACTION) * scale
    return {"wall": wallvar, "gas_default": gas_default, "gas_p109": gas_p109,
            "default": wallvar + gas_default, "p109": wallvar + gas_p109}


def native_scalars(covariance, varscale):
    """Extract scaled/unscaled projected angle and longitudinal momentum variances.

    Native parameter order: (d0, phi0, z0, theta, t0, mom). Orthonormal momentum
    basis and isotropic transverse scattering give Q[3,3]=scale*scatvar and
    Q[5,5]=scale*paramomvar, regardless of kink time. These are momentum, not
    energy, variance units; dE variance would additionally need (p/E)^2.
    """
    q = _finite(covariance, "native covariance")
    scale = _finite(varscale, "variance scale")
    if q.ndim < 2 or q.shape[-2:] != (6, 6):
        raise ValueError("native covariance must have shape (..., 6, 6)")
    if scale.shape != () or scale <= 0:
        raise ValueError("variance scale must be positive scalar")
    sv, dmv = q[..., 3, 3], q[..., 5, 5]
    if np.any(sv < 0) or np.any(dmv < 0):
        raise ValueError("negative native scalar variance")
    return {"scaled_sv": sv, "scaled_dmv": dmv, "sv": sv/scale, "dmv": dmv/scale}


def offset_kink_D(xref, saved_D, offset):
    """Add the exact linear-in-distance D increment to saved D, without time subtraction."""
    x, D, offset = _finite(xref, "kink reference"), _finite(saved_D, "saved D"), _finite(offset, "offset")
    if x.ndim != 2 or x.shape[1:] != (6,) or D.shape != (len(x), 6, 3) or offset.shape != (len(x),):
        raise ValueError("offset D shapes differ")
    if np.any(x[:, 5] <= 0) or np.any(offset < 0):
        raise ValueError("invalid offset or momentum")
    xj = jnp.asarray(x)
    zero, one = jax.vmap(K.kink_jacobian)(xj, jnp.zeros(len(x))), jax.vmap(K.kink_jacobian)(xj, jnp.ones(len(x)))
    ds = np.asarray(K.speed(xj[:, 5]))*offset
    increment = np.asarray(one-zero)*ds[:, None, None]
    return _finite(D+increment, "offset D"), ds


def kink_at_time(xref, time):
    """Evaluate local D at explicit physical times, never change the references.

    Each row uses s=(time-xref[4])*K.speed(xref[5]). No CA, paths, material
    scalars, native parameters or fitted state are reconstructed/substituted.
    Returns (s, D), batched over crossings; the caller owns time provenance.
    """
    x, time = _finite(xref, "kink reference"), _finite(time, "kink time")
    if x.ndim != 2 or x.shape[1:] != (6,) or time.shape != x.shape[:1]:
        raise ValueError("kink reference/time shape must be (N, 6)/(N,)")
    if np.any(x[:, 5] <= 0):
        raise ValueError("kink momentum must be positive")
    xj = jnp.asarray(x)
    s = (jnp.asarray(time)-xj[:, 4])*K.speed(xj[:, 5])
    D = jax.vmap(K.kink_jacobian)(xj, s)
    return _finite(s, "kink distance"), _finite(D, "kink D")
