"""Local-to-native operand conversion for diagnostics, not an F1 fitter.

Local x is (a0, b0, a1, b1, time_at_y0, momentum); native y is
(d0, phi0, z0, theta, time_at_POCA, momentum). The full JAX derivative of
kalman_jax.kl_params includes momentum-dependent flight time. Public functions
accept NumPy or concrete JAX arrays and return float64 NumPy arrays; validation
is host-side, so these functions are not themselves JIT primitives.

All leading batch axes are preserved, including empty axes. Matrix/row operands
must have exactly the same batch shape as x; material scalars may additionally
be scalar constants. No event loops, covariance regularization, or claims of
native arithmetic/fit closure are involved.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import kalman_jax as K


_parameters = jax.jit(jax.vmap(K.kl_params))
_jacobian = jax.jit(jax.vmap(jax.jacfwd(K.kl_params)))


def _finite(value, name, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite {name}")
    return value


def _reference(x):
    """Validate the coordinate chart and return y, J, inverse(J)."""
    x = _finite(x, "local parameters")
    if x.ndim < 1 or x.shape[-1] != 6:
        raise ValueError("local parameters must have shape (..., 6)")
    if np.any(x[..., 5] <= 0):
        raise ValueError("local momentum must be positive")
    # Reject the kl_params geometry floor and undefined signed-hypot derivative at zero impact.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        norm = np.sqrt(1.0 + x[..., 2] ** 2 + x[..., 3] ** 2)
        transverse2 = 1.0 - (x[..., 3] / norm) ** 2
    if (np.any(~np.isfinite(norm)) or np.any(transverse2 <= 1e-12)
            or np.any(x[..., 0] == 0)):
        raise ValueError("singular geometry for kl_params Jacobian (or geometry floor)")
    batch = x.shape[:-1]
    if x.size == 0:
        return (np.empty(batch + (6,)), np.empty(batch + (6, 6)),
                np.empty(batch + (6, 6)))
    flat = jnp.asarray(x.reshape(-1, 6))
    y = _finite(_parameters(flat), "native parameters").reshape(x.shape)
    jac = _finite(_jacobian(flat), "coordinate Jacobian (singular geometry)").reshape(batch + (6, 6))
    try:
        inverse = np.linalg.inv(jac)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular coordinate Jacobian") from exc
    return y, jac, _finite(inverse, "inverse coordinate Jacobian")


def transform_parameters(x, C):
    """Return y and first-order covariance J C J.T, with J = dy/dx.

    Shapes are (..., 6), (..., 6, 6). C must be finite; positive definiteness
    is not imposed (zero/semidefinite diagnostic covariances are supported).
    Singular/nonfinite geometry and nonpositive momentum raise ValueError.
    """
    y, jac, _ = _reference(x)
    covariance = _finite(C, "covariance", y.shape[:-1] + (6, 6))
    result = jac @ covariance @ jac.swapaxes(-1, -2)
    return y, _finite(result, "transformed covariance")


def transform_measurement(xref, H, r, w):
    """Return native local-linear (W, b) for residual r + H (x - xref).

    H has shape (..., rows, 6); r and nonnegative diagonal weights w have
    shape (..., rows). With yr = f(xref), A = inverse(J), Hy = H A:
        W = Hy.T diag(w) Hy
        b = Hy.T diag(w) (Hy yr - r).
    The affine origin is essential: b is NOT merely A.T times the local b.
    Empty row sets return zero weights/vector; zero weights are supported.
    """
    yr, _, inverse = _reference(xref)
    H = _finite(H, "measurement H")
    if H.ndim != yr.ndim + 1 or H.shape[:-2] != yr.shape[:-1] or H.shape[-1] != 6:
        raise ValueError("measurement H must have shape (..., rows, 6) matching xref")
    r = _finite(r, "measurement residual", H.shape[:-1])
    w = _finite(w, "measurement weights", H.shape[:-1])
    if np.any(w < 0):
        raise ValueError("measurement weights must be nonnegative")
    native_H = H @ inverse
    weighted_transpose = native_H.swapaxes(-1, -2) * w[..., None, :]
    W = weighted_transpose @ native_H
    target = np.einsum("...ij,...j->...i", native_H, yr) - r
    b = np.einsum("...ij,...j->...i", weighted_transpose, target)
    return _finite(W, "native measurement weights"), _finite(b, "native measurement vector")


def _scalar(value, name, batch):
    value = _finite(value, name)
    if value.shape != () and value.shape != batch:
        raise ValueError(f"{name} must be scalar or have batch shape {batch}")
    return np.broadcast_to(value, batch)


def transform_material(xref, D, sv, dm, dmv):
    """Return FIRST-ORDER native material (parameter change, covariance).

    D has shape (..., 6, 3), with two scattering columns and a momentum-change
    column. sv and dmv are nonnegative variances; dm is the signed momentum
    change. They are scalars or have xref's batch shape. The returned operands
    are J (D[:,2] dm) and (J D) diag(sv, sv, dmv) (J D).T. No finite-shift
    relinearization is performed; use exact_material_shift separately if needed.
    """
    yr, jac, _ = _reference(xref)
    batch = yr.shape[:-1]
    D = _finite(D, "material D", batch + (6, 3))
    sv = _scalar(sv, "scattering variance", batch)
    dm = _scalar(dm, "momentum change", batch)
    dmv = _scalar(dmv, "momentum variance", batch)
    if np.any(sv < 0) or np.any(dmv < 0):
        raise ValueError("material variances must be nonnegative")
    native_D = jac @ D
    shift = native_D[..., :, 2] * dm[..., None]
    variance = np.stack((sv, sv, dmv), axis=-1)
    covariance = (native_D * variance[..., None, :]) @ native_D.swapaxes(-1, -2)
    return _finite(shift, "native material shift"), _finite(covariance, "native material covariance")


def exact_material_shift(xref, D, dm):
    """Return f(xref + D[:,2] dm) - f(xref), NOT the first-order material shift.

    This is only the exact coordinate change of the supplied local update, not
    an exact physical material model. Both endpoints must pass chart/momentum
    validation. Native angles are raw differences, without wrapping.
    """
    yr, _, _ = _reference(xref)
    batch = yr.shape[:-1]
    D = _finite(D, "material D", batch + (6, 3))
    dm = _scalar(dm, "momentum change", batch)
    changed = np.asarray(xref, dtype=np.float64) + D[..., :, 2] * dm[..., None]
    yn, _, _ = _reference(changed)
    return _finite(yn - yr, "exact native coordinate shift")
