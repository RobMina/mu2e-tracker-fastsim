"""Device-only constant-Bz geometric seeds from padded measured positions.

This is the mathematical twin of ``helixseed``, not a stock/KinKal fit. Reductions
and eigensolvers may differ across backends: agreement is numerical, not bitwise
(tests use rtol=2e-10, atol=2e-9 on well-conditioned synthetic CPU comparisons).
No truth, fixed momentum, charge/direction, drift, material or covariance input.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

SUCCESS = 0
INSUFFICIENT_HITS = 1
INVALID_INPUT = 2
INVALID_CONFIGURATION = 3
ILL_CONDITIONED = 4
INVALID_GEOMETRY = 5
NONFINITE_RESULT = 6

_ESTIMATES = ("radius", "pitch", "phase_ref", "z_ref", "phase_span", "pt_abs",
              "p_abs", "circle_rms", "z_rms")


def _real_array(value, name):
    """Only static dtype validation; never inspect device array values."""
    try:
        value = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be real numeric") from exc
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric")
    return value.astype(jnp.float64)


def fit_helix_seed(positions, valid, *, bz, weights=None, condition_limit=1e12):
    """Fit measured ``positions[B,H,3]`` (mm), boolean ``valid[B,H]``.

    All returned values are device arrays, and this function is directly jittable.
    ``bz`` (tesla) and ``condition_limit`` are real scalars, including device
    scalars. Optional real scalar-position ``weights[B,H]`` must be positive and
    finite at valid hits. Shape/dtype errors raise ValueError at trace time;
    invalid *numeric values* fail their lane. A zero/nonfinite field or a
    nonfinite/<=1 condition limit fails every lane, without clamping to success.
    Masked NaN/Inf positions and weights are ignored. Empty lanes are retained.

    Uses weighted centered/scaled algebraic xy circle least squares, then a
    weighted z-versus-phase line. Phase is stably sorted by measured z and
    unwrapped. Requires >=4 hits, distinct z, identifiable circle and pitch,
    and 0 < phase span < 2*pi. True adjacent phase gaps must be < pi: undersampled
    turns cannot be resolved. No residual-quality or momentum cut is imposed.
    ``condition`` is the centered xy normal-matrix eigenvalue ratio, not an
    uncertainty; the configurable bound and a machine-precision rank guard apply.

    Returns ``success[B]`` (bool), ``status[B]`` (int32), ``n_hits[B]`` (int64),
    ``condition[B]``, ``center_xy[B,2]`` and float64 scalar estimates:
    radius, pitch=dz/dphase, phase_ref, z_ref, phase_span, pt_abs, p_abs,
    circle_rms, z_rms. The model is xy=center_xy+radius*(cos(phi),sin(phi)),
    z=z_ref+pitch*(phi-phase_ref); references are weighted means, with the phase
    branch starting at the smallest-z hit. Momentum magnitudes in MeV/c are
    0.299792458*abs(bz)*radius and 0.299792458*abs(bz)*hypot(radius,pitch).
    Pitch is geometric, not a momentum/charge/direction sign.

    Failed estimates (including center) are NaN; count and condition survive
    (condition is inf when no identifiable circle is available). Status codes:
    0 success, 1 insufficient hits, 2 invalid active position/weight,
    3 invalid field/limit, 4 ill-conditioned/rank-deficient circle,
    5 invalid phase/z/radius geometry, 6 nonfinite intermediate/result.
    Configuration, input and count failures take precedence, in that order;
    intermediate numerical failure precedes circle/geometry failure.
    Zero batch and zero hit capacity are supported without device reductions.
    """
    positions = _real_array(positions, "positions")
    valid = jnp.asarray(valid)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [batch, hits, 3]")
    if valid.shape != positions.shape[:2] or valid.dtype.kind != "b":
        raise ValueError("valid must be a boolean [batch, hits] array")
    bz = _real_array(bz, "bz")
    condition_limit = _real_array(condition_limit, "condition_limit")
    if bz.ndim != 0 or condition_limit.ndim != 0:
        raise ValueError("bz and condition_limit must be real scalars")
    if weights is None:
        weights = jnp.ones(valid.shape, dtype=jnp.float64)
    else:
        weights = _real_array(weights, "weights")
        if weights.shape != valid.shape:
            raise ValueError("weights must have shape [batch, hits]")

    batch, hits = valid.shape
    n_hits = jnp.sum(valid, axis=1, dtype=jnp.int64)
    config_ok = (jnp.isfinite(bz) & (bz != 0) & jnp.isfinite(condition_limit)
                 & (condition_limit > 1))
    finite = jnp.all(jnp.isfinite(positions), axis=-1)
    weight_ok = jnp.isfinite(weights) & (weights > 0)
    input_ok = jnp.all(~valid | (finite & weight_ok), axis=1)
    status = jnp.where(~config_ok, INVALID_CONFIGURATION,
                      jnp.where(~input_ok, INVALID_INPUT,
                                jnp.where(n_hits < 4, INSUFFICIENT_HITS, SUCCESS))).astype(jnp.int32)
    if batch == 0 or hits < 4:
        out = {key: jnp.full((batch,), jnp.nan, dtype=jnp.float64) for key in _ESTIMATES}
        out.update(center_xy=jnp.full((batch, 2), jnp.nan, dtype=jnp.float64),
                   success=jnp.zeros((batch,), dtype=jnp.bool_), status=status,
                   n_hits=n_hits, condition=jnp.full((batch,), jnp.inf, dtype=jnp.float64))
        return out

    # Sanitize masked nonfinite inputs before reductions and batched eigensolves.
    usable = valid & finite & weight_ok
    p = jnp.where(usable[..., None], positions, 0.0)
    w = jnp.where(usable, weights, 0.0)
    tiny, eps = jnp.finfo(jnp.float64).tiny, jnp.finfo(jnp.float64).eps
    w = w / jnp.maximum(jnp.max(w, axis=1, keepdims=True), tiny)
    w = w / jnp.maximum(jnp.sum(w, axis=1, keepdims=True), tiny)
    order = jnp.argsort(jnp.where(usable, p[..., 2], jnp.inf), axis=1, stable=True)
    anchor = jnp.take_along_axis(p, order[:, :1, None], axis=1)
    d = jnp.where(usable[..., None], p - anchor, 0.0)
    bounded = jnp.all(jnp.isfinite(d), axis=(1, 2))
    numeric_ok = bounded
    d = jnp.where(bounded[:, None, None], d, 0.0)
    mean = jnp.sum(w[..., None] * d, axis=1)
    d = jnp.where(usable[..., None], d - mean[:, None], 0.0)
    bounded = jnp.all(jnp.isfinite(d), axis=(1, 2)) & jnp.all(jnp.isfinite(mean), axis=1)
    numeric_ok &= bounded
    d = jnp.where(bounded[:, None, None], d, 0.0)
    scale = jnp.max(jnp.abs(d[..., :2]), axis=(1, 2))
    scale = jnp.where(scale > 0, scale, 1.0)
    xy = d[..., :2] / scale[:, None, None]
    rr = jnp.sum(xy * xy, axis=-1)
    rr_mean = jnp.sum(w * rr, axis=1)
    normal = jnp.einsum("bh,bhi,bhj->bij", w, xy, xy)
    rhs = jnp.sum((w * (rr - rr_mean[:, None]))[..., None] * xy, axis=1) / 2
    matrix_ok = (jnp.all(jnp.isfinite(normal), axis=(1, 2))
                 & jnp.all(jnp.isfinite(rhs), axis=1) & jnp.isfinite(rr_mean))
    numeric_ok &= matrix_ok
    # Replace nonfinite matrices with rank-deficient zeros before the eigensolve.
    normal = jnp.where(matrix_ok[:, None, None], normal, 0.0)
    rhs = jnp.where(matrix_ok[:, None], rhs, 0.0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(normal)
    lo, hi = eigenvalues[:, 0], eigenvalues[:, 1]
    condition = jnp.where((lo > 0) & (hi > 0) & matrix_ok,
                          hi / jnp.where(lo > 0, lo, 1.0), jnp.inf)
    circle_ok = (jnp.isfinite(condition) & (condition <= condition_limit)
                 & (lo > 32 * eps * hi))
    safe_eigenvalues = jnp.where(circle_ok[:, None], eigenvalues, 1.0)
    projection = jnp.einsum("bji,bj->bi", eigenvectors, rhs)
    center = jnp.einsum("bij,bj->bi", eigenvectors, projection / safe_eigenvalues)
    radius_scaled = jnp.sqrt(rr_mean + jnp.sum(center * center, axis=1))
    radius = scale * radius_scaled
    delta = xy - center[:, None]
    phase = jnp.arctan2(delta[..., 1], delta[..., 0])
    phase = jnp.unwrap(jnp.take_along_axis(phase, order, axis=1), axis=1)
    sw = jnp.take_along_axis(w, order, axis=1)
    sz = jnp.take_along_axis(d[..., 2], order, axis=1)
    active = jnp.take_along_axis(usable, order, axis=1)
    phase_ref = jnp.sum(sw * phase, axis=1)
    dp = jnp.where(active, phase - phase_ref[:, None], 0.0)
    phase_var = jnp.sum(sw * dp * dp, axis=1)
    pitch = jnp.sum(sw * dp * sz, axis=1) / jnp.where(phase_var > 0, phase_var, 1.0)
    span = (jnp.max(jnp.where(active, phase, -jnp.inf), axis=1)
            - jnp.min(jnp.where(active, phase, jnp.inf), axis=1))
    adjacent = active[:, 1:] & active[:, :-1]
    ordered_z = jnp.all(~adjacent | (jnp.diff(sz, axis=1) > 0), axis=1)
    geometry_ok = ((radius > 0) & ordered_z & (phase_var > eps**2)
                   & (pitch != 0) & (span > 0) & (span < 2 * jnp.pi))
    values = dict(radius=radius, pitch=pitch, phase_ref=phase_ref,
                  z_ref=anchor[:, 0, 2] + mean[:, 2], phase_span=span,
                  pt_abs=0.299792458 * jnp.abs(bz) * radius,
                  p_abs=0.299792458 * jnp.abs(bz) * jnp.hypot(radius, pitch),
                  circle_rms=scale * jnp.sqrt(jnp.sum(w * (jnp.hypot(delta[..., 0], delta[..., 1])
                                                                           - radius_scaled[:, None])**2, axis=1)),
                  z_rms=jnp.sqrt(jnp.sum(sw * (sz - pitch[:, None] * dp)**2, axis=1)))
    center_xy = anchor[:, 0, :2] + mean[:, :2] + scale[:, None] * center
    result_ok = jnp.all(jnp.isfinite(center_xy), axis=1)
    for value in values.values():  # Fixed output fields, never events or hits.
        result_ok &= jnp.isfinite(value)
    fit_status = jnp.where(~numeric_ok, NONFINITE_RESULT,
                           jnp.where(~circle_ok, ILL_CONDITIONED,
                                     jnp.where(~geometry_ok, INVALID_GEOMETRY,
                                               jnp.where(~result_ok, NONFINITE_RESULT, SUCCESS))))
    status = jnp.where(status == SUCCESS, fit_status, status).astype(jnp.int32)
    success = status == SUCCESS
    out = {key: jnp.where(success, value, jnp.nan) for key, value in values.items()}
    out.update(center_xy=jnp.where(success[:, None], center_xy, jnp.nan),
               success=success, status=status, n_hits=n_hits, condition=condition)
    return out
