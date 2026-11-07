"""Experimental Run1A measured-position helix seeds, not stock reco or a KinKal fit.

Only a constant axial field and scalar seed weights are used: no truth, fixed
momentum, drift, material, timing, charge, direction, or fitted covariance.
"""
import numpy as np


_ESTIMATES = ("radius", "pitch", "phase_ref", "z_ref", "phase_span", "pt_abs",
              "p_abs", "circle_rms", "z_rms")


def fit_helix_seed(positions, valid, *, bz, weights=None, condition_limit=1e12):
    """Fit padded measured positions [batch, hits, 3] in mm; ``bz`` is tesla.

    ``valid`` must be a boolean [batch, hits] mask. Optional weights have the
    same shape and must be finite and positive at valid hits; masked positions
    and weights are ignored, even if nonfinite. Nonfinite *valid* positions fail
    their batch, rather than silently removing measurements. Shape, weight, and
    field errors raise ValueError. All batches, including empty ones, survive.

    Weighted algebraic xy circle least squares uses centered, scaled coordinates.
    ``condition`` is the eigenvalue ratio of its centered xy normal matrix
    (squared design condition), independent of coordinate units and translation.
    The configurable finite limit > 1 defaults to 1e12; short arcs are not vetoed
    by a momentum/phase cut, but can have very uncertain curvature even below
    this limit. A machine-precision rank guard always applies. This is a
    numerical diagnostic, not a statistical uncertainty.

    Phase is ordered by measured z, then unwrapped; z is fitted against phase.
    Assumes a non-looping arc, distinct measured z, and true adjacent phase gaps
    strictly below pi; undersampled turns cannot be disambiguated. Repeated z
    (including duplicate points), fewer than four hits, singular circles, zero
    z span, unidentifiable pitch, and phase spans >= 2*pi fail. Noise can violate
    these assumptions; no outlier removal or residual-quality cut is applied.

    Returns a dict of arrays: success, n_hits, condition, center_xy [batch, 2],
    radius, pitch=dz/dphase [mm/rad], phase_ref [rad], z_ref [mm], phase_span,
    pt_abs and p_abs [MeV/c], circle_rms and z_rms [mm]. The model is
    xy = center_xy + radius*(cos(phase), sin(phase)),
    z = z_ref + pitch*(phase-phase_ref). The phase branch starts at the
    smallest-z hit; reference values are weighted means. Pitch is geometric,
    not signed momentum: |pT| = 0.299792458*|bz|*radius and
    |p| = 0.299792458*|bz|*hypot(radius, pitch). No charge or flight direction
    can be inferred here. Failed estimates are NaN; n_hits and condition remain
    diagnostics (condition is inf if no identifiable circle is available).
    """
    positions = np.asarray(positions)
    valid = np.asarray(valid)
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.dtype.kind not in "fiu":
        raise ValueError("positions must be a real numeric [batch, hits, 3] array")
    if valid.shape != positions.shape[:2] or valid.dtype.kind != "b":
        raise ValueError("valid must be a boolean [batch, hits] array")
    positions = positions.astype(np.float64, copy=False)
    for name, value in (("bz", bz), ("condition_limit", condition_limit)):
        a = np.asarray(value)
        if a.ndim != 0 or a.dtype.kind not in "fiu" or not np.isfinite(a):
            raise ValueError(f"{name} must be a finite real scalar")
    bz, condition_limit = float(bz), float(condition_limit)
    if bz == 0 or condition_limit <= 1:
        raise ValueError("bz must be nonzero and condition_limit must exceed one")
    if weights is None:
        weights = np.ones(valid.shape)
    else:
        weights = np.asarray(weights)
        if weights.shape != valid.shape or weights.dtype.kind not in "fiu":
            raise ValueError("weights must be a real numeric [batch, hits] array")
        weights = weights.astype(np.float64, copy=False)
    if np.any(valid & (~np.isfinite(weights) | (weights <= 0))):
        raise ValueError("valid weights must be finite and positive")
    batch, hits = valid.shape
    out = {key: np.full(batch, np.nan) for key in _ESTIMATES}
    out.update(center_xy=np.full((batch, 2), np.nan), success=np.zeros(batch, bool),
               n_hits=valid.sum(axis=1), condition=np.full(batch, np.inf))
    if batch == 0 or hits < 4:
        return out

    finite = np.isfinite(positions).all(axis=-1)
    usable = valid & finite
    good = (out["n_hits"] >= 4) & np.all(~valid | finite, axis=1)
    p = np.where(usable[..., None], positions, 0.0)
    w = np.where(usable, weights, 0.0)
    w /= np.maximum(w.max(axis=1, keepdims=True), np.finfo(float).tiny)
    w /= np.maximum(w.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    order = np.argsort(np.where(usable, p[..., 2], np.inf), axis=1, kind="stable")
    anchor = np.take_along_axis(p, order[:, :1, None], axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        d = np.where(usable[..., None], p - anchor, 0.0)
    bounded = np.isfinite(d).all(axis=(1, 2))
    good &= bounded
    d = np.where(bounded[:, None, None], d, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.sum(w[..., None] * d, axis=1)
        d = np.where(usable[..., None], d - mean[:, None], 0.0)
    bounded = np.isfinite(d).all(axis=(1, 2)) & np.isfinite(mean).all(axis=1)
    good &= bounded
    d = np.where(bounded[:, None, None], d, 0.0)
    scale = np.max(np.abs(d[..., :2]), axis=(1, 2))
    scale = np.where(scale > 0, scale, 1.0)
    xy = d[..., :2] / scale[:, None, None]
    rr = np.sum(xy * xy, axis=-1)
    rr_mean = np.sum(w * rr, axis=1)
    normal = np.einsum("bh,bhi,bhj->bij", w, xy, xy)
    rhs = np.sum((w * (rr - rr_mean[:, None]))[..., None] * xy, axis=1) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    lo, hi = eigenvalues[:, 0], eigenvalues[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.where((lo > 0) & (hi > 0), hi / lo, np.inf)
    out["condition"] = condition
    circle_ok = np.isfinite(condition) & (condition <= condition_limit)
    circle_ok &= lo > 32 * np.finfo(float).eps * hi
    safe_eigenvalues = np.where(circle_ok[:, None], eigenvalues, 1.0)
    projection = np.einsum("bji,bj->bi", eigenvectors, rhs)
    center = np.einsum("bij,bj->bi", eigenvectors, projection / safe_eigenvalues)
    radius_scaled = np.sqrt(rr_mean + np.sum(center * center, axis=1))
    radius = scale * radius_scaled
    delta = xy - center[:, None]
    phase = np.arctan2(delta[..., 1], delta[..., 0])
    phase = np.unwrap(np.take_along_axis(phase, order, axis=1), axis=1)
    sw = np.take_along_axis(w, order, axis=1)
    sz = np.take_along_axis(d[..., 2], order, axis=1)
    active = np.take_along_axis(usable, order, axis=1)
    phase_ref = np.sum(sw * phase, axis=1)
    dp = np.where(active, phase - phase_ref[:, None], 0.0)
    phase_var = np.sum(sw * dp * dp, axis=1)
    pitch = np.sum(sw * dp * sz, axis=1) / np.where(phase_var > 0, phase_var, 1.0)
    phase_min = np.min(np.where(active, phase, np.inf), axis=1)
    phase_max = np.max(np.where(active, phase, -np.inf), axis=1)
    span = phase_max - phase_min
    adjacent = active[:, 1:] & active[:, :-1]
    ordered_z = np.all(~adjacent | (np.diff(sz, axis=1) > 0), axis=1)
    good &= circle_ok & (radius > 0) & ordered_z & (phase_var > np.finfo(float).eps**2)
    good &= (pitch != 0) & (span > 0) & (span < 2 * np.pi)
    with np.errstate(over="ignore", invalid="ignore"):
        values = dict(radius=radius, pitch=pitch, phase_ref=phase_ref,
                      z_ref=anchor[:, 0, 2] + mean[:, 2], phase_span=span,
                      pt_abs=0.299792458 * abs(bz) * radius,
                      p_abs=0.299792458 * abs(bz) * np.hypot(radius, pitch),
                      circle_rms=scale * np.sqrt(np.sum(w * (np.hypot(delta[..., 0], delta[..., 1])
                                                                             - radius_scaled[:, None])**2, axis=1)),
                      z_rms=np.sqrt(np.sum(sw * (sz - pitch[:, None] * dp)**2, axis=1)))
        center_xy = anchor[:, 0, :2] + mean[:, :2] + scale[:, None] * center
    good &= np.isfinite(center_xy).all(axis=1)
    for key, value in values.items():
        good &= np.isfinite(value)
    out.update({key: np.where(good, value, np.nan) for key, value in values.items()})
    out.update(center_xy=np.where(good[:, None], center_xy, np.nan), success=good)
    return out
