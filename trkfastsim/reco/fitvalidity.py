"""Diagnostic-only finite/PSD checks; these do not redefine stock or F1 acceptance."""
import numpy as np


def covariance_checks(cov, tolerance=1e-10):
    """Check (..., P, P) covariances in diagonal-scaled coordinates.

    Zero diagonal entries use unit scale, so a zero PSD covariance remains valid.
    Eigenvalues describe the symmetric part, but asymmetry is reported separately.
    The tolerance is numerical, not a physics acceptance or conditioning threshold.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim < 2 or cov.shape[-1] != cov.shape[-2] or not cov.shape[-1]:
        raise ValueError("covariance must have shape (..., P, P), P > 0")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    finite = np.isfinite(cov).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], cov, np.eye(cov.shape[-1]))
    diag = np.diagonal(safe, axis1=-2, axis2=-1)
    scale = np.sqrt(np.where(diag > 0, diag, 1.0))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = safe / scale[..., :, None] / scale[..., None, :]
    scalable = np.isfinite(scaled).all(axis=(-2, -1))
    scaled = np.where(scalable[..., None, None], scaled, np.eye(cov.shape[-1]))
    transpose = np.swapaxes(scaled, -1, -2)
    asymmetry = np.max(np.abs(scaled - transpose), axis=(-2, -1))
    eigenvalues = np.linalg.eigvalsh(0.5 * scaled + 0.5 * transpose)
    evaluated = finite & scalable
    nonnegative_diagonal = finite & (diag >= 0).all(axis=-1)
    symmetric = evaluated & (asymmetry <= tolerance)
    psd = evaluated & (eigenvalues[..., 0] >= -tolerance)
    return {"finite": finite, "scalable": scalable & finite,
            "nonnegative_diagonal": nonnegative_diagonal, "symmetric": symmetric,
            "psd": psd, "valid": nonnegative_diagonal & symmetric & psd,
            "min_scaled_eigenvalue": np.where(evaluated, eigenvalues[..., 0], np.nan)}


def fit_checks(x, cov, valid_effect, t0_x, t0_cov, chi2, ndof, good, stockok, tolerance=1e-10):
    """Classify final returned effect states and the t0 state, preserving fit flags.

    Masked padding is excluded. This checks represented final states, not every
    intermediate iteration or an unrepresented trajectory piece. Population counts
    use the original good/FitOK definitions, including invalid-but-accepted tracks.
    """
    x, cov = np.asarray(x), np.asarray(cov)
    mask = np.asarray(valid_effect, dtype=bool)
    t0_x, t0_cov = np.asarray(t0_x), np.asarray(t0_cov)
    n = len(x)
    if x.ndim != 3 or cov.shape != x.shape + (x.shape[-1],) or mask.shape != x.shape[:2]:
        raise ValueError("expected x [track,effect,param], covariance [track,effect,param,param], and effect mask")
    if t0_x.shape != (n, x.shape[-1]) or t0_cov.shape != (n, x.shape[-1], x.shape[-1]):
        raise ValueError("t0 state shape does not match effect states")
    chi2, ndof, good, stockok = (np.asarray(v) for v in (chi2, ndof, good, stockok))
    if any(v.shape != (n,) for v in (chi2, ndof, good, stockok)):
        raise ValueError("statistics and acceptance flags must have shape [track]")
    good, stockok = good.astype(bool), stockok.astype(bool)
    effects = covariance_checks(cov, tolerance)
    t0 = covariance_checks(t0_cov, tolerance)
    checks = {"has_effect": mask.any(axis=1),
              "finite_parameters": (np.isfinite(x).all(axis=-1) | ~mask).all(axis=1) & np.isfinite(t0_x).all(axis=-1),
              "finite_statistics": np.isfinite(chi2) & np.isfinite(ndof),
              "nonnegative_chi2": np.isfinite(chi2) & (chi2 >= 0),
              "positive_ndof": np.isfinite(ndof) & (ndof > 0)}
    for key in ("finite", "scalable", "nonnegative_diagonal", "symmetric", "psd"):
        checks["covariance_" + key] = (effects[key] | ~mask).all(axis=1) & t0[key]
    checks["valid"] = np.logical_and.reduce(list(checks.values()))
    populations = {"all": np.ones(n, bool), "f1_good": good, "both_good": good & stockok,
                   "f1_only": good & ~stockok, "stock_only": ~good & stockok, "neither": ~good & ~stockok}
    summary = {"diagnostic_only": True, "scaled_tolerance": tolerance,
               "scope": "final returned valid-effect and t0 states; no intermediate-iteration coverage",
               "populations": {}}
    for name, selected in populations.items():
        summary["populations"][name] = {"n_tracks": int(selected.sum()),
            "n_valid": int((selected & checks["valid"]).sum()),
            "failures": {key: int((selected & ~value).sum()) for key, value in checks.items() if key != "valid"}}
    return checks, summary
