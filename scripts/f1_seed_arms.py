"""Experimental seed-component arms around the unchanged F1 fitter; no default changes."""
import jax.numpy as jnp
import numpy as np

from trkfastsim.reco import kalman_jax as K

ARMS = ("ordinary", "control", "parameters", "covariance", "range", "full")
COUPLINGS = {
    "ordinary": [],
    "control": [],
    "parameters": ["initial reference means", "initial unbiased means", "initial prior means",
                   "seed-derived momentum constraint target", "initial geometry and times"],
    "covariance": ["initial reference covariances", "initial unbiased covariances", "initial prior covariances",
                   "downstream uncertainty-dependent activity and material operands"],
    "range": ["initial single-piece owner range"],
    "full": ["all parameters-arm consequences", "all covariance-arm consequences", "initial single-piece owner range"],
}


def validate(x, hits, crossings, native, constraint_time):
    """Check one track's experiment inputs before entering JIT; never repair operands."""
    def real(value, shape, label):
        out = np.asarray(value)
        if np.ma.isMaskedArray(value) or out.shape != shape or out.dtype.kind != "f" or not np.all(np.isfinite(out)):
            raise ValueError("invalid "+label)
        return out
    if np.ma.isMaskedArray(x):
        raise ValueError("masked baseline parameters")
    x = np.asarray(x)
    if x.shape not in ((5,), (6,)):
        raise ValueError("baseline parameters require five or six components")
    real(x, x.shape, "baseline parameters")
    if x.shape == (6,) and x[5] <= 0:
        raise ValueError("nonpositive baseline momentum")
    for collection, label in ((hits, "hits"), (crossings, "crossings")):
        if np.ma.isMaskedArray(collection["valid"]):
            raise ValueError("masked "+label+" valid mask")
        valid = np.asarray(collection["valid"])
        if valid.ndim != 1 or len(valid) == 0 or valid.dtype.kind != "b":
            raise ValueError("invalid "+label+" valid mask")
        for field in ("mid", "wdir"):
            value = real(collection[field], (len(valid), 3), label+" "+field)
            if field == "wdir" and np.any(~np.any(value != 0, axis=1)):
                raise ValueError("zero "+label+" wdir")
    if np.ma.isMaskedArray(hits.get("inseed")) or np.ma.isMaskedArray(crossings["hit"]):
        raise ValueError("masked seed membership or crossing partner")
    hv, cv = np.asarray(hits["valid"]), np.asarray(crossings["valid"])
    seed = np.asarray(hits.get("inseed", np.ones_like(hv)))
    if seed.shape != hv.shape or seed.dtype.kind != "b" or not np.any(seed & hv):
        raise ValueError("no valid seed hit or invalid seed mask")
    partner = np.asarray(crossings["hit"])
    if partner.shape != cv.shape or partner.dtype.kind not in "iu" or np.any(partner < -1) or np.any(partner >= len(hv)):
        raise ValueError("invalid crossing partner")
    if np.any(cv & (partner >= 0) & ~hv[np.maximum(partner, 0)]):
        raise ValueError("crossing partner refers to padded hit")
    if set(native) != {"parameters", "covariance", "range"}:
        raise ValueError("native seed fields differ")
    q = real(native["parameters"], (6,), "native parameters")
    covariance = real(native["covariance"], (6, 6), "native covariance")
    ranges = real(native["range"], (2,), "native range")
    real(constraint_time, (), "explicit constraint time")
    if q[5] <= 0 or ranges[0] > ranges[1] or np.any(np.diag(covariance) <= 0):
        raise ValueError("invalid native seed momentum/range/diagonal")
    if not np.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-7):
        raise ValueError("asymmetric native covariance")
    if np.any(np.linalg.eigvalsh(covariance) <= 0):
        raise ValueError("nonpositive native covariance; no regularization")


def baseline(x, hits, crossings):
    """Reproduce K.fit_track initialization inside the caller's JIT, before any arm mixing."""
    if x.shape[0] == 5:
        x = jnp.concatenate((x, jnp.array([K.MOM])))
    covariance = K.seed_covariance(x)
    s_hit = K.straw_poca(x, hits["mid"], hits["wdir"])[0]
    valid = hits["valid"] & (hits["inseed"] if "inseed" in hits else True)
    speed = K.speed(x[5])
    t_hit = jnp.where(valid, x[4]+s_hit/speed, jnp.nan)
    s_crossing = K.straw_poca(x, crossings["mid"], crossings["wdir"])[0]+K.kkmaterial.STRAW["wrad"]
    crossing_valid = crossings["valid"] & (crossings["hit"] >= 0) & valid[jnp.maximum(crossings["hit"], 0)]
    t_crossing = jnp.where(crossing_valid, x[4]+s_crossing/speed, jnp.nan)
    times = jnp.concatenate((t_hit, t_crossing))
    return dict(parameters=x, covariance=covariance, range=jnp.stack((jnp.nanmin(times), jnp.nanmax(times))))


def select(arm, base, native):
    """Mixed tuples are synthetic operand ablations, never described as native seeds."""
    if arm not in ARMS:
        raise ValueError("unknown seed arm: "+str(arm))
    if arm == "ordinary":
        return None
    return {name: native[name] if arm in (name, "full") else base[name]
            for name in ("parameters", "covariance", "range")}


def fit(arm, x, hits, crossings, drift, stages, ann, native, constraint_time, *, fixed=None):
    """Unchanged full-cap fitter, explicit identical constraint time and fixed policies."""
    if arm not in ARMS:
        raise ValueError("unknown seed arm: "+str(arm))
    if constraint_time is None:
        raise ValueError("seed arms require explicit constraint time")
    base = baseline(x, hits, crossings) if arm not in ("ordinary", "full") else None
    seed = None if arm == "ordinary" else native if arm == "full" else select(arm, base, native)
    return K.fit_track(x, hits, crossings, drift, stages, ann, fixed=fixed, maxniter=10,
                       time_model="stock", reference_model="previous", crossing_time="current",
                       xing_activity="hit_or_doca", constraint_time=constraint_time, seed_input=seed)
