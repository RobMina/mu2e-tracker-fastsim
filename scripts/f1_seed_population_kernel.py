"""Population seed-component kernels around the unchanged F1 fitter and schedule."""
from numbers import Integral

import jax.numpy as jnp

import f1_seed_arms as S
from trkfastsim.reco import kalman_jax as K

ARMS = ("ordinary", "control", "parameters", "momentum", "geometry_time", "covariance", "range", "full")


def select(arm, base, native):
    """Select synthetic seed operands without recomputing covariance or range after mixing q."""
    if arm not in ARMS:
        raise ValueError("unknown seed arm: " + str(arm))
    if arm in S.ARMS:
        return S.select(arm, base, native)
    first, last = (base, native) if arm == "momentum" else (native, base)
    q = jnp.concatenate((first["parameters"][:5], last["parameters"][5:]))
    return dict(parameters=q, covariance=base["covariance"], range=base["range"])


def fit(arm, x, h, c, drift, stages, ann, native, constraint_time, *, fixed=None, maxniter=10):
    """Return the complete fitter result; cap 3 supports production's two-pass orchestration."""
    if arm not in ARMS:
        raise ValueError("unknown seed arm: " + str(arm))
    if constraint_time is None:
        raise ValueError("seed arms require explicit constraint time")
    if not isinstance(maxniter, Integral) or maxniter not in (3, 10):
        raise ValueError("maxniter must be 3 or 10")
    base = S.baseline(x, h, c) if arm not in ("ordinary", "full") else None
    seed = select(arm, base, native)
    return K.fit_track(x, h, c, drift, stages, ann, fixed=fixed, maxniter=maxniter,
                       time_model="stock", reference_model="previous", crossing_time="current",
                       xing_activity="hit_or_doca", constraint_time=constraint_time, seed_input=seed)


def project(result):
    """Project one full result to population fields without changing its status or state masks."""
    cf, hist, good = result
    xt0, Ct0, _ = K.t0_piece(cf)
    return dict(xt0=xt0, Ct0=Ct0, kl=K.kl_params(xt0), good=good, stat=hist[0], niter=hist[3],
                chi2=hist[1][-1], ndof=hist[2][-1], fact=cf["st"]["active"], flr=cf["st"]["lr"],
                xact_fit=cf["xact"])
