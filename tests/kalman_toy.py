"""Toy cosmic tracks for the F1/F2 fit tests: straws in planes of constant z, a downward line, drift times from a
linear drift table, hits generated with the F2 residual model so the true line has zero residuals before smearing."""
import jax.numpy as jnp
import numpy as np

from trkfastsim.reco.linefit_jax import _resid_one

DRIFT = {"dt": np.linspace(-20.0, 80.0, 101)}
DRIFT["rdrift"] = np.clip(DRIFT["dt"], 0.0, None) * 0.05
DRIFT["sderr"] = np.full(101, 0.15)
DRIFT["uderr"] = np.full(101, 0.3)
DRIFT["dvel"] = np.full(101, 0.05)


def make_track(rng, n=20, xtrue=(50.0, -100.0, 0.3, -0.2, 10.0), smear=True):
    """Hits along the line x = xtrue; returns (x_true, hit dict [N], crossing dict [N], drift dict)."""
    xt = np.asarray(xtrue, float)
    p0 = np.array([xt[0], 0.0, xt[1]])
    u = np.array([xt[2], -1.0, xt[3]]); u /= np.linalg.norm(u)
    s = np.sort(rng.uniform(-600, 600, n))
    pts = p0 + u * s[:, None]
    phi = rng.uniform(0, np.pi, n)
    wdir = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], 1)
    nrm = np.cross(u, wdir); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    doca = rng.uniform(-2.3, 2.3, n)
    along = rng.uniform(-200, 200, n)
    mid = pts + nrm * doca[:, None] - wdir * along[:, None]
    h = {"active": np.ones(n, bool), "lr": np.sign(doca), "tmeas": np.zeros(n), "hlen": np.full(n, 500.0),
         "esign": np.ones(n), "vsig": np.full(n, 200.0), "mid": mid, "wdir": wdir, "totdt": np.zeros(n),
         "tvar": np.full(n, 4.0), "uvar": np.full(n, 900.0), "wdist": np.zeros(n), "udir_dot_w": np.ones(n),
         "edep": np.full(n, 0.002)}
    d = {k: jnp.asarray(v) for k, v in DRIFT.items()}
    hj = {k: jnp.asarray(v) for k, v in h.items()}
    # zero residuals at the truth: choose tmeas so that rdrift(dT) = |DOCA|, totdt = dT, wdist = wire position
    r0, dT0, _, dc = _resid_one(jnp.asarray(xt), hj, d, None, True)
    dT_need = np.abs(np.asarray(dc)) / 0.05
    h["tmeas"] = np.asarray(dT_need - np.asarray(dT0))
    h["lr"] = np.sign(np.asarray(dc))
    hj = {k: jnp.asarray(v) for k, v in h.items()}
    r1 = np.asarray(_resid_one(jnp.asarray(xt), hj, d, None, True)[0])
    h["totdt"] = h["totdt"] - (-r1[:, 1])
    h["wdist"] = h["wdist"] + r1[:, 2] * np.where(h["esign"] * h["udir_dot_w"] < 0, -1.0, 1.0)
    if smear:
        h["tmeas"] = h["tmeas"] + rng.normal(0, 0.15 / 0.05, n)
        h["totdt"] = h["totdt"] + rng.normal(0, 2.0, n)
        h["wdist"] = h["wdist"] + rng.normal(0, 30.0, n)
    crs = {"mid": mid.copy(), "wdir": wdir.copy(), "valid": np.ones(n, bool)}
    return xt, h, crs, DRIFT
