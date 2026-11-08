"""Reco response on the device (JAX): table.apply_numpy for device arrays, one call per batch of events.

Per event with a tracker entry: a fit with probability eff_fit of its (nrow, tan dip) bin, then the quality pass with
eff_q, then the residuals and fit-quality outputs drawn by inverse CDF from the bin's tables and added to the event's
own entry truth, and the CE fiducial cuts on the smeared tan(dip), d0 and r_max. The seven drawn quantities are
correlated through the bin's Gaussian copula.
"""
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.special  # noqa: F401

from trkfastsim.recoresp import table as RT

jax.config.update("jax_enable_x64", True)


class Tables(NamedTuple):
    nrow_edges: object
    tandip_edges: object
    eff_fit: object
    eff_q: object
    icdf: object         # [K, ne, nt, NQ] in the order RT.RES + RT.QUAL
    chol: object         # [ne, nt, K, K] Gaussian-copula Cholesky factors
    cuts: object
    axis_xy: object
    bz: float
    icdf_dt0: object = None   # [ne, nt, NQ] t0 at the front - entry time, or None (older tables)


def to_device(tab, dtype=np.float64):
    dt = np.dtype(dtype)
    return Tables(jnp.asarray(tab["nrow_edges"], dt), jnp.asarray(tab["tandip_edges"], dt),
                  jnp.asarray(tab["eff_fit"], dt), jnp.asarray(tab["eff_q"], dt),
                  jnp.asarray(np.stack([tab[f"icdf_{k}"] for k in RT.RES + RT.QUAL]), dt), jnp.asarray(tab["chol"], dt),
                  jnp.asarray(tab["cuts"], dt), jnp.asarray(tab["axis_xy"], dt), float(tab["bz"]),
                  jnp.asarray(tab["icdf_dt0"], dt) if "icdf_dt0" in tab else None)


def entry_truth(x, p, axis_xy, bz, q=-1.0):
    """table.entry_truth on device arrays."""
    pt = jnp.hypot(p[:, 0], p[:, 1])
    R = pt / (RT.KB * jnp.abs(bz))
    s = jnp.sign(q * bz)
    ptn = jnp.maximum(pt, 1e-12)
    cx = x[:, 0] - axis_xy[0] + s * R * p[:, 1] / ptn
    cy = x[:, 1] - axis_xy[1] - s * R * p[:, 0] / ptn
    rc = jnp.hypot(cx, cy)
    return jnp.linalg.norm(p, axis=1), p[:, 2] / ptn, jnp.abs(rc - R), rc + R


def _bin(edges, v):
    return jnp.clip(jnp.searchsorted(edges, v, side="right", method="compare_all") - 1, 0, edges.shape[0] - 2)


@jax.jit
def apply(tb, key, has_entry, nrow, x, p, t=None):
    """Per event (device arrays): fit, qual, ce (bool) and p_true, p_reco, tandip, d0, rmax, fitcon, chi2ndf, nactive
    (NaN where no quality-passing track). x, p [n, 3]: entry state (Mu2e frame); nrow [n]: step rows of the event.
    t [n] (optional): entry time [ns]; with tables that hold it, adds t0 (track time at the front)."""
    dt = x.dtype
    ptrue, td, d0, rmax = entry_truth(x, p, tb.axis_xy, tb.bz)
    bn = _bin(tb.nrow_edges, nrow.astype(dt))
    bt = _bin(tb.tandip_edges, td)
    K = tb.icdf.shape[0]
    nq = tb.icdf.shape[-1]
    k1, k2 = jax.random.split(key)
    u = jax.random.uniform(k1, (2, x.shape[0]), dt)
    z = jnp.einsum("nij,nj->in", tb.chol[bn, bt], jax.random.normal(k2, (x.shape[0], K), dt))
    uc = jax.scipy.special.ndtr(z)
    fit = has_entry & (u[0] < tb.eff_fit[bn, bt])
    qual = fit & (u[1] < tb.eff_q[bn, bt])
    f = jnp.clip((nq - 1) * jnp.arccos(jnp.clip(1.0 - 2.0 * uc, -1.0, 1.0)) / jnp.pi, 0.0, nq - 1.000001)
    i = f.astype(jnp.int32)
    w = f - i
    kk = jnp.arange(K)[:, None]
    draw = tb.icdf[kk, bn[None], bt[None], i] * (1 - w) + tb.icdf[kk, bn[None], bt[None], i + 1] * w
    c = tb.cuts
    nan = jnp.asarray(jnp.nan, dt)
    obs = dict(p_reco=ptrue + draw[0], tandip=td + draw[1], d0=d0 + draw[2], rmax=rmax + draw[3],
               fitcon=draw[4], chi2ndf=draw[5], nactive=draw[6])
    ce = qual & (obs["tandip"] >= c[0]) & (obs["tandip"] <= c[1]) & (obs["d0"] >= c[2]) & (obs["d0"] <= c[3]) & \
        (obs["rmax"] >= c[4]) & (obs["rmax"] <= c[5])
    if t is not None and tb.icdf_dt0 is not None:   # own key: the draws above are unchanged
        ut = jax.random.uniform(jax.random.fold_in(key, 7), (x.shape[0],), dt)
        ft = jnp.clip((nq - 1) * jnp.arccos(jnp.clip(1.0 - 2.0 * ut, -1.0, 1.0)) / jnp.pi, 0.0, nq - 1.000001)
        it_ = ft.astype(jnp.int32)
        wt = ft - it_
        obs["t0"] = t + tb.icdf_dt0[bn, bt, it_] * (1 - wt) + tb.icdf_dt0[bn, bt, it_ + 1] * wt
    obs = {k: jnp.where(qual, v, nan) for k, v in obs.items()}
    return dict(fit=fit, qual=qual, ce=ce, p_true=jnp.where(has_entry, ptrue, nan), **obs)
