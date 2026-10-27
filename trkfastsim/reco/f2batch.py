"""Caller-side fixed-shape F2 batches, without changing seeding or fit physics."""
from operator import index
from time import perf_counter

import numpy as np

from . import linefit


def _integer(value, name, minimum):
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _indices(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(value > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64 range")
    return value.astype(np.int64, copy=False)


def fit_seeds(sh, sd, cond, drift, nevents, niter=4, track_batch=64, min_hits=64):
    """Return raw good-fit statistics in seed order, including zero-fit events.

    Hit capacities are max(min_hits, nextpow2(seed hit count)); each fitter call
    has exactly track_batch tracks. Duplicate rows are discarded before cuts.
    Profile seconds include synchronized fitting, and shape_counts maps TxH to
    the number of calls (not a measurement of JAX compilations).
    """
    start = perf_counter()
    nevents = _integer(nevents, "nevents", 0)
    track_batch = _integer(track_batch, "track_batch", 1)
    min_hits = _integer(min_hits, "min_hits", 1)
    niter = _integer(niter, "niter", 0)
    ns = _indices(sd["n"], "seed n")
    ev = _indices(sd["ev"], "seed ev")
    member = _indices(sd["member"], "seed member")
    nseed = len(ns)
    if np.any(ns < 0) or np.any(ns > len(member)) or ns.sum() != len(member):
        raise ValueError("seed member length must equal the sum of nonnegative seed n")
    if len(ev) != nseed or np.any((ev < 0) | (ev >= nevents)):
        raise ValueError("seed ev must have one valid event index per seed")
    if np.any((member < 0) | (member >= len(sh["ev"]))):
        raise ValueError("seed member index is outside straw hits")
    params = [np.asarray(sd[k]) for k in ("a0", "b0", "a1", "b1", "t0")]
    if any(p.shape != (nseed,) for p in params):
        raise ValueError("seed parameters must have one value per seed")
    profile = {"pack_s": 0.0, "fit_s": 0.0, "reduce_s": 0.0, "total_s": 0.0,
               "blocks": 0, "tracks_padded": 0, "shape_counts": {},
               "track_batch": track_batch, "min_hits": min_hits}
    out = {"seeds": int(nseed), "fits_per_event": np.zeros(nevents, np.int64),
           "nactive": np.empty(0, np.int64), "chi2ndf": np.empty(0, np.float64), "profile": profile}
    if not nseed:
        profile["total_s"] = perf_counter() - start
        return out
    from . import linefit_jax

    tick = perf_counter()
    x0 = linefit.seed_params(*params)
    starts = np.cumsum(ns) - ns
    capacities = np.maximum(min_hits, np.left_shift(1, np.ceil(np.log2(np.maximum(ns, 1))).astype(np.int64)))
    good = np.zeros(nseed, bool)
    nactive = np.zeros(nseed, np.int64)
    chi2ndf = np.zeros(nseed, np.float64)
    profile["pack_s"] += perf_counter() - tick
    for capacity in np.unique(capacities):
        seeds = np.flatnonzero(capacities == capacity)
        for first in range(0, len(seeds), track_batch):
            tick = perf_counter()
            ids = seeds[first:first + track_batch]
            counts = ns[ids]
            nreal = len(ids)
            trk = np.repeat(np.arange(nreal), counts)
            offsets = np.arange(len(trk)) - np.repeat(np.cumsum(counts) - counts, counts)
            members = member[np.repeat(starts[ids], counts) + offsets]
            H = linefit.build_hits(trk, members, np.zeros(len(trk)), np.ones(len(trk), bool),
                                   sh, cond, nreal, nmax=int(capacity))
            rows = np.minimum(np.arange(track_batch), nreal - 1)
            H = {k: v[rows] for k, v in H.items()}
            xx = x0[ids][rows]
            profile["pack_s"] += perf_counter() - tick
            tick = perf_counter()
            f = linefit_jax.fit_own_states(xx, H, drift, niter=niter, n_pad=int(capacity), chunk=track_batch)
            f = {k: np.asarray(v)[:nreal] for k, v in f.items()}
            profile["fit_s"] += perf_counter() - tick
            tick = perf_counter()
            nact = f["active"].sum(1)
            keep = (nact >= 5) & np.isfinite(f["chi2"]) & (f["ndof"] >= 1)
            good[ids] = keep
            nactive[ids] = nact
            chi2ndf[ids[keep]] = f["chi2"][keep] / np.maximum(f["ndof"][keep], 1)
            shape = f"{track_batch}x{int(capacity)}"
            profile["shape_counts"][shape] = profile["shape_counts"].get(shape, 0) + 1
            profile["blocks"] += 1
            profile["tracks_padded"] += track_batch - nreal
            profile["reduce_s"] += perf_counter() - tick
    tick = perf_counter()
    out.update(fits_per_event=np.bincount(ev[good], minlength=nevents).astype(np.int64, copy=False),
               nactive=nactive[good], chi2ndf=chi2ndf[good])
    profile["reduce_s"] += perf_counter() - tick
    profile["total_s"] = perf_counter() - start
    return out
