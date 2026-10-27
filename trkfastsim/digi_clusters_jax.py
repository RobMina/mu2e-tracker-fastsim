"""Explicit-key JAX ionisation clusters, with the distributions of :mod:`digi`.

Only shape selection, padding, compaction and minion electron expansion run on the
host. Device kernels use fixed-size step batches and power-of-two cluster buckets.
Random streams are folded by the original step index and cluster ordinal, so batch
boundaries, bucket membership and selection order do not change a step's draws.
Minion residual energy is exhausted by an uncapped, masked JAX while loop.

JAX is optional until a public function is called; this backend enables float64.
An allocation exceeding MAX_CLUSTER_BUCKET raises OverflowError, never truncates a
physics tail. Callers may raise that resource limit explicitly before retrying.
"""
from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax, random
except ImportError:
    jax = None
else:
    jax.config.update("jax_enable_x64", True)

__all__ = ["make_clusters", "cluster_gain", "wire_coordinates"]
BATCH_SIZE = 128
MAX_CLUSTER_BUCKET = 4096
MAX_DRIFT_CLUSTERS = 4_000_000


def _require_jax():
    if jax is None:
        raise ImportError("JAX is required for the JAX digitisation backend")


def _jit(fun=None, **kwargs):
    if fun is None:
        return partial(_jit, **kwargs)
    return jax.jit(fun, **kwargs) if jax is not None else fun


def _fold(keys, tag):
    return jax.vmap(lambda k: random.fold_in(k, tag))(keys)


def _step_keys(key, idx):
    lo = idx.astype(jnp.uint32)
    hi = (idx.astype(jnp.uint64) >> 32).astype(jnp.uint32)
    keys = jax.vmap(lambda i: random.fold_in(key, i))(lo)
    return jax.vmap(random.fold_in)(keys, hi)


def _uniform(keys):
    return jax.vmap(lambda k: random.uniform(k, (), dtype=jnp.float64))(keys)


def _poisson(keys, mean):
    return jax.vmap(lambda k, m: random.poisson(k, m, dtype=jnp.int64))(keys, mean)


@_jit
def _counts(key, idx, eion, length, single, minion, mean_free_path, eaverage, e1, maxnclu):
    keys = _step_keys(key, idx)
    nc = jnp.maximum(_poisson(_fold(keys, 0), length / mean_free_path), 1)
    nc = jnp.where(minion, nc, jnp.minimum(nc, maxnclu))
    nc = jnp.minimum(nc, jnp.floor(eion / e1).astype(jnp.int64))
    ne = jnp.maximum(_poisson(_fold(keys, 1), jnp.where(single, eion / eaverage, 0.0)), 1)
    return nc, ne


def _trunc_ne(keys, probabilities, energies, emax):
    """Exact inverse CDF of the reference's strictly energy-truncated draw.

    At emax == E1 the reference's fallback still returns one electron; preserving
    that convention also makes exactly integral energy budgets terminate.
    """
    kmax = jnp.searchsorted(energies, emax, side="left")
    u = _uniform(keys) * probabilities[jnp.maximum(kmax, 1) - 1]
    return jnp.searchsorted(probabilities, u, side="right") + 1


@_jit(static_argnames=("capacity",))
def _minion_clusters(keys, energies, probabilities, etot, nc, capacity):
    """Padded [batch, capacity] electron counts and final residual energies.

    nc == 0 denotes an inactive lane. Both the ordered initial fills and residual
    ionisations run in lock step, but every lane owns an independent key stream.
    No residual iteration limit or energy-tail approximation is used.
    """
    rows = jnp.arange(nc.size)
    ne = jnp.zeros((nc.size, capacity), jnp.int64)
    initial_keys = _fold(keys, 4)

    def fill_condition(state):
        i, _, _ = state
        return jnp.any(nc > i)

    def fill(state):
        i, ne, esum = state
        active = nc > i
        emax = etot - esum - (nc - i - 1) * energies[0]
        ie = _trunc_ne(_fold(initial_keys, i), probabilities, energies, emax)
        ne = ne.at[rows, i].set(jnp.where(active, ie, 0))
        esum += jnp.where(active, energies[jnp.minimum(ie, energies.size) - 1], 0.0)
        return i + 1, ne, esum

    _, ne, esum = lax.while_loop(fill_condition, fill, (jnp.int64(0), ne, jnp.zeros_like(etot)))
    residual = jnp.where(nc > 0, etot - esum, 0.0)

    def residual_condition(state):
        _, _, res = state
        return jnp.any((nc > 0) & (res >= energies[0]))

    def residual_fill(state):
        keys, ne, res = state
        parts = jax.vmap(lambda k: random.split(k, 3))(keys)
        active = (nc > 0) & (res >= energies[0])
        ie = _trunc_ne(parts[:, 1], probabilities, energies, res)
        cl = jnp.floor(_uniform(parts[:, 2]) * jnp.maximum(nc, 1)).astype(jnp.int64)
        cl = jnp.minimum(cl, jnp.maximum(nc - 1, 0))
        ne = ne.at[rows, cl].add(jnp.where(active, ie, 0))
        res -= jnp.where(active, energies[jnp.minimum(ie, energies.size) - 1], 0.0)
        return parts[:, 0], ne, res

    _, ne, residual = lax.while_loop(residual_condition, residual_fill, (_fold(keys, 5), ne, residual))
    return ne, residual


@_jit(static_argnames=("capacity",))
def _sample(key, idx, eion, nc, minion, p0, p1, sdir, width, eaverage, energies, probabilities, capacity):
    keys = _step_keys(key, idx)
    slots = jnp.arange(capacity, dtype=jnp.uint32)
    slot_keys = jax.vmap(lambda k: jax.vmap(lambda i: random.fold_in(k, i))(slots))(_fold(keys, 2))
    u = jax.vmap(_uniform)(slot_keys)
    path = p1 - p0
    q1 = jnp.cross(path, sdir)
    n1 = jnp.linalg.norm(q1, axis=1)
    q1 /= jnp.where(n1 > 0, n1, 1.0)[:, None]
    q2 = jnp.cross(path, q1)
    n2 = jnp.linalg.norm(q2, axis=1)
    q2 /= jnp.where(n2 > 0, n2, 1.0)[:, None]
    radial_keys = jax.vmap(lambda ks: _fold(ks, 0))(slot_keys)
    normals = jax.vmap(jax.vmap(lambda k: random.normal(k, (2,), dtype=jnp.float64)))(radial_keys)
    pos = p0[:, None, :] + u[:, :, None] * path[:, None, :]
    pos += width[:, None, None] * (q1[:, None, :] * normals[:, :, :1] + q2[:, None, :] * normals[:, :, 1:])
    electron_keys = jax.vmap(lambda ks: _fold(ks, 1))(slot_keys)
    means = jnp.broadcast_to(jnp.where(minion, 0.0, eion / (jnp.maximum(nc, 1) * eaverage))[:, None], nc.shape + (capacity,))
    ne = jnp.maximum(jax.vmap(_poisson)(electron_keys, means), 1)
    mine, _ = _minion_clusters(keys, energies, probabilities, eion, jnp.where(minion, nc, 0), capacity)
    ne = jnp.where(minion[:, None], mine, ne)
    return pos, ne


def _pad(a):
    """Zero-pad the leading dimension to the fixed device lane count."""
    return np.pad(a, ((0, BATCH_SIZE - len(a)),) + ((0, 0),) * (a.ndim - 1))


def make_clusters(key, c, st, idx):
    """Return NumPy (step_indices[int64], positions[float64,n,3], ne[int64]).

    Like digi.make_clusters, single clusters come first, then multi-cluster steps
    in idx order. Minion multi-clusters expand into one entry per drift electron;
    their positions repeat exactly. Geometry is indexed using st['_us'], not sid.
    Supply a JAX PRNG key, not a NumPy Generator. No key is implicitly consumed or
    returned: split keys at the caller when independent stages/calls are wanted.
    """
    _require_jax()
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("idx must be one-dimensional")
    if not idx.size:
        return np.empty(0, np.int64), np.empty((0, 3), np.float64), np.empty(0, np.int64)
    if np.any(idx < 0) or np.any(idx >= len(st["eion"])):
        raise IndexError("idx must contain original nonnegative step indices")
    eion = np.asarray(st["eion"][idx], np.float64)
    length = np.asarray(st["len"][idx], np.float64)
    energies, probabilities = np.asarray(c.EIonize, np.float64), np.asarray(c.intNProb, np.float64)
    if (not np.isfinite(eion).all() or not np.isfinite(length).all()
            or np.any(eion < 0) or np.any(length < 0)):
        raise ValueError("step energy and length must be finite and nonnegative")
    if (energies.ndim != 1 or not energies.size or not np.isfinite(energies).all()
            or energies[0] <= 0 or np.any(np.diff(energies) <= 0)
            or probabilities.ndim != 1 or len(probabilities) < len(energies)
            or not np.isfinite(probabilities).all() or probabilities[0] <= 0
            or np.any(np.diff(probabilities) < 0)
            or not np.isfinite(c.meanFreePath) or c.meanFreePath <= 0
            or not np.isfinite(c.EAverage) or c.EAverage <= 0):
        raise ValueError("invalid ionisation conditions")
    if (np.any(eion / energies[0] >= 2 ** 52) or np.any(eion / c.EAverage >= 2 ** 52)
            or np.any(length / c.meanFreePath >= 2 ** 52)):
        raise OverflowError("step exceeds safe float64/integer sampling range")
    single = (st["shape"][idx] == 3) | (length < c.meanFreePath)
    minion = st["ion"][idx] == 0
    p0 = np.stack([st[k][idx] for k in ("x0", "y0", "z0")], axis=1).astype(np.float64)
    p1 = np.stack([st[k][idx] for k in ("x1", "y1", "z1")], axis=1).astype(np.float64)
    nc, single_ne = np.empty(len(idx), np.int64), np.empty(len(idx), np.int64)
    for start in range(0, len(idx), BATCH_SIZE):
        sl = slice(start, start + BATCH_SIZE)
        counts, electrons = _counts(key, _pad(idx[sl]), _pad(eion[sl]), _pad(length[sl]),
                                    _pad(single[sl]), _pad(minion[sl]), c.meanFreePath,
                                    c.EAverage, energies[0], c.maxnclu)
        n = len(idx[sl])
        nc[sl], single_ne[sl] = np.asarray(counts)[:n], np.asarray(electrons)[:n]
    multi = np.flatnonzero(~single & (nc > 0))
    out_s, out_p, out_n = [idx[single]], [p0[single]], [single_ne[single]]
    if multi.size:
        if nc[multi].max() > MAX_CLUSTER_BUCKET:
            raise OverflowError(f"cluster count {nc[multi].max()} exceeds MAX_CLUSTER_BUCKET={MAX_CLUSTER_BUCKET}")
        capacities = 2 ** np.ceil(np.log2(nc[multi])).astype(np.int64)
        pieces_s, pieces_p, pieces_n = [], [], []
        for capacity in np.unique(capacities):
            group = multi[capacities == capacity]
            for start in range(0, len(group), BATCH_SIZE):
                m = group[start:start + BATCH_SIZE]
                sdir = np.asarray(c.straw_dir[st["_us"][idx[m]]], np.float64) if c.randrad else np.zeros((len(m), 3))
                width = np.asarray(st["width"][idx[m]], np.float64) if c.randrad else np.zeros(len(m))
                pos, ne = _sample(key, _pad(idx[m]), _pad(eion[m]), _pad(nc[m]), _pad(minion[m]),
                                  _pad(p0[m]), _pad(p1[m]), _pad(sdir), _pad(width),
                                  c.EAverage, energies, probabilities, int(capacity))
                keep = np.arange(capacity)[None, :] < nc[m, None]
                pieces_s.append(np.broadcast_to(m[:, None], keep.shape)[keep])
                pieces_p.append(np.asarray(pos)[:len(m)][keep])
                pieces_n.append(np.asarray(ne)[:len(m)][keep])
        local = np.concatenate(pieces_s)
        order = np.argsort(local, kind="stable")
        local, pos, ne = local[order], np.concatenate(pieces_p)[order], np.concatenate(pieces_n)[order]
        isminion = minion[local]
        repeats = np.where(isminion, ne, 1)
        if int(repeats.sum()) + int(single.sum()) > MAX_DRIFT_CLUSTERS:
            raise OverflowError(f"electron expansion exceeds MAX_DRIFT_CLUSTERS={MAX_DRIFT_CLUSTERS}")
        out_s.append(np.repeat(idx[local], repeats))
        out_p.append(np.repeat(pos, repeats, axis=0))
        out_n.append(np.repeat(np.where(isminion, 1, ne), repeats))
    return np.concatenate(out_s), np.concatenate(out_p), np.concatenate(out_n)


@_jit
def _gain(key, ne, a, ngauss, slope, gas_gain):
    kg, kn = random.split(key)
    shape = jnp.where(ne == 1, a, ne * (a + 1) - 1).astype(jnp.float64)
    gamma = random.gamma(kg, shape, dtype=jnp.float64) / shape
    normal = 1.0 + random.normal(kn, ne.shape, dtype=jnp.float64) * slope / jnp.sqrt(ne)
    return gas_gain * jnp.where(ne < ngauss, gamma, normal)


def cluster_gain(key, c, ne):
    """NumPy float64 gains: exact Polya/gamma or Gaussian, with no tail clipping.

    Unlike make_clusters, this standalone stage keys an entire input array; callers
    wanting per-step chunk invariance should keep the same gain-array partition.
    """
    _require_jax()
    ne = np.asarray(ne, np.int64)
    if ne.ndim != 1 or np.any(ne < 1):
        raise ValueError("ne must be a one-dimensional array of positive counts")
    return np.asarray(_gain(key, ne, c.polyaA, c.nGainGauss, c.gainRMSSlope, c.gasGain))


@_jit
def _wire(pos, smid, sdir, wmid, wdir, radius):
    d = pos - smid
    cp = d - jnp.sum(d * sdir, axis=1)[:, None] * sdir
    rho = jnp.linalg.norm(cp, axis=1)
    pos = jnp.where((rho > radius)[:, None], pos - cp / jnp.where(rho > 0, rho, 1)[:, None] * (rho - radius)[:, None], pos)
    d = pos - wmid
    dw = jnp.sum(d * wdir, axis=1)
    cp = d - dw[:, None] * wdir
    pdir = jnp.cross(wdir, jnp.array([0.0, 0.0, 1.0]))
    pdir *= jnp.where(jnp.sum(pdir * wmid, axis=1) < 0, -1.0, 1.0)[:, None]
    phi = jnp.arctan2(jnp.sum(cp * pdir, axis=1), cp[:, 2])
    return jnp.linalg.norm(cp, axis=1), dw, phi


def wire_coordinates(c, us, pos):
    """NumPy float64 (rho, along-wire z, phi), after straw-radius truncation."""
    _require_jax()
    us, pos = np.asarray(us, np.int64), np.asarray(pos, np.float64)
    return tuple(np.asarray(a) for a in _wire(pos, c.straw_mid[us], c.straw_dir[us],
                                             c.wire_mid[us], c.wire_dir[us], c.strawInnerRadius))
