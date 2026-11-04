"""SLAB_KERNEL: the one-call slab loss (mu2efastsim.slab_jax) equals the compacted slice-round loop to rounding (same
keys and draws; a separately compiled program rounds differently in the last bit), for the interior support slabs and
the world band loss, in its XLA implementation and the Pallas kernel (interpreter)."""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                                   # noqa: E402

pytest.importorskip("mu2efastsim.slab_jax")
from mu2efastsim import eloss_jax, materials as M                         # noqa: E402
from trkfastsim.sim import steps_jax as sj, world_jax as wj               # noqa: E402

MATS = (M.CONCRETE_MARS, M.MILD_STEEL, M.ALUMINIUM, M.POLYSTYRENE)


def _mc(mats):
    cs = [eloss_jax.consts(m) for m in mats]
    return eloss_jax.MatConsts(*(jnp.asarray([getattr(c, f) for c in cs]) for f in eloss_jax.MatConsts._fields))


def _case(R, seed, pmax=1e4):
    rng = np.random.default_rng(seed)
    col = rng.choice([0.0, 0.0, 2.0, 40.0, 400.0], (R, len(MATS))) * rng.uniform(0.5, 1.5, (R, len(MATS)))
    p = np.geomspace(80.0, pmax, R)[rng.permutation(R)]
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(seed), i))(jnp.arange(R, dtype=jnp.uint32))
    return jnp.asarray(col), jnp.asarray(p), keys


@pytest.fixture
def setting():
    old = sj.SLAB_KERNEL
    yield lambda v: setattr(sj, "SLAB_KERNEL", v)
    sj.SLAB_KERNEL = old


def test_interior_slab_kernel_equals_loop(setting):
    col, p, keys = _case(600, 1)
    from mu2efastsim import muradiative_jax as RJ
    dev = SimpleNamespace(MC=_mc(MATS), RT=RJ.stack(MATS))     # with the core default radiative losses
    setting("loop")
    ref = sj.DeviceInterior.slab(dev, col, p, keys)
    setting("xla")
    out = sj.DeviceInterior.slab(dev, col, p, keys)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-9, atol=1e-12)
    setting("interpret")
    np.testing.assert_allclose(np.asarray(sj.DeviceInterior.slab(dev, col[:64], p[:64], keys[:64])),
                               np.asarray(ref[:64]), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("impl", ["xla", "interpret"])
def test_world_path_equals_compacted_band_loop(impl):
    n = 640 if impl == "xla" else 64
    col, p, keys = _case(n, 2)
    rng = np.random.default_rng(3)
    bands = []
    for k, mats in enumerate((MATS[:2], MATS[2:], ())):
        mc = _mc(mats + (M.AIR,))
        coeff = jnp.asarray(rng.uniform(0.001, 0.01, (3, len(mats) + 1)))
        c = col[:, :len(mats)] if k != 1 else col[:, 2:]
        bands.append((mc, coeff, c, c.sum(1) + rng.uniform(0, 800.0, n)))
    lost = jnp.zeros(n)
    ref = []
    for k, (mc, coeff, c, seg) in enumerate(bands):
        lost, *rest = wj._compact_band_loss(mc, coeff, c, seg, p, lost, wj._fold(keys, k))
        ref.append([lost] + rest)
    ends = wj._world_path(tuple(b[0] for b in bands), tuple(b[2] for b in bands), tuple(b[3] for b in bands), p, keys,
                          impl)
    before = jnp.zeros(n)
    for k, (mc, coeff, c, seg) in enumerate(bands):
        out = [ends[k]] + list(wj._band_moments(coeff, c, seg, p, before, ends[k]))
        before = ends[k]
        for a, b in zip(ref[k], out):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("impl", ["xla", "interpret"])
def test_world_path_radiative_close_to_compacted_band_loop(impl):
    """With radiative losses the loop carries each segment's radiative loss into the next segment's ionisation; the
    one-call path does not (path_radiative): equal to well below a part in 1e3 of the loss."""
    from mu2efastsim import muradiative_jax as RJ
    n = 640 if impl == "xla" else 64
    col, p, keys = _case(n, 4, pmax=2e5)
    rng = np.random.default_rng(5)
    bands = []
    for k, mats in enumerate((MATS[:2], MATS[2:])):
        mats = mats + (M.AIR,)
        coeff = jnp.asarray(rng.uniform(0.001, 0.01, (3, len(mats))))
        c = col[:, :2] if k == 0 else col[:, 2:]
        bands.append((_mc(mats), coeff, c, c.sum(1) + rng.uniform(0, 800.0, n), RJ.stack(mats)))
    lost = jnp.zeros(n)
    for k, (mc, coeff, c, seg, rt) in enumerate(bands):
        lost, *_ = wj._compact_band_loss(mc, coeff, c, seg, p, lost, wj._fold(keys, k), rt)
    ion = jnp.zeros(n)
    for k, (mc, coeff, c, seg, rt) in enumerate(bands):
        ion, *_ = wj._compact_band_loss(mc, coeff, c, seg, p, ion, wj._fold(keys, k))
    ends = wj._world_path(tuple(b[0] for b in bands), tuple(b[2] for b in bands), tuple(b[3] for b in bands), p, keys,
                          impl, RTs=tuple(b[4] for b in bands))
    a, b, i = np.asarray(lost), np.asarray(ends[-1]), np.asarray(ion)
    assert (a >= i - 1e-6).all() and (a - i).max() > 0                    # the radiative losses are there
    assert np.abs(b - a).max() <= 1e-3 * np.maximum(a, 1.0).max()
