"""digi_resident: the device-resident packing gives exactly digi_jax's digis for the same steps and key."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from trkfastsim import devfloat as df  # noqa: E402
from trkfastsim.digi import digi_jax as DJ, digi_resident as DR  # noqa: E402
from trkfastsim.digi import digicond as DC  # noqa: E402
from test_digi import cond, _step, _steps  # noqa: E402,F401

pytestmark = pytest.mark.skipif(df.SINGLE, reason="digi_resident: float64 tracker device float only")


def _random_steps(c, rng, nev=6, per_event=40):
    """Several events, repeated straws (several passages per straw), minion and plain ionisation, early and late times."""
    rows, ev = [], []
    straws = rng.choice(np.arange(500, 900), size=30, replace=False)
    for e in range(nev):
        for _ in range(per_event):
            us = int(rng.choice(straws))
            rows.append(_step(c, us, b=float(rng.uniform(-2.3, 2.3)), zpos=float(rng.uniform(-300, 300)),
                              t=float(rng.uniform(300, 1650)), eion=float(rng.uniform(2e-4, 6e-3)),
                              ion=int(rng.uniform() < 0.8)))
            ev.append(e)
    st = _steps(rows, ev)
    st["ion"] = 1 - st["ion"]          # mostly minion (ion == 0), as fast electrons
    return st


def _same(a, b):
    assert set(a) == set(b)
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.shape == y.shape and x.dtype == y.dtype, k
        assert np.array_equal(x, y, equal_nan=True), k


def test_offspill_digis_equal_digi_jax(cond):
    c = cond
    st = _random_steps(c, np.random.default_rng(1))
    key = jax.random.key(5)
    a = DJ.digitize(st, 6, c, key)
    b = DR.digitize({k: jax.numpy.asarray(v) for k, v in st.items()}, 6, c, key)
    assert len(a["ev"]) > 20
    _same(a, b)



def test_padded_steps_and_padded_output_give_the_same_digis(cond):
    """nstep= (steps already in the pad_steps layout) and padded=True (power-of-two output plus count) change only
    where the arrays are cut, not the digis."""
    c = cond
    st = {k: jax.numpy.asarray(v) for k, v in _random_steps(c, np.random.default_rng(1)).items()}
    key = jax.random.key(5)
    a = DR.digitize(st, 6, c, key)
    n = len(st["ev"])
    sp = DR.pad_steps(st, n)
    assert sp["ev"].shape[0] == DR._b(n) and not np.asarray(sp["t"][n:]).any()
    out, cnt = DR.digitize(sp, 6, c, key, nstep=n, padded=True)
    assert cnt == len(a["ev"]) and out["ev"].shape[0] == DR._b(cnt)
    _same(a, {k: np.asarray(v)[:cnt] for k, v in out.items()})
    with pytest.raises(ValueError):
        DR.digitize(st, 6, c, key, nstep=n)          # not padded


def test_batches_without_host_reads_equal_the_per_batch_loop(cond, monkeypatch):
    """ASYNC_BATCHES (offset and flags on the device, one host read) gives the per-batch loop's digis and stats."""
    c = cond
    st = {k: jax.numpy.asarray(v) for k, v in _random_steps(c, np.random.default_rng(1)).items()}
    key = jax.random.key(5)
    sa, sb = {}, {}
    a = DR.digitize(st, 6, c, key, stats=sa)
    monkeypatch.setattr(DR, "ASYNC_BATCHES", False)
    b = DR.digitize(st, 6, c, key, stats=sb)
    _same(a, b)
    assert sa == sb and sa["jax_buckets"] > 1


def test_a_failed_batch_falls_back_to_the_per_batch_loop(cond, monkeypatch):
    """A batch whose threshold search fails at pair_cap 4 sends the call to the per-batch loop (which retries it at a
    larger pair_cap): the same digis as that loop."""
    c = cond
    st = {k: jax.numpy.asarray(v) for k, v in _random_steps(c, np.random.default_rng(1)).items()}
    key = jax.random.key(5)
    orig = DJ._electronics_kernel

    def failing(c_, cap, pair_cap, batch):
        k = orig(c_, cap, pair_cap, batch)
        if pair_cap != 4 or 2 * cap <= 4:
            return k

        def run(*a, **kw):
            r = dict(k(*a, **kw))
            r["failed"] = jax.numpy.asarray(True)
            return r
        return run
    monkeypatch.setattr(DJ, "_electronics_kernel", failing)
    n0 = DR.FALLBACKS[0]
    a = DR.digitize(st, 6, c, key)
    assert DR.FALLBACKS[0] == n0 + 1
    monkeypatch.setattr(DR, "ASYNC_BATCHES", False)
    _same(a, DR.digitize(st, 6, c, key))


@pytest.mark.slow   # second full compile of both digitisers; per PR the off-spill case checks the same packing
def test_onspill_digis_with_ghosts_equal_digi_jax(cond):
    c = DC.Conditions(cond.s, cond.a)
    c.s.update(onspill=True, eventWindowLength=1695.)
    rng = np.random.default_rng(2)
    st = _random_steps(c, rng)
    st["t"] = st["t"] + rng.choice([0.0, 1695.0, -10.0, 1690.0], size=len(st["t"]))   # ghosts at both ends
    timing = {"pbtime": rng.uniform(0, 20, 6), "event_length": np.full(6, 1695.)}
    key = jax.random.key(6)
    a = DJ.digitize(st, 6, c, key, event_timing=timing)
    b = DR.digitize({k: jax.numpy.asarray(v) for k, v in st.items()}, 6, c, key, event_timing=timing)
    assert len(a["ev"]) > 20
    _same(a, b)


def test_no_steps_in_the_window_gives_the_empty_layout(cond):
    c = cond
    st = _steps([_step(c, 520, eion=0.0)])
    b = DR.digitize({k: jax.numpy.asarray(v) for k, v in st.items()}, 1, c, jax.random.key(1))
    a = DJ._empty(c)
    for k in a:
        assert np.asarray(b[k]).shape == a[k].shape
