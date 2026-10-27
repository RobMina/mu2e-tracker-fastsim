"""JAX digitiser contracts; NumPy remains an untouched independent reference."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from trkfastsim import digi as DG
from trkfastsim import digi_jax as DJ
from test_digi import cond, _step, _steps


def test_repeatability_empty_and_truth(cond):
    st = _steps([_step(cond, 5 * 96 + 40), _step(cond, 5 * 96 + 41)], ev=[0, 2])
    a = DJ.digitize(st, 3, cond, jax.random.key(3))
    b = DJ.digitize(st, 3, cond, jax.random.key(3))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert np.array_equal(a["ev"], [0, 2])
    assert np.array_equal(a["step"], [[0, 0], [1, 1]])
    assert a["adc"].shape == (2, 15)
    empty = DJ.digitize({k: v[:0] for k, v in st.items()}, 3, cond, jax.random.key(3))
    assert empty["tdc"].shape == (0, 2) and empty["adc"].shape == (0, 15)


def test_selection_and_readall(cond):
    st = _steps([_step(cond, 700, eion=1e-6), _step(cond, 701, t=100), _step(cond, 92, t=100)])
    a = DJ.digitize(st, 1, cond, jax.random.key(1))
    assert np.array_equal(a["sid"], [DG.unique_straw_to_sid(92)])
    assert np.array_equal(a["step"], [[2, 2]])


def test_deadtime_and_pair_capacity(cond):
    st = _steps([_step(cond, 900, t=t) for t in (1000, 1200, 2000, 3000, 4000, 5000, 6000)])
    a = DJ.digitize(st, 1, cond, jax.random.key(2))
    assert len(a["sid"]) == 6
    assert np.all(np.diff(a["t_analog"][:, 0]) > 400)
    assert np.array_equal(a["step"][:, 0], [0, 2, 3, 4, 5, 6])


def test_nonminion_saturation(cond):
    st = _steps([_step(cond, 1000, eion=.04, ion=1)])
    stats = {}
    a = DJ.digitize(st, 1, cond, jax.random.key(5), stats)
    assert len(a["sid"]) == 1
    assert stats["saturated_ends"] == 2
    assert np.all(a["adc"] >= 0) and np.all(a["adc"] <= cond.maxADC)
    assert np.isfinite(a["t_analog"]).all()


def test_bad_population_and_key(cond):
    st = _steps([_step(cond, 1000)], ev=[1])
    with pytest.raises(ValueError, match="event index"):
        DJ.digitize(st, 1, cond, jax.random.key(1))
    with pytest.raises(TypeError):
        DJ.digitize(st, 2, cond, np.random.default_rng(1))


@pytest.mark.parametrize("charge", [.2, 5.])
def test_identical_clusters_electronics_vs_numpy(cond, charge):
    from trkfastsim.digicond import Conditions
    import jax.numpy as jnp
    c = Conditions(cond.s, cond.a)
    for name in ("strawNoise", "thresholdAnalogNoise", "TDCResolution", "adcAnalogNoise"):
        c.s[name] = 0.
    t = np.array([[1000., 1010., 2000., 2010.], [1001., 1011., 2001., 2011.]])
    q = np.full((2, 4), charge)
    wd = np.array([[400.] * 4, [600.] * 4])
    us = np.array([520])
    step = np.tile(np.arange(4), (2, 1))
    wf = DG.Waveforms(c, np.repeat([0, 1], 4), t.ravel(), q.ravel(), wd.ravel(),
                      np.repeat(us, 8), step.ravel())
    ps, pt, pv, pi = DG.find_crossings(wf, np.random.default_rng(1), us, np.array([False]))
    result = DJ._electronics_kernel(c, 4, 4, 1)(jax.random.key(1),
        *map(jnp.asarray, (t, q, wd, step, np.array([4, 4]), us, np.array([False]),
                          np.array([520]), np.array([True]), np.zeros(1))))
    assert not bool(result["failed"]) and not bool(result["sat_overflow"])
    mask = np.asarray(result["keep"])[0]
    got = {k: np.asarray(result[k])[0, mask] for k in ("tdc", "adc", "tot", "t_analog")}
    assert mask.sum() == len(ps) == 2
    np.testing.assert_allclose(got["t_analog"], pt, atol=.04, rtol=0)
    np.testing.assert_allclose(got["tdc"], c.tdc_response(pt), atol=2, rtol=0)
    tot = np.stack([DG._tot(wf, 2 * ps + e, pv[:, e], pt[:, e]) for e in (0, 1)], 1)
    np.testing.assert_array_equal(got["tot"], tot)
    phase = np.maximum(0, np.ceil(pt[:, 0] / c.ADCPeriod).astype(int) - c.nADCPresamples)
    times = (phase[:, None] + np.arange(c.nADCSamples)) * c.ADCPeriod + c.ADCOffset
    volts = DG._adc_volts(wf, 2 * ps, times) + DG._adc_volts(wf, 2 * ps + 1, times)
    adc = np.clip(np.floor(volts / c.ADCLSB).astype(int) + c.adc_ped[us[ps], None], 0, c.maxADC)
    np.testing.assert_allclose(got["adc"], adc, atol=1, rtol=0)


def test_nonfinite_input_and_waveform_capacity_fail_closed(cond, monkeypatch):
    st = _steps([_step(cond, 1000)])
    st["t"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite step"):
        DJ.digitize(st, 1, cond, jax.random.key(1))
    st["t"][0] = 1000.
    monkeypatch.setattr(DJ, "MAX_ROW_CLUSTERS", 1)
    with pytest.raises(OverflowError, match="MAX_ROW_CLUSTERS"):
        DJ.digitize(st, 1, cond, jax.random.key(1))
