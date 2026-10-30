"""Largest admitted waveform bucket retains complete saturated charge at bounded memory."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from trkfastsim import digi as DG, digi_jax as DJ
from trkfastsim.digicond import Conditions
from test_digi import cond


@pytest.mark.slow
def test_dense_limit_identical_clusters_vs_numpy(cond):
    import jax.numpy as jnp
    c = Conditions(cond.s, cond.a)
    for name in ("strawNoise", "thresholdAnalogNoise", "TDCResolution", "adcAnalogNoise"):
        c.s[name] = 0.
    cap = DJ.MAX_ROW_CLUSTERS
    t = np.broadcast_to(np.array([[1000.], [1001.]]), (2, cap)).copy()
    q = np.full((2, cap), .01)
    wd = np.broadcast_to(np.array([[400.], [600.]]), (2, cap)).copy()
    us = np.array([520])
    step = np.tile(np.arange(cap), (2, 1))
    wf = DG.Waveforms(c, np.repeat([0, 1], cap), t.ravel(), q.ravel(), wd.ravel(),
                      np.repeat(us, 2 * cap), step.ravel())
    ps, pt, pv, pi = DG.find_crossings(wf, np.random.default_rng(1), us, np.array([False]))
    result = DJ._electronics_kernel(c, cap, 4, 1)(jax.random.key(1),
        *map(jnp.asarray, (t, q, wd, step, np.array([cap, cap]), us, np.array([False]),
                          np.array([520]), np.array([True]), np.zeros(1))))
    assert not bool(result["failed"]) and not bool(result["sat_overflow"])
    keep = np.asarray(result["keep"])[0]
    assert keep.sum() == len(ps) == 1
    assert int(result["saturated_ends"]) == 2
    np.testing.assert_allclose(np.asarray(result["t_analog"])[0, keep], pt, atol=.04, rtol=0)
    np.testing.assert_allclose(np.asarray(result["tdc"])[0, keep], c.tdc_response(pt), atol=2, rtol=0)
    phase = np.maximum(0, np.ceil(pt[:, 0] / c.ADCPeriod).astype(int) - c.nADCPresamples)
    times = (phase[:, None] + np.arange(c.nADCSamples)) * c.ADCPeriod + c.ADCOffset
    volts = DG._adc_volts(wf, 2 * ps, times) + DG._adc_volts(wf, 2 * ps + 1, times)
    adc = np.clip(np.floor(volts / c.ADCLSB).astype(int) + c.adc_ped[us[ps], None], 0, c.maxADC)
    np.testing.assert_allclose(np.asarray(result["adc"])[0, keep], adc, atol=1, rtol=0)
