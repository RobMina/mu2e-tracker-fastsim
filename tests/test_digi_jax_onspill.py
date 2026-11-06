"""Bounded CPU contracts for marker timing, per-end ghosts and field-dependent drift."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from trkfastsim import devfloat as df
from trkfastsim.digi import digi_jax as DJ, digi_clusters_jax as CJ
from trkfastsim.digi import digitize as DG, digicond as DC
from test_digi import cond, _step, _steps


@pytest.fixture(scope="module")
def onspill(cond):
    c = DC.Conditions(cond.s, cond.a)
    c.s.update(onspill=True, eventWindowLength=1695.)
    return c


def _timing(nev=1, length=1695., pbtime=0.):
    return {"pbtime": np.full(nev, pbtime), "event_length": np.full(nev, length)}


def test_field_drift_interpolates_radius_and_folded_phi():
    dd = np.arange(6) * .5
    dphi = np.pi / 8
    tt = dd[:, None] ** 2 * (10 + np.arange(5)[None, :] ** 2)
    phi = np.array([0., -.17, np.pi / 2, np.pi - .31, -3 * np.pi - .42, 2 * np.pi + .2])
    rho = np.array([0., .23, 1.12, 1.73, 2.51, 2.7])
    got = DJ._d2t(*map(jnp.asarray, (rho, phi, dd, tt)), .5, dphi)
    expected = DC.d2t(rho, phi, dd, tt, .5, dphi)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)
    assert expected[-1] != DC.d2t(rho[-1], 0., dd, tt, .5, dphi)


def test_charge_uses_wire_phi_not_first_table_column(cond):
    c = DC.Conditions(cond.s, cond.a)
    c.a["drift_times"] = c.drift_dist[:, None] * (10 + np.arange(c.drift_times.shape[1])[None, :])
    c.a["dtvar"] = np.zeros_like(c.dtvar)
    us = np.full(4, 520)
    pos = c.wire_mid[us] + np.array([[100., 0., 1.], [100., 1., 0.], [100., .5, .5], [100., -.5, -.5]])
    rho, zw, phi = DG.wire_coordinates(c, us, pos)
    expected = DC.d2t(rho, phi, c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi).astype(np.float32)
    prop = (np.stack([c.wire_halflen[us] - zw, c.wire_halflen[us] + zw], 1).astype(np.float32) / c.vprop).astype(np.float32)
    expected = (1000. + expected[:, None].astype(float) + prop.astype(float)).astype(np.float32)
    if df.SINGLE:
        pos -= c.straw_mid[us]
    got = DJ._charge_kernel(c)(jax.random.key(2), *map(jnp.asarray, df.dev((us, pos, np.ones(4, int), np.full(4, 1000.)))))
    np.testing.assert_allclose(got[0], expected, atol=2 * np.spacing(np.float32(1000)), rtol=0)
    assert np.ptp(np.asarray(got[0])[:, 0]) > 1


def test_onspill_periodic_inputs_and_explicit_phase(onspill):
    c = onspill
    st = _steps([_step(c, 520, t=1000.)])
    key = jax.random.key(3)
    a = DJ.digitize(st, 1, c, key, event_timing=_timing())
    assert len(a["sid"]) == 1
    for shift, pbtime in ((-3 * c.mbtime, 0.), (2 * c.mbtime - 17., 17.)):
        shifted = dict(st, t=st["t"] + shift)
        b = DJ.digitize(shifted, 1, c, key, event_timing=_timing(pbtime=pbtime))
        for field in a:
            np.testing.assert_array_equal(a[field], b[field])
    np.testing.assert_array_equal(a["step"], [[0, 0]])


@pytest.mark.parametrize("timing", [None, {}, {"pbtime": [0.], "event_length": [np.nan]},
    {"pbtime": [np.inf], "event_length": [1695.]}, {"pbtime": [], "event_length": [1695.]},
    {"pbtime": [0.], "event_length": [0.]}, {"pbtime": [0.], "event_length": [100.]}])
def test_onspill_timing_validation_even_when_empty(onspill, timing):
    st = {k: v[:0] for k, v in _steps([_step(onspill, 520)]).items()}
    with pytest.raises(ValueError):
        DJ.digitize(st, 1, onspill, jax.random.key(1), event_timing=timing)


@pytest.mark.skipif(jax.default_backend() != "cpu" or df.DRAW64, reason="CPU golden with native-precision draws")
def test_offspill_matches_main_golden(cond):
    # Main d7dbe3021cffc2172e23e141272f117c6b84bb4f; JAX/jaxlib 0.11.2, NumPy 2.4.6, arm64 CPU.
    # Unmodified test_digi fixture; float64 default or core float32, DRAW64=0, threefry2x32 seed 23.
    st = _steps([_step(cond, 520), _step(cond, 521, t=1800.), _step(cond, 92, t=100.),
                 _step(cond, 522, t=3000.)], ev=[0, 0, 2, 2])
    key = jax.random.key(23, impl="threefry2x32")
    if df.SINGLE:
        expected = {
            "adc": [[390, 390, 391, 391, 392, 454, 649, 635, 521, 441, 395, 370, 358, 354, 352],
                    [391, 390, 390, 391, 391, 475, 677, 579, 481, 420, 386, 367, 359, 356, 357],
                    [390, 390, 390, 391, 390, 459, 647, 586, 489, 426, 390, 370, 362, 359, 358],
                    [390, 390, 391, 390, 391, 430, 601, 610, 509, 439, 396, 374, 363, 359, 358]],
            "tdc": [[51908, 51958], [92833, 92912], [5887, 5922], [154378, 154430]],
            "tot": [[8, 8], [6, 6], [7, 7], [8, 8]],
            "pmp": [259, 287, 257, 220],
        }
        analog = [[1014.215124130249, 1015.1231098175049],
                  [1813.6762208938599, 1815.028738975525],
                  [114.52722930908203, 115.24673175811768],
                  [3015.097702026367, 3016.025552749634]]
    else:
        expected = {
            "adc": [[390, 391, 390, 390, 390, 430, 603, 574, 486, 427, 393, 374, 366, 362, 362],
                    [391, 390, 391, 390, 391, 425, 566, 529, 461, 417, 391, 377, 370, 368, 368],
                    [391, 390, 391, 391, 391, 432, 655, 607, 502, 433, 392, 371, 361, 357, 357],
                    [390, 390, 391, 390, 391, 436, 624, 583, 490, 427, 392, 372, 364, 361, 359]],
            "tdc": [[51984, 52050], [92982, 93004], [5926, 5977], [154464, 154486]],
            "tot": [[7, 7], [6, 7], [7, 7], [7, 7]],
            "pmp": [213, 176, 265, 234],
        }
        analog = [[1015.8975498682379, 1017.0410868042683],
                  [1816.4135725859994, 1816.9605714529237],
                  [115.69024435137072, 116.92312224528332],
                  [3016.156636495361, 3016.7840557792215]]
    expected.update(ev=[0, 0, 2, 2], sid=[680, 681, 92, 682], step=[[0, 0], [1, 1], [2, 2], [3, 3]])
    for timing in (None, _timing(nev=3, length=cond.eventWindowLength)):
        out = DJ.digitize(st, 3, cond, key, event_timing=timing)
        assert set(out) == set(expected) | {"t_analog"}
        for name, values in expected.items():
            assert out[name].dtype == np.dtype("int64"), name
            np.testing.assert_array_equal(out[name], values, err_msg=name)
        assert out["t_analog"].dtype == np.dtype("float64")
        # A 0.001 ns roundoff allowance is below one TDC tick; all digital outputs remain exact.
        np.testing.assert_allclose(out["t_analog"], analog, rtol=0, atol=1e-3)
    empty = {name: values[:0] for name, values in st.items()}
    for nev in (0, 3):
        out = DJ.digitize(empty, nev, cond, key)
        assert set(out) == set(expected) | {"t_analog"}
        for name, values in {**expected, "t_analog": analog}.items():
            dtype = np.float64 if name == "t_analog" else np.int64
            golden_empty = np.asarray(values, dtype=dtype)[:0]
            assert out[name].dtype == golden_empty.dtype, name
            np.testing.assert_array_equal(out[name], golden_empty, err_msg=name)


def _fixed_clusters(monkeypatch, c):
    monkeypatch.setattr(CJ, "make_clusters", lambda key, c, st, idx: (np.array([0, 1]), c.wire_mid[[520, 520]], np.ones(2, int)))
    gt = np.array([[100., 201.], [1600., 1500.]], np.float32) - (1000. if df.SINGLE else 0.)
    charge = np.array([.25, .5], np.float32)
    wd = np.array([[400., 600.], [450., 550.]], np.float32)
    monkeypatch.setattr(DJ, "_charge_blocks_kernel", lambda c: lambda *args: (gt, charge, wd))
    monkeypatch.setattr(DJ, "_panel_jitter", lambda key, ev, panel, scale: np.zeros(len(ev)))
    return _steps([_step(c, 520), _step(c, 520)])


def test_per_end_ghost_packing_copies_truth_charge_distance(onspill, monkeypatch):
    st = _fixed_clusters(monkeypatch, onspill)
    monkeypatch.setattr(DJ, "_waveform_batch", lambda cap: 1)
    seen = []

    def kernel(c, cap, pair_cap, batch):
        def run(key, t, q, wd, step, n, us, ra, sk, live, jitter, *extra, **kwargs):
            t, q, wd, step, n = map(np.asarray, (t, q, wd, step, n))
            if df.SINGLE:
                t = t + np.asarray(extra[0]).repeat(2)[:, None] * df.COMMON_PERIOD
            np.testing.assert_array_equal(n, [4, 3])
            np.testing.assert_array_equal(t[0], [-95., 100., 1600., 1795.])
            np.testing.assert_array_equal(t[1, :3], [-195., 201., 1500.])
            np.testing.assert_array_equal(step[0], [1, 0, 1, 0])
            np.testing.assert_array_equal(step[1, :3], [1, 0, 1])
            np.testing.assert_array_equal(q[0], [.5, .25, .5, .25])
            np.testing.assert_array_equal(q[1, :3], [.5, .25, .5])
            np.testing.assert_array_equal(wd[0], [450., 400., 450., 400.])
            np.testing.assert_array_equal(wd[1, :3], [550., 600., 550.])
            assert cap == 4
            seen.append(True)
            result = {name: np.zeros((1, pair_cap, 2)) for name in ("tdc", "tot", "step", "t_analog")}
            result.update(adc=np.zeros((1, pair_cap, c.nADCSamples)), pmp=np.zeros((1, pair_cap)),
                          keep=np.zeros((1, pair_cap), bool), failed=False, sat_overflow=False)
            return result
        return run

    monkeypatch.setattr(DJ, "_electronics_kernel", kernel)
    DJ.digitize(st, 1, onspill, jax.random.key(1), event_timing=_timing())
    assert seen == [True]


def test_ghosts_count_toward_waveform_capacity(onspill, monkeypatch):
    st = _fixed_clusters(monkeypatch, onspill)
    monkeypatch.setattr(DJ, "MAX_ROW_CLUSTERS", 3)
    with pytest.raises(OverflowError, match="MAX_ROW_CLUSTERS"):
        DJ.digitize(st, 1, onspill, jax.random.key(1), event_timing=_timing())


def test_per_event_tdc_end_and_readall_stock_rules(onspill):
    c = DC.Conditions(onspill.s, onspill.a)
    for name in ("strawNoise", "thresholdAnalogNoise", "TDCResolution", "adcAnalogNoise"):
        c.s[name] = 0.
    batch = 4
    t = np.full((2 * batch, 1), 1600.)
    args = (t, np.full_like(t, .2), np.full_like(t, 400.), np.zeros_like(t, int),
            np.ones(2 * batch, int), np.full(batch, 520), np.array([False, True, True, False]),
            np.arange(batch), np.ones(batch, bool), np.zeros(batch))
    ends = c.tdc_response(np.array([1695., 1695., 1550., 1550.]) - c.electronicsTimeDelay)
    result = DJ._electronics_kernel(c, 1, 4, batch)(jax.random.key(4), *map(jnp.asarray, args),
                                                  event_end_tdc=jnp.asarray(ends))
    np.testing.assert_array_equal(np.asarray(result["keep"]).sum(1), [0, 1, 0, 0])
    assert not bool(result["failed"])


def test_crossing_bound_uses_digitization_end_not_long_event(onspill):
    c = DC.Conditions(onspill.s, onspill.a)
    c.s["eventWindowLength"] = 5000.
    for name in ("strawNoise", "thresholdAnalogNoise", "TDCResolution", "adcAnalogNoise"):
        c.s[name] = 0.
    t = np.full((2, 1), 1800.)
    args = (t, np.full_like(t, .2), np.full_like(t, 400.), np.zeros_like(t, int),
            np.ones(2, int), np.array([520]), np.array([True]), np.array([0]), np.array([True]), np.zeros(1))
    result = DJ._electronics_kernel(c, 1, 4, 1)(jax.random.key(4), *map(jnp.asarray, args))
    assert int(result["pairs"]) == 0
    assert not np.asarray(result["keep"]).any()


def test_onspill_normal_tdc_limit_is_configured_end(onspill):
    c = DC.Conditions(onspill.s, onspill.a)
    for name in ("strawNoise", "thresholdAnalogNoise", "TDCResolution", "adcAnalogNoise"):
        c.s[name] = 0.
    t = np.full((4, 1), 1000.)
    args = (t, np.full_like(t, .2), np.full_like(t, 400.), np.zeros_like(t, int),
            np.ones(4, int), np.full(2, 520), np.zeros(2, bool), np.arange(2), np.ones(2, bool), np.zeros(2))
    ends = c.tdc_response(np.array([900., 1695.]) - c.electronicsTimeDelay)
    result = DJ._electronics_kernel(c, 1, 4, 2)(jax.random.key(4), *map(jnp.asarray, args),
                                              event_end_tdc=jnp.asarray(ends))
    np.testing.assert_array_equal(np.asarray(result["keep"]).sum(1), [1, 1])


def test_ghost_crossings_retain_original_step_truth(onspill, monkeypatch):
    c = DC.Conditions(onspill.s, onspill.a)
    for name in ("strawNoise", "thresholdAnalogNoise", "TDCResolution", "adcAnalogNoise"):
        c.s[name] = 0.
    st = _steps([_step(c, 92, t=100.), _step(c, 92, t=1500.)])
    monkeypatch.setattr(CJ, "make_clusters", lambda key, c, st, idx: (np.array([1]), c.wire_mid[[92]], np.ones(1, int)))
    gt = np.array([[1795., 1796.]], np.float32) - (1500. if df.SINGLE else 0.)
    monkeypatch.setattr(DJ, "_charge_blocks_kernel", lambda c: lambda *args: (gt, np.array([.2], np.float32), np.array([[400., 600.]], np.float32)))
    monkeypatch.setattr(DJ, "_panel_jitter", lambda key, ev, panel, scale: np.zeros(len(ev)))
    out = DJ.digitize(st, 1, c, jax.random.key(1), event_timing=_timing())
    np.testing.assert_array_equal(out["step"], [[1, 1]])
    assert np.all((out["t_analog"] > 95.) & (out["t_analog"] < 130.))


def test_step_selection_uses_onspill_end_with_readall_bypass(onspill, monkeypatch):
    c = DC.Conditions(onspill.s, onspill.a)
    c.s.update(eventWindowLength=5000., electronicsTimeDelay=200.)
    st = _steps([_step(c, 520, t=1650.), _step(c, 521, t=1200.), _step(c, 92, t=1650.)])
    selected = []

    def clusters(key, c, st, idx):
        selected.extend(idx.tolist())
        return np.array([], int), np.zeros((0, 3)), np.array([], int)

    monkeypatch.setattr(CJ, "make_clusters", clusters)
    DJ.digitize(st, 1, c, jax.random.key(1), event_timing=_timing(length=5000.))
    assert selected == [1, 2]
