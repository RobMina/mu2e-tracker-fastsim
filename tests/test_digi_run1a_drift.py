"""Run1a field-on drift and off-spill contracts, retaining the field-free fixtures.

Stock StrawDrift::D2T folds phi in double, stores the result in a float, and
interpolates double tables. The port retains double folded phi: this is a known
floating approximation, not a bitwise stock match. Independent scalar-oracle
checks measure the maximum radius/angle-grid error and require <1e-4 ns, including
JAX; no stock-precision discrepancy is skipped or treated as an expected failure.
"""
import math

import numpy as np
import pytest

from test_digi import FIX, HL, _step, _steps, cond
from trkfastsim.digi import digicond as DC
from trkfastsim.digi import digitize as DG

try:
    import jax
    import jax.numpy as jnp
    from trkfastsim.digi import digi_jax as DJ
except ImportError:
    jax = None

BACKENDS = ["numpy", pytest.param("jax", marks=pytest.mark.skipif(jax is None, reason="optional JAX backend"))]


TOL_NS = 1e-4
BZ_T = 1.23456789


def _stock_d2t(distance, phi, distances, times, delta_d, delta_phi):
    """Scalar Offline/TrackerConditions/src/StrawDrift.cc D2T, not either port."""
    distance, phi = float(distance), float(phi)
    folded = abs(math.fmod(phi, math.pi))
    if folded > math.pi / 2:
        folded = math.pi - folded
    folded = float(np.float32(folded))
    lower_phi = max(0, math.floor(folded / delta_phi))
    upper_phi = min(times.shape[1] - 1, math.ceil(folded / delta_phi))
    radial = min(len(distances) - 2, max(0, math.floor(distance / delta_d)))
    fraction = (distance - float(distances[radial])) / delta_d
    lower = float(times[radial, lower_phi]) + fraction * (
        float(times[radial + 1, lower_phi]) - float(times[radial, lower_phi]))
    upper = float(times[radial, upper_phi]) + fraction * (
        float(times[radial + 1, upper_phi]) - float(times[radial, upper_phi]))
    if phi == 0:
        return lower
    return lower + (folded - lower_phi * delta_phi) / delta_phi * (upper - lower)


@pytest.fixture(scope="module")
def drift_tables(cond):
    cfg = DC.parse_fhicl_dump(FIX.read_text())["services"]["ProditionsService"]["strawDrift"]
    return {bz: DC.straw_drift(cfg, cond.wireRadius, cond.strawOuterRadius, bz=bz)
            for bz in (0.0, BZ_T, -BZ_T)}


@pytest.fixture(scope="module")
def field_on(cond, drift_tables):
    c = DC.Conditions(cond.s, cond.a)
    table = drift_tables[BZ_T]
    c.a.update({name: table[name] for name in ("drift_dist", "drift_times")})
    c.s.update(field_bz_T=BZ_T, drift_dD=table["drift_dD"], drift_dphi=table["drift_dphi"])
    c.s["sources"] = dict(cond.sources, field_bz_T=BZ_T, bFieldFile="test-origin-field")
    assert not c.onspill and c.eventWindowLength == 100000.0
    assert np.any(c.drift_times[:, -1] > c.drift_times[:, 0])
    return c


@pytest.fixture(scope="module")
def oracle_grid(field_on):
    c = field_on
    radii = np.unique(np.r_[np.linspace(0, c.strawOuterRadius, 129), -c.drift_dD,
                            c.wireRadius, c.strawInnerRadius, c.drift_dD,
                            c.drift_dist[-1] - c.drift_dD / 2, c.drift_dist[-1],
                            c.drift_dist[-1] + c.drift_dD / 2, c.strawOuterRadius + c.drift_dD])
    knots = np.arange(c.drift_times.shape[1]) * c.drift_dphi
    near_knots = np.r_[knots, knots - 1e-7, knots + 1e-7]
    angles = np.unique(np.r_[np.linspace(-4 * math.pi, 4 * math.pi, 193),
                             near_knots, -near_knots, math.pi - near_knots,
                             -math.pi + near_knots, -0.0])
    rho, phi = (x.ravel() for x in np.meshgrid(radii, angles, indexing="ij"))
    expected = np.array([_stock_d2t(r, p, c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi)
                         for r, p in zip(rho, phi)])
    return rho, phi, expected


def _check_stock_error(label, actual, expected):
    actual = np.asarray(actual)
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    error = float(np.max(np.abs(actual - expected)))
    print(f"{label}: {expected.size} radius/angle points, max stock D2T error = {error:.12g} ns")
    assert error < TOL_NS, f"{label}: max stock D2T error {error:.12g} ns >= {TOL_NS} ns"


def test_no_field_identity(cond, drift_tables):
    table = drift_tables[0.0]
    np.testing.assert_array_equal(table["drift_dist"], cond.drift_dist)
    np.testing.assert_array_equal(table["drift_times"], cond.drift_times)
    np.testing.assert_array_equal(table["drift_times"],
                                  np.broadcast_to(table["drift_times"][:, :1], cond.drift_times.shape))
    rho = np.linspace(0, cond.strawOuterRadius, 67)
    for phi in (0.0, -0.37, 0.37, math.pi / 2, -math.pi, 3 * math.pi + 0.2):
        actual = DC.d2t(rho, phi, cond.drift_dist, cond.drift_times, cond.drift_dD, cond.drift_dphi)
        expected = [_stock_d2t(r, phi, cond.drift_dist, cond.drift_times, cond.drift_dD,
                               cond.drift_dphi) for r in rho]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=TOL_NS)
        np.testing.assert_array_equal(actual, DC.d2t(rho, 0, cond.drift_dist, cond.drift_times,
                                                    cond.drift_dD, cond.drift_dphi))


def test_field_signed_phi_and_b_reversal(cond, drift_tables, field_on):
    positive, negative = drift_tables[BZ_T], drift_tables[-BZ_T]
    for name in positive:
        np.testing.assert_array_equal(positive[name], negative[name])
    np.testing.assert_array_equal(positive["drift_times"][:, 0], cond.drift_times[:, 0])
    assert np.all(positive["drift_times"][1:, -1] > positive["drift_times"][1:, 0])
    c = field_on
    phi = np.linspace(0, math.pi / 2, 51)
    expected = np.array([_stock_d2t(2.4, p, c.drift_dist, c.drift_times,
                                    c.drift_dD, c.drift_dphi) for p in phi])
    for angles in (phi, -phi, math.pi - phi, phi + 2 * math.pi, -phi - 3 * math.pi):
        actual = DC.d2t(2.4, angles, c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=TOL_NS)
    assert expected[0] < expected[len(phi) // 2] < expected[-1]


def test_numpy_d2t_stock_mixed_precision_grid(field_on, oracle_grid):
    """Bound the known double-folded-phi approximation, rather than requiring bitwise equality."""
    c = field_on
    rho, phi, expected = oracle_grid
    actual = DC.d2t(rho, phi, c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi)
    _check_stock_error("NumPy float64", actual, expected)


@pytest.mark.skipif(jax is None, reason="optional JAX backend")
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_jax_d2t_same_stock_oracle(field_on, oracle_grid, dtype):
    """Both JAX precisions use the independent stock oracle with an absolute ns budget."""
    c = field_on
    rho, phi, expected = oracle_grid
    arguments = [jnp.asarray(x, dtype=dtype)
                 for x in (rho, phi, c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi)]
    actual = jax.jit(DJ._d2t)(*arguments)
    _check_stock_error(f"JAX {np.dtype(dtype).name}", actual, expected)


@pytest.fixture(scope="module", params=BACKENDS)
def offspill_digis(field_on, request):
    c = field_on
    rows = [_step(c, 5 * 96 + 40, zpos=120.0, t=1000.0),
            _step(c, 5 * 96 + 41, zpos=-120.0, t=10000.0)]
    st = _steps(rows, ev=[0, 1])
    timing = {"pbtime": np.zeros(2), "event_length": np.full(2, c.eventWindowLength)}
    if request.param == "numpy":
        default = DG.digitize(st, 2, c, np.random.default_rng(3))
        explicit = DG.digitize(st, 2, c, np.random.default_rng(3), event_timing=timing)
    else:
        default = DJ.digitize(st, 2, c, jax.random.key(3))
        explicit = DJ.digitize(st, 2, c, jax.random.key(3), event_timing=timing)
    return c, st, default, explicit


@pytest.mark.slow
def test_field_on_offspill_default_timing_is_identity(offspill_digis):
    c, st, default, explicit = offspill_digis
    assert len(default["sid"]) == 2
    assert default.keys() == explicit.keys()
    for name in default:
        np.testing.assert_array_equal(default[name], explicit[name], err_msg=name)
    np.testing.assert_array_equal(default["ev"], st["ev"])
    np.testing.assert_array_equal(default["sid"], st["sid"])
    np.testing.assert_array_equal(default["step"], [[0, 0], [1, 1]])
    assert default["adc"].shape == (2, c.nADCSamples)
    assert np.all(default["pmp"] > 20)


@pytest.mark.pr_params("numpy")   # jax digitiser per PR: stock golden and d2t oracle; field-on jax runs nightly
def test_field_on_offspill_straw_endpoints_and_no_time_folding(offspill_digis):
    """End ordering follows propagation; noisy, unequal end responses also cause threshold time walk."""
    c, st, result, _ = offspill_digis
    assert len(result["sid"]) == 2
    z = np.array([120.0, -120.0])
    times = result["tdc"] * c.TDCLSB
    difference = times[:, 1] - times[:, 0]
    assert np.all(difference * z > 0)
    np.testing.assert_allclose(difference, 2 * z / c.vprop, rtol=0, atol=1.5)
    propagation = np.stack([HL - z, HL + z], axis=1) / c.vprop
    residual = times - st["t"][:, None] - propagation
    assert np.all((residual > 0) & (residual < 100.0))
    assert np.all(times[1] > c.digitizationEnd)
    assert np.all(result["tdc"] > c.digitizationStartTDC)
    assert np.all((result["tot"] >= 1) & (result["tot"] <= c.maxTOT))


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_field_on_offspill_event_window_endpoints(field_on, backend):
    """Off-spill hits obey their event length, not the shorter on-spill digitization end."""
    c = field_on
    times = [-200.0, 100.0, 1000.0, 4000.0, 5100.0, 10000.0, 100010.0]
    st = _steps([_step(c, 520, t=t) for t in times], ev=np.arange(len(times)))
    lengths = np.array([5000.0] * 5 + [c.eventWindowLength] * 2)
    timing = {"pbtime": np.zeros(len(times)), "event_length": lengths}
    if backend == "numpy":
        result = DG.digitize(st, len(times), c, np.random.default_rng(3), event_timing=timing)
    else:
        result = DJ.digitize(st, len(times), c, jax.random.key(3), event_timing=timing)
    np.testing.assert_array_equal(result["ev"], [2, 3, 5])
    np.testing.assert_array_equal(result["step"], [[2, 2], [3, 3], [5, 5]])
    assert np.all(result["tdc"] < c.tdc_response(lengths[result["ev"]] - c.electronicsTimeDelay)[:, None])
    assert np.all(result["tdc"] > c.digitizationStartTDC)
