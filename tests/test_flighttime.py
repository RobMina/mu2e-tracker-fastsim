"""Synthetic timing tests; no production transport, stopping-law or G4 validation."""
import numpy as np
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import transportledger as ledger
from trkfastsim.flighttime import C_MM_PER_NS, history_dt_ns, linear_kinetic_dt_ns


@pytest.fixture
def meta():
    return ledger.provenance(
        geometry_sha256="0" * 64, tracker_commit="1" * 40, core_commit="2" * 40,
        inputs=[{"name": "synthetic-timing-fixture", "sha256": "3" * 64}],
        frame="synthetic Cartesian", settings={"mode": "synthetic timing only"},
        media=[ledger.medium(), ledger.medium(materials.ALUMINIUM, cut_MeV=1e9,
                                               loss_basis="unrestricted")])


def make_history(meta, *, stopped=False):
    kin = np.array([20., 20., 8., 8.]) if not stopped else np.zeros(4)
    kout = np.array([20., 8., 8., 0.]) if not stopped else np.zeros(4)
    length = np.array([10., 3., 0., 2.]) if not stopped else np.zeros(4)
    entry = np.array([[0., 0., 0.], [10., 0., 0.], [13., 0., 0.], [13., 1., 0.]])
    exit_ = np.array([[10., 0., 0.], [13., 0., 0.], [13., 1., 0.], [15., 1., 0.]])
    if stopped:
        entry = exit_ = np.zeros((4, 3))
    known = np.array(["known"] * 4)
    return ledger.SegmentHistory(dict(
        run=np.ones(4, dtype=int), subrun=np.ones(4, dtype=int), event=np.ones(4, dtype=int),
        sp=np.arange(4) if stopped else np.zeros(4, dtype=int),
        segment=np.zeros(4, dtype=int) if stopped else np.arange(4),
        medium=np.zeros(4, dtype=int) if stopped else np.array([0, 1, -1, 1]),
        kind=["physical"] * 4 if stopped else ["physical", "physical", "displacement", "physical"],
        terminal=["exhausted"] * 4 if stopped else ["none", "none", "none", "exhausted"],
        path_mm=length, entry_mm=entry, exit_mm=exit_, ke_in_MeV=kin, ke_out_MeV=kout,
        sampled_loss_MeV=kin - kout, applied_loss_MeV=kin - kout,
        deposited_MeV=kin - kout, escaping_MeV=np.zeros(4), sample_status=known,
        deposit_status=known, escape_status=known,
        dt_ns=np.zeros(4) if stopped else np.array([np.nan, np.nan, 0., np.nan]),
        dt_status=known if stopped else ["unknown", "unmodeled", "known", "unknown"]), meta)


@pytest.mark.parametrize("momentum", [0.001, 1., 100., 500., 1e6])
def test_constant_momentum_vacuum(momentum):
    energy = np.hypot(momentum, eloss.MMU)
    kinetic = momentum * (momentum / (energy + eloss.MMU))
    length = np.array([0., 1., 1000.])
    expected = length * energy / (C_MM_PER_NS * momentum)
    np.testing.assert_allclose(linear_kinetic_dt_ns(length, kinetic, kinetic), expected, rtol=3e-15)


@pytest.mark.parametrize("kinetic", [1e-280, 1e-12, 0.01, 10., 1000.])
def test_analytic_stopping(kinetic):
    length = 2.5
    momentum = np.sqrt(kinetic) * np.sqrt(kinetic + 2 * eloss.MMU)
    expected = (length / C_MM_PER_NS) * (momentum / kinetic)
    actual = linear_kinetic_dt_ns(length, kinetic, 0.)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=3e-15)


@pytest.mark.parametrize("kout", [0., 0.001, 19.999999999999996, 20.])
@pytest.mark.parametrize("parts", [1, 2, 17, 1000])
def test_linear_kinetic_subdivision_is_exact(kout, parts):
    length, kin = 37., 20.
    endpoints = np.linspace(kin, kout, parts + 1)
    increments = linear_kinetic_dt_ns(length / parts, endpoints[:-1], endpoints[1:])
    np.testing.assert_allclose(increments.sum(), linear_kinetic_dt_ns(length, kin, kout), rtol=3e-15)


def test_numerical_path_integration_converges():
    exact = linear_kinetic_dt_ns(23., 15., 2.)
    errors = []
    for parts in (10, 100, 1000):
        kinetic = 15. - 13. * (np.arange(parts) + 0.5) / parts
        inverse_beta = (kinetic + eloss.MMU) / np.sqrt(kinetic * (kinetic + 2 * eloss.MMU))
        errors.append(abs(23. / C_MM_PER_NS * inverse_beta.mean() - exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-6 * exact


def test_vectorized_broadcast_and_piecewise_sum():
    length = np.array([[2.], [5.]])
    kin, kout = np.array([20., 7., 0.01]), np.array([20., 2., 0.])
    actual = linear_kinetic_dt_ns(length, kin, kout)
    assert actual.shape == (2, 3)
    p0 = np.sqrt(kin * (kin + 2 * eloss.MMU))
    p1 = np.sqrt(kout * (kout + 2 * eloss.MMU))
    expected = length * (kin + kout + 2 * eloss.MMU) / (C_MM_PER_NS * (p0 + p1))
    np.testing.assert_allclose(actual, expected, rtol=3e-15)
    np.testing.assert_allclose(actual.sum(axis=1), expected.sum(axis=1), rtol=3e-15)


def test_empty_zero_path_and_stopped_lanes():
    assert linear_kinetic_dt_ns([], [], []).shape == (0,)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(linear_kinetic_dt_ns(0., [0., 4., 5.], [0., 4., 0.]), 0.)
        assert linear_kinetic_dt_ns(0., 0., 0.) == 0.
        mixed = linear_kinetic_dt_ns([0., 1., 0.], [0., 3., 3.], [0., 0., 3.])
    assert mixed[0] == mixed[2] == 0. and mixed[1] > 0.


@pytest.mark.parametrize("args", [
    (-1., 1., 1.), (1., -1., 0.), (1., 1., -1.), (1., 1., 2.), (0., 1., 2.),
    (1., 0., 0.), (np.nan, 1., 1.), (np.inf, 1., 1.), (1., np.nan, 0.),
    (1., np.inf, 0.), (1., 1., np.nan), (1., 1., np.inf), (0., np.nan, 0.),
    ([0., 1.], [0., 0.], [0., 0.]),
])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        linear_kinetic_dt_ns(*args)


def test_incompatible_shapes():
    with pytest.raises(ValueError):
        linear_kinetic_dt_ns(np.ones(2), np.ones(3), 0.)


def test_history_explicit_assumption_piecewise_and_no_mutation(meta):
    history = make_history(meta)
    before = {name: history[name].copy() for name in ledger.FIELDS}
    metadata = history.metadata
    actual = history_dt_ns(history, energy_evolution="linear-in-path")
    e = 20. + eloss.MMU
    p20 = np.sqrt(20. * (20. + 2 * eloss.MMU))
    p8 = np.sqrt(8. * (8. + 2 * eloss.MMU))
    expected = np.array([10. * e / p20, 3. * (p20 - p8) / 12., 0., 2. * p8 / 8.]) / C_MM_PER_NS
    np.testing.assert_allclose(actual, expected, rtol=3e-15)
    np.testing.assert_allclose(actual.sum(), expected.sum(), rtol=3e-15)
    actual[:] = -1.
    for name in ledger.FIELDS:
        np.testing.assert_equal(history[name], before[name])
    assert history.metadata == metadata


@pytest.mark.parametrize("declaration", [None, False, "unknown", "linear-in-time", "constant"])
def test_history_rejects_unspecified_or_unsupported_evolution(meta, declaration):
    with pytest.raises(ValueError, match="explicit"):
        history_dt_ns(make_history(meta), energy_evolution=declaration)


def test_history_requires_declaration_and_history_type(meta):
    with pytest.raises(TypeError):
        history_dt_ns(make_history(meta))
    with pytest.raises(TypeError, match="SegmentHistory"):
        history_dt_ns({}, energy_evolution="linear-in-path")


def test_empty_and_all_stopped_histories(meta):
    empty = history_dt_ns(ledger.SegmentHistory.empty(meta), energy_evolution="linear-in-path")
    assert empty.shape == (0,) and empty.sum() == 0.
    with np.errstate(all="raise"):
        stopped = history_dt_ns(make_history(meta, stopped=True), energy_evolution="linear-in-path")
    np.testing.assert_array_equal(stopped, np.zeros(4))


def test_subnormal_path_does_not_underflow_before_inverse_beta():
    length, kinetic = 1e-323, 1e-300
    expected = (length * np.sqrt(eloss.MMU / (2 * kinetic))) / C_MM_PER_NS
    actual = linear_kinetic_dt_ns(length, kinetic, kinetic)
    assert actual > 0.
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0.)


@pytest.mark.parametrize("length,kinetic", [(np.finfo(float).max, 1e-300), (np.nextafter(0., 1.), 20.)])
def test_unrepresentable_positive_time_is_refused(length, kinetic):
    with pytest.raises(ValueError, match="representable"):
        linear_kinetic_dt_ns(length, kinetic, kinetic)
