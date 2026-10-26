"""One state/time convention at arbitrary shared crossings of a candidate history."""
import numpy as np
import pytest

from trkfastsim import historystate
from trkfastsim.flighttime import linear_kinetic_dt_ns
from test_orderedtransport import metadata, run, intervals, deterministic


def test_crossings_on_both_sides_of_shared_boundaries(metadata, monkeypatch):
    deterministic(monkeypatch)
    result = run(metadata, intervals([(0, 0, 2, "vacuum"), (0, 2, 4, "G4_Al"), (0, 4, 5, "vacuum")]),
                 timing="linear-in-path", fractional_loss=1.)
    query = np.array([0., 1., 2., 2.5, 3., 4., 5., 5.1])
    state = historystate.at_path(result["history"], np.tile([1, 1, 1, 1], (len(query), 1)), query,
                                energy_evolution="linear-in-path")
    np.testing.assert_array_equal(state["available"], [True] * 7 + [False])
    np.testing.assert_allclose(state["ke_MeV"][:7], [20, 20, 20, 19.5, 19, 18, 18])
    np.testing.assert_allclose(state["position_mm"][:7, 0], query[:7])
    assert (np.diff(state["dt_from_start_ns"][:7]) > 0).all()
    expected = linear_kinetic_dt_ns(2, 20, 20) + linear_kinetic_dt_ns(2, 20, 18) + linear_kinetic_dt_ns(1, 18, 18)
    np.testing.assert_allclose(state["dt_from_start_ns"][6], expected)
    assert np.isnan(state["ke_MeV"][-1])


def test_terminal_endpoint_available_but_never_transmitted(metadata, monkeypatch):
    from mu2efastsim import eloss
    monkeypatch.setattr(eloss, "dedx_restricted", lambda m, p, c: np.ones_like(p))
    monkeypatch.setattr(eloss, "sample_loss", lambda r, m, p, l, c: np.full_like(p, 40.))
    result = run(metadata, intervals([(0, 0, 1, "G4_Al")]), mode="ordered-support-sampled-stop-v1",
                 fractional_loss=1., timing="linear-in-path")
    state = historystate.at_path(result["history"], np.tile([1, 1, 1, 1], (3, 1)), [.25, .5, .75],
                                energy_evolution="linear-in-path")
    np.testing.assert_array_equal(state["available"], [True, True, False])
    np.testing.assert_array_equal(state["alive"], [True, False, False])
    assert state["p_MeV_c"][1] == 0 and np.isfinite(state["dt_from_start_ns"][1])


def test_multi_muon_missing_and_zero_path_states(metadata):
    result = run(metadata, intervals([(0, 0, 1, "vacuum"), (1, 0, 2, "vacuum")]), ke=(5., 8., 0.), timing="linear-in-path")
    state = historystate.at_path(result["history"], [[1, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 3], [1, 1, 1, 9]],
                                [1., 1., 0., 0.], energy_evolution="linear-in-path")
    np.testing.assert_array_equal(state["available"], [True, True, True, False])
    np.testing.assert_array_equal(state["alive"], [True, True, False, False])
    np.testing.assert_allclose(state["ke_MeV"][:3], [8., 5., 0.])


def test_unknown_time_and_missing_assumption_are_refused(metadata):
    h = run(metadata, intervals([(0, 0, 1, "vacuum")]))["history"]
    with pytest.raises(ValueError, match="known"):
        historystate.at_path(h, [[1, 1, 1, 1]], [0.], energy_evolution="linear-in-path")
    with pytest.raises(ValueError, match="explicit"):
        historystate.at_path(h, [[1, 1, 1, 1]], [0.], energy_evolution="constant")


def test_empty_and_negative_paths(metadata):
    h = run(metadata, intervals([]), ke=())["history"]
    assert not historystate.at_path(h, [[1, 1, 1, 1]], [0.], energy_evolution="linear-in-path")["available"].any()
    result = historystate.at_path(h, np.empty((0, 4), int), [], energy_evolution="linear-in-path")
    assert result["position_mm"].shape == (0, 3)
    h = run(metadata, intervals([(0, 0, 1, "vacuum")]), timing="linear-in-path")["history"]
    assert not historystate.at_path(h, [[1, 1, 1, 1]], [-1.], energy_evolution="linear-in-path")["available"].any()
