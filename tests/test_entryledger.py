"""Opt-in cumulative entry accounting on fixed trajectories; legacy behavior is retained."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import entryledger, steps
from test_step_energy_ledger import generated, empty_interior, plane_state, _geometry, _momentum, P_START


@pytest.mark.parametrize("mode", ["energy", "vacuum"])
def test_energy_closure_support_air_and_stop(mode):
    p = np.array([80., 300., 0.])
    support = np.array([2., 5., 1.])
    distance = np.array([1000., 2000., 0.])
    out = entryledger.evaluate(p, support, distance, np.maximum(p - support, 0.), mode=mode)
    air = eloss.dedx_restricted(materials.AIR, p, 1e9) * distance
    expected = np.maximum(np.hypot(p, eloss.MMU) - eloss.MMU - support - (air if mode == "energy" else 0.), 0.)
    np.testing.assert_allclose(out["ke_written_MeV"], expected, atol=1e-12)
    np.testing.assert_allclose(out["energy_defect_MeV"], 0., atol=1e-12)
    assert not out["above_stop_threshold"][-1]
    assert out["p_written_MeV_c"][-1] == 0.


def test_empty_and_invalid_inputs():
    out = entryledger.evaluate([], [], [], [], mode="energy")
    assert all(len(v) == 0 for v in out.values())
    assert not len(entryledger.keep_before_stop([], [], []))
    for p in ([-1.], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            entryledger.evaluate(p, [0.], [0.], [1.])
    with pytest.raises(ValueError, match="unknown"):
        entryledger.evaluate([], [], [], [], mode="typo")


def test_stop_is_absorbing_even_if_later_entry_energy_recovers():
    keep = entryledger.keep_before_stop([0, 1, 0, 0, 1], [30., 10., 10., 20., 20.], [True, True, True, False, True])
    np.testing.assert_array_equal(keep, [False, True, True, False, True])


@pytest.mark.parametrize("interior_msc", [False, True])
def test_capture_is_inert_and_opt_in_air_energy_closes(generated, empty_interior, interior_msc):
    kwargs = dict(interior=empty_interior, interior_msc=interior_msc, exit_scatter=False, msc_tail=False)
    a, pa = steps.make_steps(generated, _geometry(), np.random.default_rng(19), **kwargs)
    b, pb = steps.make_steps(generated, _geometry(), np.random.default_rng(19), capture_entry_ledger=True, **kwargs)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(pa, pb, check_exact=True)
    legacy_ledger = steps.make_steps.entry_ledger.copy()
    assert (legacy_ledger.energy_defect_MeV > 0.).all()
    c, _ = steps.make_steps(generated, _geometry(), np.random.default_rng(19), entry_ledger="energy",
                           capture_entry_ledger=True, **kwargs)
    ledger = steps.make_steps.entry_ledger
    np.testing.assert_allclose(ledger.energy_defect_MeV, 0., atol=1e-12)
    np.testing.assert_allclose(np.hypot(_momentum(c), eloss.MMU) - eloss.MMU, ledger.ke_requested_MeV, atol=1e-12)
    assert (_momentum(c) < _momentum(b)).all()
    np.testing.assert_array_equal(a[["x0", "y0", "z0", "x1", "y1", "z1"]], c[["x0", "y0", "z0", "x1", "y1", "z1"]])
    np.testing.assert_array_equal(a.t, c.t)
    assert steps.make_steps.path_vertices is None


@pytest.mark.parametrize("interior_msc", [False, True])
def test_vacuum_writer_matches_air_disabled_control(generated, empty_interior, plane_state, interior_msc):
    plane_state(P_START)
    kwargs = dict(stage="C", interior=empty_interior, interior_msc=interior_msc, exit_scatter=False, msc_tail=False)
    a, _ = steps.make_steps(generated, _geometry(), np.random.default_rng(19), air_loss=False, **kwargs)
    b, _ = steps.make_steps(generated, _geometry(), np.random.default_rng(19), entry_ledger="vacuum",
                           capture_entry_ledger=True, **kwargs)
    np.testing.assert_allclose(_momentum(a), _momentum(b), atol=1e-12)
    assert (steps.make_steps.entry_ledger.air_estimate_MeV > 0.).all()
    assert (steps.make_steps.entry_ledger.air_debit_MeV == 0.).all()
    np.testing.assert_allclose(steps.make_steps.entry_ledger.energy_defect_MeV, 0., atol=1e-12)


@pytest.mark.parametrize("interior_msc", [False, True])
def test_energy_mode_drops_rows_after_overloss(generated, empty_interior, monkeypatch, interior_msc):
    original = eloss.dedx_restricted
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut, *args: np.ones_like(p) if mat == materials.AIR else original(mat, p, cut, *args))
    rows, _ = steps.make_steps(generated, _geometry(), np.random.default_rng(19), interior=empty_interior,
                               interior_msc=interior_msc, exit_scatter=False, msc_tail=False,
                               entry_ledger="energy", capture_entry_ledger=True)
    assert rows.empty
    assert not steps.make_steps.entry_ledger.emitted.any()
    assert (steps.make_steps.entry_ledger.p_written_MeV_c == 0.).all()


def test_diagnostic_configuration_rejected_before_transport(generated):
    for kwargs, message in (({"entry_ledger": "energy"}, "exit_scatter=False"),
                            ({"entry_ledger": "vacuum", "exit_scatter": False}, "stage C"),
                            ({"entry_ledger": "other"}, "unknown"),
                            ({"entry_ledger": "energy", "exit_scatter": False, "sampler": object()}, "primary-only")):
        with pytest.raises(ValueError, match=message):
            steps.make_steps(generated, _geometry(), np.random.default_rng(19), **kwargs)
