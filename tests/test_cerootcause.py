"""CE root-cause diagnostics preserve zeros, strict tracker thresholds and exact event identities."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import cerootcause as cr


def keys(n=4):
    return pd.DataFrame({"run": np.uint64(1430), "subrun": np.uint64(100),
                         "event": np.arange(2**54, 2**54 + n, dtype=np.uint64)})


def test_migration_four_cells_and_conditioning_is_not_closure():
    g, f = [True, True, False, False], [True, False, True, False]
    r = cr.migration(g, f)
    assert [r[k] for k in ("both", "g4_only", "fast_only", "neither")] == [1, 1, 1, 1]
    assert r["paired_fast_minus_g4_error_n"][0] == 0
    r = cr.migration(np.array(g)[g], np.array(f)[g])
    assert r["paired_fast_minus_g4_error_n"][0] == -0.5
    assert cr.migration([], [])["events"] == 0


def test_conditioning_reports_both_directions_and_intersection():
    d = keys().assign(g4_steps=20, chain_steps=20, g4_tracker_term=[True, True, False, False],
                      chain_tracker_term=[True, False, True, False], g4_track=[True, True, False, False],
                      chain_track=[True, False, True, False])
    r = cr.migrations(d)["good_track_by_reference_condition"]
    assert r["g4_primary_tracker_term"]["paired_fast_minus_g4_error_n"][0] == -0.5
    assert r["fast_primary_tracker_term"]["paired_fast_minus_g4_error_n"][0] == 0.5
    assert r["both_primary_tracker_terms"]["events"] == 1
    assert r["either_primary_tracker_term"]["events"] == 3


def test_primary_tracker_term_strict_thresholds_and_missing_event_zero():
    d = keys(4)
    events = np.repeat(d.event.to_numpy(), [12, 12, 11, 12])
    s = pd.DataFrame({"run": np.uint64(1430), "subrun": np.uint64(100), "event": events,
                      "sp": 1, "px": 51.0, "py": 0.0, "pz": 0.0, "eion": 2e-6})
    s.loc[s.event == d.event.iloc[1], "px"] = 50.0
    s.loc[s.event == d.event.iloc[3], "eion"] = 1e-6
    r = cr.attach(keys(5), cr.step_flags(s), "steps")
    assert r.tracker_term.fillna(False).tolist() == [True, False, False, False, False]
    assert r.steps.iloc[:4].tolist() == [12, 12, 11, 12]
    assert np.isnan(r.steps.iloc[4])
    with pytest.raises(ValueError, match="primary"):
        cr.step_flags(s.assign(sp=2))


def test_endpoint_labels_do_not_infer_history():
    env = SimpleNamespace(axis=np.zeros(2), r=850.0, z0=-100.0, z1=100.0)
    f = pd.DataFrame({"x": [0, 0, 0, 0, 500, 900], "y": 0.0,
                      "z": [0, 0, -110, 110, -110, 0], "status": [1, 2, 0, 0, 0, 0]})
    assert cr.endpoint_labels(f, env, 380.0).tolist() == ["stopped", "path_or_chord_cap",
            "front_exit_inside_gas_inner_radius", "back_exit_inside_gas_inner_radius", "front_exit", "outer_exit"]


def test_attach_rejects_duplicate_and_outside_ids():
    d = keys()
    with pytest.raises(ValueError, match="duplicate"):
        cr.attach(d, pd.concat([d, d.iloc[:1]]), "duplicate")
    with pytest.raises(ValueError, match="outside"):
        cr.attach(d, keys(5), "extra")
    a = cr.attach(d, d.iloc[:1].assign(value=1), "subset")
    np.testing.assert_array_equal(a.event, d.event)
    assert a.value.notna().sum() == 1


def test_minimum_gas_radius_uses_finite_axis_segment():
    geom = SimpleNamespace(mid=np.array([[500.0, 0, 0], [0, 600.0, 0]]),
                           dir=np.array([[0, 1.0, 0], [1.0, 0, 0]]), halflen=np.array([50, 50]), radius=2.5)
    assert cr.gas_inner_radius(geom, np.zeros(2)) == 497.5
