"""Synthetic initial-CAD population/probe contracts; no native evidence or jobs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_initial_cad_audit as D
from test_f1_captured_ca_population import sample


CAD = [400, 800000, 50, 50, "TOT:LongVal", "Null:Inactive", "", 0]
PROBE_OUTPUTS = {"rdrift": "rdrift", "velocity": "dvel", "sderr": "sderr", "uderr": "uderr"}


def synthetic_drift(times):
    times = np.asarray(times)
    return dict(rdrift=.5 + .001*times, velocity=.02 + .00001*times,
                sderr=.2 + .00002*times, uderr=.3 + .00003*times)


@pytest.fixture
def population(sample, monkeypatch):
    s = sample
    s.qseed[:, 3] = s.qref[:, 3] = np.pi/2
    mid = np.array([[250., 180., s.qref[0, 2]+2.], [290., 170., s.qref[1, 2]+3.]])
    wire = np.array([[.8, .6, 0.], [-.8, -.6, 0.]])
    s.measured = dict(early=mid+500*wire, late=mid-500*wire,
                      measurement_time=np.array([73003., 73010.]), corrected_time=s.corrected)
    s.speed = np.array([180., 220.])
    s.line = D.A.sensor(s.measured["early"], s.measured["late"], s.measured["measurement_time"], s.speed)
    s.built = D.C.solve(s.qseed, s.line, *D.C.constructor_hint(s.qseed, s.line, s.corrected, s.mass),
                        precision=D.NATIVE_CA_PRECISION, mass=s.mass)
    s.state = D.C.solve(s.qref, s.line, s.built["particle_time"], s.built["sensor_time"],
                        precision=D.NATIVE_CA_PRECISION, mass=s.mass)
    assert np.all(np.abs(s.state["particle_poca"][:, 2]-s.line["mid"][:, 2]) <= 100.)
    s.constructor_time = s.built["particle_time"].copy()
    s.H = D.fixed_state_rows(s.qref, s.line, s.state, s.mass)["H"]
    s.covariance = np.broadcast_to(np.diag([1., .001, 2., .001, 3., 4.]), (2, 6, 6)).copy()
    updates = dict(particle_poca=np.column_stack((s.state["particle_poca"], s.state["particle_time"])),
                   sensor_poca=np.column_stack((s.state["sensor_poca"], s.state["sensor_time"])),
                   particle_direction=s.state["direction"], sensor_direction=s.state["sensor_direction"],
                   doca_variance=np.sum(s.H[:, 0]**2*np.diagonal(s.covariance, axis1=1, axis2=2), axis=1))
    for field, value in updates.items():
        s.rows = ak.with_field(s.rows, value, field)
    s.table, s.config, s.policy, s.drift_calls = object(), object(), {"cad": CAD.copy()}, []

    def drift(times, table, config):
        assert table is s.table and config is s.config
        assert times.shape == (2,)
        s.drift_calls.append(times.copy())
        return synthetic_drift(times)

    monkeypatch.setattr(D, "drift_info", drift)
    return s


def compare(s):
    return D.compare_population(s.native, s.rows, s.qseed, s.qref, s.covariance, s.mass,
                                s.constructor_time, s.measured, s.speed, s.table, s.config, s.policy)


def replace_second(table, field, value):
    values = D.array(table[field]).copy()
    values[1] = value
    return ak.with_field(table, values, field)


def test_population_all_arms_full_keys_retained_hints_and_independent_drift(population, monkeypatch):
    s = population
    s.rows = s.rows[::-1]
    original, calls = D.C.solve, []

    def solve(q, line, particle_hint, sensor_hint, *, precision, mass):
        assert len(q) == 2 and mass is s.mass
        assert precision == D.NATIVE_CA_PRECISION == float(np.float32(1e-4))
        if not calls:
            assert q is s.qseed
        else:
            prior = s.built if len(calls) == 1 else s.state
            np.testing.assert_array_equal(particle_hint, prior["particle_time"])
            np.testing.assert_array_equal(sensor_hint, prior["sensor_time"])
            if len(calls) == 1:
                assert q is s.qref
        calls.append(q.copy())
        return original(q, line, particle_hint, sensor_hint, precision=precision, mass=mass)

    monkeypatch.setattr(D.C, "solve", solve)
    monkeypatch.setattr(ak, "to_list", lambda *a, **k: pytest.fail("no per-hit record conversion"))
    result = compare(s)
    assert len(calls) == 5 and len(s.drift_calls) == 3
    assert result["n_hits"] == result["n_targets"] == 2 and result["n_excluded_hits"] == 0
    assert result["hit_keys"] == D.keys(s.native, D.HIT_KEYS).tolist()
    assert result["hit_key_fields"] == list(D.HIT_KEYS)
    assert result["constructor_time"]["exact"]["passed"]
    assert all(m["exact"]["passed"] for m in result["reference_geometry"].values())
    assert result["reference_doca_variance"]["criterion"]["passed"]
    assert tuple(result["arms"]) == D.ARMS
    for index, (arm, item) in enumerate(result["arms"].items()):
        generated = item["reconstructed_unbiased"]
        np.testing.assert_array_equal(calls[index+2], generated["parameters"])
        np.testing.assert_allclose(generated["parameters"], s.qref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(generated["covariance"], s.covariance, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(generated["delta_time"], s.drift_calls[index])
        for field, expected in synthetic_drift(s.drift_calls[index]).items():
            np.testing.assert_array_equal(generated["drift"][field], expected)
        assert item["cad_outputs"]["state"] == [0, 0]
        assert item["cad_outputs"]["active"] == [True, True]
        assert item["state_agreement"] and not item["native_unbiased_intermediates_compared"]
        assert item["inversion"]["n_inversions"] == (0 if arm == "ideal_zero_weight_control" else 4)
        if arm == "ideal_zero_weight_control":
            for field in D.P.STATE_FIELDS:
                np.testing.assert_array_equal(generated["state"][field], s.state[field])
    json.dumps(result, allow_nan=False)


def test_capture_state_flags_and_row_weights_cannot_drive_generated_outputs(population):
    s = population
    baseline = compare(s)
    changed = dict(particle_poca=[101., 202., 303., 400.], sensor_poca=[-11., -22., -33., 500.],
                   particle_direction=[.3, .4, .5], sensor_direction=[.6, .7, .8], ca_usable=False,
                   doca_variance=1e8, doca=900., hit_state=-2, active=False, constrain_tot=False,
                   constrain_long=False, constrain_absdrift=True, row_active=[0, 0, 0],
                   row_derivative=np.full(18, 77.), row_value=[99., 88., 77.],
                   row_mvar=[1e20, 2e20, 3e20], variance_scale=1e15,
                   row_pvar=[1., 2., 3.], dDdP=np.full(6, 33.), dTdP=np.full(6, 44.))
    for field, value in changed.items():
        s.rows = replace_second(s.rows, field, value)
    for field, value in dict(time=-123., weight_matrix=np.full(36, 1e25),
                             weight_vector=np.full(6, -1e25)).items():
        s.native = replace_second(s.native, field, value)
    result = compare(s)
    assert result["n_hits"] == result["n_targets"] == 2 and result["n_excluded_hits"] == 0
    assert result["hit_keys"] == baseline["hit_keys"]
    failing_key = result["hit_keys"][1:]
    for field in ("particle_time", "sensor_time", "particle_poca", "sensor_poca", "direction",
                  "sensor_direction", "usable"):
        metric = result["reference_geometry"][field]["criterion"]
        assert not metric["passed"] and metric["n_failing_hits"] == 1
        assert metric["failing_hit_keys"] == failing_key
    assert result["reference_doca_variance"]["criterion"]["failing_hit_keys"] == failing_key
    for arm, item in result["arms"].items():
        for field in ("inversion", "reference_roundtrip", "reconstructed_unbiased", "cad_outputs", "cad_cuts", "margins"):
            assert item[field] == baseline["arms"][arm][field]
        assert not item["state_agreement"]
        for metric in item["state_vs_capture"].values():
            assert not metric["exact"]["passed"]
            assert metric["exact"]["failing_hit_keys"] == failing_key
            assert metric["exact"]["per_hit_failed_components"] == [0, 1]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("missing", [False, True])
def test_population_join_never_drops_missing_or_duplicate_capture(population, monkeypatch, missing):
    s = population
    s.rows = s.rows[:1] if missing else ak.concatenate([s.rows, s.rows[:1]])
    monkeypatch.setattr(D.C, "solve", lambda *a, **k: pytest.fail("identity guard must precede solves"))
    with pytest.raises(ValueError, match="missing full identity|duplicate"):
        compare(s)


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_population_unusable_constructor_reference_or_unbiased_aborts(population, monkeypatch, stage):
    original, calls = D.C.solve, []

    def solve(*args, **kwargs):
        state = original(*args, **kwargs)
        if len(calls) == stage:
            state["usable"][1] = False
        calls.append(1)
        return state

    monkeypatch.setattr(D.C, "solve", solve)
    with pytest.raises(ValueError, match="unusable.*(constructor|reference|unbiased)"):
        compare(population)
    assert len(calls) == stage+1 and not population.drift_calls


def test_population_retry_branch_aborts_instead_of_excluding_lane(population, monkeypatch):
    original, calls = D.C.solve, []

    def solve(*args, **kwargs):
        state = original(*args, **kwargs)
        if len(calls) == 1:
            state["particle_poca"][1, 2] = population.line["mid"][1, 2]+101.
        calls.append(1)
        return state

    monkeypatch.setattr(D.C, "solve", solve)
    with pytest.raises(ValueError, match="unsupported initial hit retry"):
        compare(population)
    assert len(calls) == 2 and not population.drift_calls


@pytest.fixture
def probes(monkeypatch):
    times = np.linspace(-100., 1900., 2001)
    generated = synthetic_drift(times)
    saved = {stored: generated[name][None, :].copy() for name, stored in PROBE_OUTPUTS.items()}
    saved.update(dtgrid=times[None, :], strawOuterRadius=np.array([2.5]))
    s = SimpleNamespace(saved=saved, times=times, table=object(), config=object(), calls=[])

    def drift(dt, table, config):
        assert table is s.table and config is s.config
        np.testing.assert_array_equal(dt, s.times)
        s.calls.append(dt.copy())
        return synthetic_drift(dt)

    monkeypatch.setattr(D, "drift_info", drift)
    return s


@pytest.mark.parametrize("field", list(PROBE_OUTPUTS))
def test_all_2001_matching_probes_pass_and_last_probe_mismatch_is_reported(probes, field):
    s = probes
    result = D.probe_comparison(s.saved, s.table, s.config)
    assert result["passed"] and result["n_probes"] == 2001 and len(s.calls) == 1
    assert result["range"] == [-100., 1900.]
    for item in result["metrics"].values():
        assert item["criterion"]["passed"] and item["exact"]["passed"]
        assert item["criterion"]["n_components"] == 2001
    s.saved[PROBE_OUTPUTS[field]][0, -1] += 1.
    result = D.probe_comparison(s.saved, s.table, s.config)
    assert not result["passed"] and result["n_probes"] == 2001
    for name, item in result["metrics"].items():
        assert item["criterion"]["passed"] == (name != field)
    metric = result["metrics"][field]["criterion"]
    assert metric["mismatches"] == 1 and metric["first_mismatch_indices"] == [[2000]]
    json.dumps(result, allow_nan=False)


def test_probe_requires_every_field_one_row_and_all_2001_values(probes):
    s = probes
    for field in D.PROBE_FIELDS:
        for rows in (0, 2):
            bad = deepcopy(s.saved)
            bad[field] = np.repeat(bad[field], rows, axis=0)
            with pytest.raises(ValueError, match="one calibration probe row"):
                D.probe_comparison(bad, s.table, s.config)
        bad = deepcopy(s.saved)
        del bad[field]
        with pytest.raises((ValueError, KeyError)):
            D.probe_comparison(bad, s.table, s.config)
    assert not s.calls
    for field in ("dtgrid", *PROBE_OUTPUTS.values()):
        for count in (2000, 2002):
            bad = deepcopy(s.saved)
            bad[field] = np.zeros((1, count))
            with pytest.raises(ValueError, match="2001 native calibration probes|invalid native drift probe"):
                D.probe_comparison(bad, s.table, s.config)


def test_probe_masks_cannot_hide_any_native_probe(probes):
    s = probes
    for field in ("dtgrid", *PROBE_OUTPUTS.values()):
        bad = deepcopy(s.saved)
        bad[field] = np.ma.array(bad[field], mask=np.zeros((1, 2001), dtype=bool))
        bad[field].mask[0, -1] = True
        with pytest.raises(ValueError, match="missing"):
            D.probe_comparison(bad, s.table, s.config)


def test_probe_nonfinite_values_fail_in_every_native_field(probes):
    s = probes
    for field in ("dtgrid", *PROBE_OUTPUTS.values()):
        for value in (np.nan, np.inf, -np.inf):
            bad = deepcopy(s.saved)
            bad[field][0, -1] = value
            with pytest.raises(ValueError, match="2001 native calibration probes|invalid native drift probe"):
                D.probe_comparison(bad, s.table, s.config)
