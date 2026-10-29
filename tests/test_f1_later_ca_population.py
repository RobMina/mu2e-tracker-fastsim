"""Synthetic adjacent-hit population: real CA generation, captured comparison only.

No capture files, native builds, fits or remote IO. Only joins/measurement loading
are replaced; hit hints, piece search, hit-z retry and comparisons remain real.
"""
from copy import deepcopy
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_later_ca_audit as D


def replace(table, field, lane, value):
    values = ak.to_list(table[field])
    values[lane] = value
    return ak.with_field(table, ak.Array(values), field)


@pytest.fixture
def population(monkeypatch):
    def build(prior=(-3, 1, -3, 1), give_up=(False, True, True, False)):
        prior, give_up = np.asarray(prior), np.asarray(give_up)
        n = len(prior)
        par = np.zeros((3, 2, 6))
        par[1, :, 1], par[1, :, 3] = np.pi / 2, np.pi / 4
        par[1, :, 4], par[1, :, 5] = [.5, 3.5], 2000.
        ranges = np.zeros((3, 2, 2))
        ranges[1] = [[-2., 2.], [2., 5.]]
        counts = np.array([0, 2, 0])  # Empty owners exercise compaction.
        parent = np.ones(n, dtype=int)
        measured = dict(early=np.tile([100., 0., 1.], (n, 1)),
                        late=np.tile([-100., 0., 1.], (n, 1)),
                        measurement_time=np.full(n, 4.), energy=np.full(n, .001),
                        dtime=np.full(n, .25))
        line = D.A.sensor(measured["early"], measured["late"],
                          measured["measurement_time"], np.full(n, 200.))
        mass = np.full(n, 105.6583755)
        straw_z = np.where(give_up, 250., .5)
        old = D.L.piecewise(par[1:2], ranges[1:2], counts[1:2],
                            np.zeros(n, int), line, np.full(n, .5),
                            np.full(n, 3.5), mass=mass)
        real_update = D.hit_update
        _, final, policy = real_update(par[1:2], ranges[1:2], counts[1:2],
                                        np.zeros(n, int), line, old,
                                        measured["dtime"], straw_z, mass)
        np.testing.assert_array_equal(policy["give_up"], give_up)

        before, after, rows0, rows1, seeds = [], [], [], [], []
        for lane in range(n):
            # Repeated event/seed/source indices across runs require full keys.
            key = dict(run=1 + lane, subrun=2, event=3, iseed=4)
            raw_sid = ((lane // 96) << 7) | (lane % 96)
            common = dict(key, input_index=7, shindex=0, strawid=raw_sid)
            before.append(dict(common, iteration_ordinal=0, effect_index=11,
                               time=float(old["particle_time"][lane])))
            index = final["index"][lane]
            after.append(dict(common, iteration_ordinal=1, effect_index=22,
                              time=float(final["particle_time"][lane]),
                              reference_range=ranges[1, index].tolist(),
                              reference_parameters=par[1, index].tolist()))
            for state, rows, hit_state in ((old, rows0, prior[lane]),
                    (final, rows1, -3 if give_up[lane] else prior[lane])):
                rows.append(dict(particle_poca=[*state["particle_poca"][lane],
                                                state["particle_time"][lane]],
                                 sensor_poca=[*state["sensor_poca"][lane],
                                              state["sensor_time"][lane]],
                                 particle_direction=state["direction"][lane].tolist(),
                                 sensor_direction=state["sensor_direction"][lane].tolist(),
                                 ca_usable=bool(state["usable"][lane]),
                                 ca_status=int(state["status"][lane]),
                                 precision=D.NATIVE_CA_PRECISION,
                                 hit_state=int(hit_state)))
            seeds.append(dict(key, mass=mass[lane]))
        # Do not let positional joining accidentally pass.
        tables = dict(effects=None, iterations=None, pieces=None,
                      seed_contract=ak.Array(seeds[::-1]))
        conditions = dict(straw=dict(midz=straw_z.copy()))
        inferred = dict(midpoint=np.column_stack((np.zeros(n), np.zeros(n), straw_z)))
        fixture = dict(before=ak.Array(before), after=ak.Array(after),
                       row0=ak.Array(rows0), row1=ak.Array(rows1), parent=parent,
                       tables=tables, conditions=conditions, measured=measured,
                       par=par, ranges=ranges, counts=counts, calls=[],
                       census=dict(n_all_snapshots=2*n, n_adjacent_updates=n))
        monkeypatch.setattr(D, "EXPECTED_SNAPSHOTS", 2*n)
        monkeypatch.setattr(D, "EXPECTED_HIT_UPDATES", n)
        monkeypatch.setattr(D, "adjacent_hits", lambda *a: (
            fixture["before"], fixture["after"], fixture["row0"], fixture["row1"],
            fixture["parent"], fixture["census"]))
        monkeypatch.setattr(D, "pad_candidates", lambda *a: (par, ranges, counts))
        monkeypatch.setattr(D.R, "measured_inputs", lambda *a: (measured, None))
        monkeypatch.setattr(D.P, "source_signal_speed", lambda *a: np.full(n, 200.))

        def record(*args, **kwargs):
            inputs = deepcopy((args, kwargs))
            result = real_update(*args, **kwargs)
            fixture["calls"].append((inputs, deepcopy(result)))
            return result

        monkeypatch.setattr(D, "hit_update", record)
        fixture["run"] = lambda: D.audit_hits(tables, None, None, None,
                                               conditions, inferred, None)
        return fixture
    return build


def assert_same_generated(saved, actual):
    # Include all generated state, visited pieces, solver status and retry policy.
    for left, right in zip(saved[:2], actual[:2]):
        assert left.keys() == right.keys()
        for field in left:
            np.testing.assert_array_equal(left[field], right[field], err_msg=field)
    for field in saved[2]:
        if field == "ztime":
            if saved[2][field] is None:
                assert actual[2][field] is None
            else:
                for name in saved[2][field]:
                    np.testing.assert_array_equal(saved[2][field][name], actual[2][field][name])
        else:
            np.testing.assert_array_equal(saved[2][field], actual[2][field], err_msg=field)


def test_population_real_generation_complete_fullkey_census_and_state_transition(population):
    f = population()
    report = f["run"]()
    assert report["n_targets"] == 4
    assert report["n_excluded_comparisons"] == 0
    assert report["census"] == f["census"]
    assert report["final_ca_agreement"] == dict(criterion=True, exact=True)
    assert report["n_retry"] == report["n_give_up"] == 2
    source_ids = D.keys(f["before"], D.ITER_KEYS + ("input_index",)).tolist()
    assert report["retry_keys"] == report["give_up_keys"] == source_ids[1:3]
    assert report["previous_hit_keys"] == D.keys(f["before"], D.HIT_KEYS).tolist()
    assert report["following_hit_keys"] == D.keys(f["after"], D.HIT_KEYS).tolist()
    trans = report["hit_state_transition"]
    assert trans["comparison"]["exact"]["passed"]
    assert trans["n_previously_unusable"] == 2
    assert trans["n_new_give_up"] == 1
    assert trans["n_persistent_unusable_without_retry"] == 1
    assert report["give_up_vs_observed_state"]["n_captured_unusable_hit_state"] == 3
    assert report["give_up_vs_observed_state"]["n_captured_unusable_without_predicted_give_up"] == 1
    # Generation receives compacted owner arrays, not the captured following reference.
    args, kwargs = f["calls"][-1][0]
    np.testing.assert_array_equal(args[2], [2])
    np.testing.assert_array_equal(args[3], [0, 0, 0, 0])
    assert args[0].shape == (1, 2, 6) and not kwargs


@pytest.mark.parametrize("field,component,metric", [
    ("particle_poca", 0, "particle_poca"),
    ("particle_poca", 3, "particle_time"),
    ("sensor_poca", 0, "sensor_poca"),
    ("sensor_poca", 3, "sensor_time"),
    ("particle_direction", 1, "direction"),
    ("sensor_direction", 2, "sensor_direction"),
    ("ca_status", None, "status"),
    ("hit_state", None, "hit_state"),
])
def test_following_capture_changes_comparison_never_generated_ca(population, field, component, metric):
    f = population()
    f["run"]()
    saved = f["calls"][-1][1]
    value = ak.to_list(f["row1"][field])[0]
    if component is None:
        value = 1 if field == "ca_status" else 2
    else:
        value[component] += .125
    f["row1"] = replace(f["row1"], field, 0, value)
    if field == "particle_poca" and component == 3:
        f["after"] = replace(f["after"], "time", 0, value[3])
    report = f["run"]()
    assert_same_generated(saved, f["calls"][-1][1])
    comparison = (report["hit_state_transition"]["comparison"] if metric == "hit_state"
                  else report["final"]["status"] if metric == "status"
                  else report["final"]["geometry"][metric])
    assert comparison["criterion"]["n_failing_hits"] == 1
    assert comparison["exact"]["failing_hit_keys"] == [report["previous_hit_keys"][0]]
    assert report["n_excluded_comparisons"] == 0
    assert report["final_ca_agreement"]["criterion"] == (metric == "hit_state")


def test_following_reference_only_controls_pointer_comparison(population):
    f = population()
    f["run"]()
    saved = f["calls"][-1][1]
    for field, value in (("reference_range", f["ranges"][1, 1].tolist()),
                         ("reference_parameters", f["par"][1, 1].tolist())):
        f["after"] = replace(f["after"], field, 0, value)
    report = f["run"]()
    assert_same_generated(saved, f["calls"][-1][1])
    assert report["pointer_index"][0] == 1
    assert report["final"]["pointer"]["exact"]["n_failing_hits"] == 1
    assert all(v["exact"]["passed"] for v in report["final"]["geometry"].values())
    assert report["final"]["status"]["exact"]["passed"]


def test_prior_ca_time_is_a_permitted_hint_not_following_capture(population):
    f = population()
    f["run"]()
    saved = f["calls"][-1][1]
    value = ak.to_list(f["row0"].particle_poca)[0]
    value[3] = 3.5
    f["row0"] = replace(f["row0"], "particle_poca", 0, value)
    f["before"] = replace(f["before"], "time", 0, 3.5)
    report = f["run"]()
    generated = f["calls"][-1][1]
    assert saved[1]["index"][0] == 0 and generated[1]["index"][0] == 1
    assert not report["final_ca_agreement"]["criterion"]


def test_real_population_state_arithmetic_246_giveups_not_246_final_unusable(population):
    # Synthetic lanes reproduce the observed source-semantic partition, not data.
    prior = [-3]*110 + [1]*168
    give_up = [True]*79 + [False]*31 + [True]*167 + [False]
    f = population(prior, give_up)
    report = f["run"]()
    trans = report["hit_state_transition"]
    assert report["n_retry"] == report["n_give_up"] == 246
    assert trans["n_previously_unusable"] == 110
    assert trans["n_new_give_up"] == 167
    assert trans["n_persistent_unusable_without_retry"] == 31
    assert trans["comparison"]["exact"]["passed"]
    observed = report["give_up_vs_observed_state"]
    assert observed["n_captured_unusable_hit_state"] == 277
    assert observed["n_captured_unusable_without_predicted_give_up"] == 31
    assert observed["n_predicted_give_up_not_captured_unusable"] == 0
    assert report["n_excluded_comparisons"] == 0


@pytest.mark.parametrize("side", ["row0", "row1"])
@pytest.mark.parametrize("field", ["particle_poca", "sensor_poca", "particle_direction", "sensor_direction"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_malformed_captured_vectors_fail_whole_population(population, side, field, bad):
    f = population()
    value = ak.to_list(f[side][field])[0]
    value[0] = bad
    f[side] = replace(f[side], field, 0, value)
    with pytest.raises(ValueError, match="missing|nonfinite|finite"):
        f["run"]()


@pytest.mark.parametrize("side", ["row0", "row1"])
@pytest.mark.parametrize("field,bad", [("ca_usable", None), ("ca_usable", False),
                                       ("precision", None), ("precision", float("nan")),
                                       ("precision", 1e-3)])
def test_invalid_capture_validity_and_precision_are_rejected_not_excluded(population, side, field, bad):
    f = population()
    f[side] = replace(f[side], field, 0, bad)
    with pytest.raises(ValueError):
        f["run"]()
    assert not f["calls"]


@pytest.mark.parametrize("side,field", [("row0", "hit_state"), ("row1", "hit_state"), ("row1", "ca_status")])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_malformed_status_and_hit_state_fail_closed(population, side, field, bad):
    f = population()
    f[side] = replace(f[side], field, 0, bad)
    with pytest.raises(ValueError, match="missing|nonfinite|finite|integer"):
        f["run"]()


@pytest.mark.parametrize("mutation", ["snapshot_census", "update_census", "missing_owner",
                                      "wrong_reference", "zero_width_reference", "time_mismatch"])
def test_population_contract_failures_not_success_selection(population, mutation):
    f = population()
    if mutation == "snapshot_census":
        f["census"]["n_all_snapshots"] -= 1
    elif mutation == "update_census":
        D.EXPECTED_HIT_UPDATES += 1  # Fixture monkeypatch restores original value.
    elif mutation == "missing_owner":
        f["parent"][0] = 0
    elif mutation == "wrong_reference":
        value = ak.to_list(f["after"].reference_parameters)[0]
        value[0] += 1.
        f["after"] = replace(f["after"], "reference_parameters", 0, value)
    elif mutation == "zero_width_reference":
        f["after"] = replace(f["after"], "reference_range", 0, [0., 0.])
    else:
        f["after"] = replace(f["after"], "time", 0, 123.)
    with pytest.raises(ValueError):
        f["run"]()
    assert not f["calls"]


def test_prior_hit_state_does_not_disable_real_ca_retry(population):
    f = population()
    baseline = f["run"]()
    saved = f["calls"][-1][1]
    # Persistent -3 is NOT a CA usability flag and does not trigger zTime.
    f["row0"] = replace(f["row0"], "hit_state", 0, 1)
    report = f["run"]()
    assert_same_generated(saved, f["calls"][-1][1])
    assert report["retry_keys"] == baseline["retry_keys"]
    assert report["hit_state_transition"]["n_previously_unusable"] == 1
    assert report["hit_state_transition"]["n_persistent_unusable_without_retry"] == 0
    assert report["hit_state_transition"]["comparison"]["exact"]["n_failing_hits"] == 1
    assert report["final_ca_agreement"] == baseline["final_ca_agreement"]


@pytest.mark.parametrize("field", ["reference_parameters", "reference_range"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_following_reference_malformed_operands_fail_before_generation(population, field, bad):
    f = population()
    value = ak.to_list(f["after"][field])[0]
    value[0] = bad
    f["after"] = replace(f["after"], field, 0, value)
    with pytest.raises(ValueError, match="missing|nonfinite"):
        f["run"]()
    assert not f["calls"]
