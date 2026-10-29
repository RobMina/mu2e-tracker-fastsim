"""Synthetic conditional-history joins, recursive independence and search trace semantics."""
from copy import deepcopy
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_crossing_history as H
from test_f1_crossing_constructor import sample


def fixture():
    f = sample(3)
    rows, trajectories = [], []
    initial, associated = {}, {}
    for i in range(3):
        ctor = deepcopy(f["records"][3*i])
        ctor["category"] = i
        if i == 2:
            ctor["partner"] = -1
        rows.append(ctor)
        t = deepcopy(f["trajectories"][i])
        t["category"] = i
        trajectories.append(t)
        if i < 2:
            source = {name: f["gen"]["state"][name][i:i+1].tolist() for name in (*H.STATE, "index")}
            generated = dict(offset=f["gen"]["offset"][i:i+1].tolist(), particle_hint=[10.])
            generated["crossing" if i == 0 else "state"] = source
            report = dict(identities=dict(object_keys=[[ctor[k] for k in H.B.OBJECT]]), generated=generated)
            if i == 0:
                initial = report
            else:
                associated = report
        for step in range(2):
            newt = deepcopy(t)
            newt["trajectory_index"] += step+1
            newt["category"] = 3 if step == 0 else 4
            newt["parameters"][0] += .2*(step+1)*(i+1)
            newt["parameters"][5] += 15*(step+1)
            trajectories.append(newt)
            row = dict(ctor, sequence=1+step*3, stage=2 if step == 0 else 1, trajectory_index=newt["trajectory_index"], hint_kind=1)
            rows.extend([row, dict(row, stage=3, sequence=row["sequence"]+1), dict(row, stage=4, sequence=row["sequence"]+2, partner=20)])
        rows[-3]["partner"] = 20
        rows[-2]["partner"] = 20
    return dict(records=ak.Array(rows), trajectories=ak.Array(trajectories), initial=initial, associated=associated,
                geometry=dict(cal=f["args"][5], hv=f["args"][6], radius=f["args"][-1]))


def run(f):
    plan = H.schedule(f["records"])
    trajectories = H.pack(f["trajectories"])
    seed = H.bootstrap(plan["constructors"], f["geometry"], trajectories, f["initial"], f["associated"])
    out, traces = H.replay(f["records"], plan, trajectories, seed)
    return plan, seed, out, traces


def observed(f):
    plan, seed, out, traces = run(f)
    r = f["records"]
    for name, field in (("direction", "particle_direction"), ("sensor_direction", "sensor_direction"),
                        ("reference_parameters", "reference_parameters"), ("reference_range", "reference_range"),
                        ("status", "ca_status"), ("usable", "ca_usable"), ("effect_time", "effect_time")):
        r = ak.with_field(r, out[name], field)
    for prefix in ("particle", "sensor"):
        r = ak.with_field(r, np.column_stack((out[prefix+"_poca"], out[prefix+"_time"])), prefix+"_poca")
    r = ak.with_field(r, np.full(len(r), True), "has_search")
    r = ak.with_field(r, np.zeros(len(r), dtype=bool), "search_failed")
    for name, field in (("index", "search_final_index"), ("solution_index", "search_solution_index"), ("exhausted", "search_exhausted"),
                        ("hint_kind", "hint_kind"), ("particle_hint", "search_particle_hint"), ("sensor_hint", "search_sensor_hint")):
        r = ak.with_field(r, out[name], field)
    r = ak.with_field(r, np.maximum(out["solution_history"][:, 0], 0), "search_initial_index")
    r = ak.with_field(r, out["solution_history"][:, :1], "search_solution_indices")
    for name, field in (("particle_hint", "search_input_particle_hints"), ("sensor_hint", "search_input_sensor_hints"),
                        ("particle_time", "search_particle_times"), ("sensor_time", "search_sensor_times"),
                        ("status", "search_statuses"), ("usable", "search_usable")):
        r = ak.with_field(r, traces[name][:, :1], field)
    r = ak.with_field(r, H.partners(r, plan), "partner")
    return r, plan, seed, out, traces


def test_recursive_history_and_retention():
    f = fixture()
    plan, seed, out, _ = run(f)
    assert plan["waves"].shape == (3, 2)
    for obj in range(3):
        first, second = plan["waves"][obj]
        assert out["particle_hint"][first] == seed["state"]["particle_time"][obj]
        assert out["particle_hint"][second] == out["particle_time"][first]
        assert out["particle_time"][first+1] == out["particle_time"][first+2] == out["particle_time"][first]
    np.testing.assert_array_equal(out["effect_time"], out["particle_time"]+seed["offset"][plan["parent"]])


def test_compare_generated_history():
    args = observed(fixture())
    result = H.compare(*args)
    assert result["passed"] and result["exact"]
    assert result["n_updates"] == 6 and result["n_excluded"] == 0


@pytest.mark.parametrize("field", ["particle_poca", "sensor_poca", "reference_parameters", "reference_range", "ca_status", "ca_usable", "line_end", "offset", "effect_time", "search_particle_hint", "search_sensor_hint", "search_final_index", "search_solution_index", "search_initial_index", "hint_kind", "search_exhausted"])
def test_later_outcomes_do_not_feed_generation(field):
    f = fixture()
    before = run(f)[2]
    r, plan, seed, _, _ = observed(f)
    index = plan["update_rows"][0]
    v = r[field]
    if v.ndim == 2:
        a = H.array(v)
        a[index, 0] += 1
    else:
        a = H.array(v)
        a[index] = ~a[index] if a.dtype.kind == "b" else a[index]+1
    f["records"] = ak.with_field(r, a, field)
    after = run(f)[2]
    for key in H.STATE:
        np.testing.assert_array_equal(before[key], after[key])


def test_partner_mutations_propagate_only_at_sethit():
    f = fixture()
    plan = H.schedule(f["records"])
    partner = H.partners(f["records"], plan)
    assert partner[0] == 19 and partner[3] == 20 and partner[4] == 20
    r = ak.with_field(f["records"], ak.where(f["records"].stage == 3, 888, f["records"].partner), "partner")
    np.testing.assert_array_equal(H.partners(r, plan), partner)


def test_permutation_full_key():
    f = fixture()
    out = run(f)[2]
    keys = H.B.integer_keys(f["records"], H.B.KEYS+("sequence",))
    rng = np.random.default_rng(8)
    f["records"] = f["records"][rng.permutation(len(f["records"]))]
    f["trajectories"] = f["trajectories"][::-1]
    after = run(f)[2]
    join = H.match_rows(keys, H.B.integer_keys(f["records"], H.B.KEYS+("sequence",)))
    for name in H.STATE:
        np.testing.assert_array_equal(out[name], after[name][join])


@pytest.mark.parametrize("field", ["records", "trajectories"])
def test_duplicate_keys_fail(field):
    f = fixture()
    f[field] = ak.concatenate([f[field], f[field][:1]])
    with pytest.raises(ValueError):
        run(f)


def test_update_before_constructor_fails():
    f = fixture()
    r = f["records"]
    f["records"] = ak.with_field(r, ak.where(r.sequence == 0, 100, r.sequence), "sequence")
    with pytest.raises(ValueError, match="constructor"):
        run(f)


def test_captured_material_seed_is_declared_input():
    f = fixture()
    before = run(f)[1]
    r = f["records"]
    f["records"] = ak.with_field(r, ak.where((r.category == 2) & (r.stage == 0), r.offset*2, r.offset), "offset")
    after = run(f)[1]
    assert after["offset"][2] == 2*before["offset"][2]
    np.testing.assert_array_equal(after["offset"][:2], before["offset"][:2])


def test_unusable_fails_whole_history():
    f = fixture()
    plan, seed, _, _ = run(f)
    seed["state"]["usable"][0] = False
    with pytest.raises(ValueError, match="fallback"):
        H.replay(f["records"], plan, H.pack(f["trajectories"]), seed)


def test_generated_trace_cap_final_pointer_not_last_solution(monkeypatch):
    q = np.array([[[.3, .4, .1, .7, 10., 1000.], [.4, .4, .1, .7, 10., 1200.]]])
    ranges = np.array([[[0., 10.], [10., 20.]]])
    line = H.N.sensor(np.array([[.2, 100., 3.]]), np.array([[.2, -100., 3.]]), np.array([10.]), np.array([H.N.LIGHT]))
    def solve(parameters, sensor, ph, sh, precision, mass):
        pt = np.where(parameters[:, 0] == .3, 15., 5.)
        pp, direction, speed = H.C.particle(parameters, pt, mass)
        sp = H.C.sensor_position(sensor, sh)
        return dict(particle_time=pt, sensor_time=sh, particle_poca=pp, sensor_poca=sp, direction=direction,
                    sensor_direction=sensor["direction"], speed=speed, status=np.zeros(len(ph), dtype=np.int32), usable=np.ones(len(ph), dtype=bool))
    monkeypatch.setattr(H.L, "_solve", solve)
    state, trace = H.solve_trace(q, ranges, np.array([2]), np.array([0]), line, np.array([5.]), np.array([10.]), np.array([105.]), max_moves=10)
    assert state["exhausted"][0] and state["index"][0] == 1 and state["solution_index"][0] == 0
    assert state["status"][0] == 1 and np.all(trace["status"] == 0)
    np.testing.assert_array_equal(state["solution_history"][0], [0,1,0,1,0,1,0,1,0,1,0])
    np.testing.assert_array_equal(trace["particle_hint"][0, 1:], trace["particle_time"][0, :-1])


@pytest.mark.parametrize("field", ["particle_poca", "sensor_poca", "line_start", "line_end", "reference_parameters", "reference_range", "ca_status", "ca_usable", "effect_time", "offset", "partner", "search_initial_index", "search_final_index", "search_solution_index", "search_exhausted", "hint_kind", "search_particle_hint", "search_sensor_hint", "search_solution_indices", "search_input_particle_hints", "search_input_sensor_hints", "search_particle_times", "search_sensor_times", "search_statuses", "search_usable"])
def test_every_comparison_family_detects_corruption(field):
    records, plan, seed, out, traces = observed(fixture())
    x = H.array(records[field]).copy()
    index = plan["update_rows"][0]
    if x.ndim == 2:
        x[index, 0] = ~x[index, 0] if x.dtype.kind == "b" else x[index, 0]+1
    else:
        x[index] = ~x[index] if x.dtype.kind == "b" else x[index]+1
    r = ak.with_field(records, x, field)
    result = H.compare(r, plan, seed, out, traces)
    assert not result["passed"] and not result["exact"]


@pytest.mark.parametrize("field", ["search_solution_indices", "search_statuses"])
def test_native_unsigned_history_comparison(field):
    records, plan, seed, out, traces = observed(fixture())
    value = ak.values_astype(records[field], np.uint64)
    records = ak.with_field(records, value, field)
    assert H.compare(records, plan, seed, out, traces)["exact"]


@pytest.mark.parametrize("field", ["search_max_moves", "precision", "search_precision"])
def test_unsupported_search_configuration(field):
    records, plan, seed, out, traces = observed(fixture())
    records = ak.with_field(records, records[field]+1, field)
    with pytest.raises(ValueError):
        H.compare(records, plan, seed, out, traces)


def test_zero_duration_trajectory_piece_allowed():
    f = fixture()
    t = f["trajectories"]
    q = H.array(t.parameters)
    r = H.array(t.range)
    r[:, 1] = r[:, 0]
    t = ak.with_field(t, r, "range")
    packed = H.pack(t)
    np.testing.assert_array_equal(packed["ranges"][:, :, 0], packed["ranges"][:, :, 1])


def test_no_intermediate_capture_hint_feedback():
    f = fixture()
    before = run(f)[2]
    r = f["records"]
    r = ak.with_field(r, ak.where(r.stage == 3, 999999., r.search_particle_hint), "search_particle_hint")
    f["records"] = r
    after = run(f)[2]
    np.testing.assert_array_equal(before["particle_hint"], after["particle_hint"])


@pytest.mark.parametrize("bad", [np.array([True]), np.array([1.]), np.array([-2]), np.array([2**64-1], dtype=np.uint64)])
def test_partner_types_and_bounds(bad):
    with pytest.raises(ValueError):
        H.signed_partner(bad)


def test_bad_material_status_rejected():
    f = fixture()
    r = f["records"]
    f["records"] = ak.with_field(r, ak.where((r.stage == 0) & (r.category == 2), -1, r.ca_status), "ca_status")
    with pytest.raises(ValueError):
        run(f)


def test_const_offset_not_recomputed_after_momentum_changes():
    f = fixture()
    plan, seed, out, _ = run(f)
    rows = plan["update_rows"]
    speed = H.C.particle(out["reference_parameters"][rows], out["particle_time"][rows], np.full(len(rows), 105.6583755))[2]
    incorrect = np.full(len(rows), .0125)/speed
    assert np.any(incorrect != seed["offset"][plan["parent"][rows]])
    np.testing.assert_array_equal(out["effect_time"][rows], out["particle_time"][rows]+seed["offset"][plan["parent"][rows]])
