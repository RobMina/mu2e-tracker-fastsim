"""Generated material bootstrap independence, full-key joins and receipt guards."""
from copy import deepcopy
import json
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

import f1_discovered_history_audit as D
from test_f1_crossing_history import fixture, observed


def prepared():
    f = fixture()
    plan = D.H.schedule(f["records"])
    ctors = plan["constructors"]
    t = D.H.pack(f["trajectories"])
    parent = D.A.match_rows(D.B.integer_keys(ctors, D.B.TRAJECTORY, unique=False), t["keys"])
    g = f["geometry"]
    generated = D.B.generate(t["q"], t["ranges"], t["counts"], parent, t["mass"][parent], g["cal"], g["hv"],
                             np.full(3, 10.), np.full(3, 10.1), g["radius"])
    ids = D.B.integer_keys(ctors, D.B.OBJECT)
    receipts = []
    for category in range(3):
        take = np.flatnonzero(D.A.column(ctors, "category") == category)
        source = {k: generated["state"][k][take].tolist() for k in (*D.H.STATE, "index")}
        payload = {k: generated[k][take].tolist() for k in ("offset", "particle_hint")}
        payload["crossing" if category == 0 else "state"] = source
        receipts.append(dict(identities=dict(object_keys=ids[take].tolist()), generated=payload))
    return f, plan, t, receipts


def seed(f, plan, t, receipts):
    return D.bootstrap(plan["constructors"], f["geometry"], t, *receipts)


def args(tmp_path):
    return SimpleNamespace(**{k: tmp_path/k for k in ("out", "full_check", "initial", "associated", "history", "material", "capture", "native_source")})


def test_all_generated_bootstrap_matches_prior_conditional_fixture():
    f, plan, t, receipts = prepared()
    s = seed(f, plan, t, receipts)
    original = D.H.bootstrap(plan["constructors"], f["geometry"], t, f["initial"], f["associated"])
    for name in D.H.STATE:
        np.testing.assert_array_equal(s["state"][name], original["state"][name])
    np.testing.assert_array_equal(s["offset"], original["offset"])
    records, oldplan, _, _, _ = observed(f)
    out, traces = D.H.replay(records, oldplan, t, s)
    assert D.H.compare(records, oldplan, s, out, traces)["exact"]


@pytest.mark.parametrize("field", ["particle_poca", "sensor_poca", "particle_direction", "sensor_direction", "offset", "effect_time", "line_measurement_time", "ca_status", "ca_usable", "search_final_index", "reference_parameters", "reference_range", "search_particle_hint", "search_sensor_hint"])
def test_constructor_outcome_columns_not_inputs(field):
    f, plan, t, receipts = prepared()
    before = seed(f, plan, t, receipts)
    plan["constructors"] = ak.without_field(plan["constructors"], field)
    after = seed(f, plan, t, receipts)
    for name in D.H.STATE:
        np.testing.assert_array_equal(after["state"][name], before["state"][name])
    np.testing.assert_array_equal(after["offset"], before["offset"])
    np.testing.assert_array_equal(after["line"]["measurement_time"], before["line"]["measurement_time"])


def test_generated_material_changes_feed_later_history():
    f, plan, t, receipts = prepared()
    before = seed(f, plan, t, receipts)
    receipts[2]["generated"]["offset"][0] *= 2
    receipts[2]["generated"]["state"]["particle_time"][0] += .2
    after = seed(f, plan, t, receipts)
    out, _ = D.H.replay(f["records"], plan, t, after)
    first = plan["waves"][2, 0]
    assert out["particle_hint"][first] == before["state"]["particle_time"][2]+.2
    assert after["offset"][2] == before["offset"][2]*2
    np.testing.assert_array_equal(after["offset"][:2], before["offset"][:2])


@pytest.mark.parametrize("problem", ["missing", "duplicate", "target", "status", "index", "offset", "usable_type", "usable_value"])
def test_invalid_generated_material_fails(problem):
    f, plan, t, receipts = prepared()
    material = receipts[2]
    if problem == "missing":
        material["identities"]["object_keys"] = []
    elif problem == "duplicate":
        material["identities"]["object_keys"] *= 2
    elif problem == "target":
        material["identities"]["object_keys"][0][0] += 100
    elif problem == "status":
        material["generated"]["state"]["status"][0] = 4
    elif problem == "index":
        material["generated"]["state"]["index"][0] = True
    elif problem == "offset":
        material["generated"]["offset"][0] = 0.
    elif problem == "usable_type":
        material["generated"]["state"]["usable"][0] = 1
    else:
        material["generated"]["state"]["usable"][0] = False
    with pytest.raises(ValueError):
        seed(f, plan, t, receipts)


def test_full_key_permutation():
    f, plan, t, receipts = prepared()
    before = seed(f, plan, t, receipts)
    plan["constructors"] = plan["constructors"][::-1]
    f["geometry"] = {k: v[::-1] for k, v in f["geometry"].items()}
    after = seed(f, plan, t, receipts)
    for name in D.H.STATE:
        np.testing.assert_array_equal(after["state"][name], before["state"][name][::-1])


def test_helpers_and_foreign_bootstrap(monkeypatch):
    assert len(D.helpers({})) == 15
    monkeypatch.setattr(D, "bootstrap", lambda *a: None)
    with pytest.raises(ValueError):
        D.helpers({})


def test_failure_output_exclusive(tmp_path):
    a = args(tmp_path)
    result = D.run(a)
    assert not result["passed"] and not result["completed"] and result["hashes_unchanged"]
    assert "FileNotFoundError" in result["error"]
    assert json.loads(a.out.read_text()) == result
    with pytest.raises(FileExistsError):
        D.run(a)


@pytest.mark.parametrize("name", ["full_check", "initial", "associated", "history", "material", "capture", "native_source"])
def test_protected_output(tmp_path, name):
    a = args(tmp_path)
    a.out = getattr(a, name)/"report.json" if name == "native_source" else getattr(a, name)
    with pytest.raises(ValueError, match="protected"):
        D.run(a)


def test_end_hash_change_fails(tmp_path, monkeypatch):
    a = args(tmp_path)
    p = tmp_path/"input"
    p.write_text("one")
    def helpers(watched):
        D.D.watch(p, watched)
        p.write_text("two")
        raise ValueError("stop")
    monkeypatch.setattr(D, "helpers", helpers)
    result = D.run(a)
    assert not result["passed"] and not result["hashes_unchanged"]
    assert not result["generated_constructor_ca_history"]
