"""Synthetic historical-cap evidence only; no captures, builds or remote IO."""
from copy import deepcopy
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_later_ca_audit as D


def cap_fixture(monkeypatch, identity_field="iseed", same_input=False):
    targets = [dict(zip(D.KEYS, (11, 22, 33, 0)))]
    targets.append(dict(targets[0], **{identity_field: targets[0][identity_field] + 1}))
    indices = [1, 1 if same_input else 2]
    offsets = [.125, .75]
    iterations, effects, pieces, seeds, capkeys = [], [], [], [], []
    parameters = np.zeros((2, 2, 6))
    ranges = np.array([[[0., 5.], [5., 10.]], [[20., 25.], [25., 30.]]])
    for lane, target in enumerate(targets):
        parameters[lane] = np.arange(12).reshape(2, 6) + 100 * (lane + 1)
        seed_offsets = [9. + lane] * (indices[lane] + 1)
        seed_offsets[indices[lane]] = offsets[lane]
        seeds.append(dict(target, sx_offset=seed_offsets))
        capkeys.append([*target.values(), 4, indices[lane]])
        for ordinal in (4, 5):
            base = dict(target, iteration_ordinal=ordinal)
            iterations.append(dict(base, miter=2, iter=ordinal - 4, swapped=ordinal == 4))
            actual = 1
            effects.append(dict(base, identity_kind=5, input_index=indices[lane],
                                effect_index=100 * ordinal + lane, strawid=lane,
                                has_reference=True,
                                time=([4., 8.] if ordinal == 4 else [40.5, 89.])[lane] + offsets[lane],
                                reference_range=ranges[lane, actual].tolist(),
                                reference_parameters=parameters[lane, actual].tolist()))
            if ordinal == 4:
                for index in (0, 1):
                    pieces.append(dict(base, owner=1, piece_index=index,
                                       parameters=parameters[lane, index].tolist(),
                                       range=ranges[lane, index].tolist()))
    # Deliberately unrelated table orders, and a same-key non-crossing decoy.
    decoy = dict(effects[0], identity_kind=1, time=-1000.)
    tables = dict(iterations=[iterations[i] for i in (3, 0, 2, 1)],
                  effects=[effects[i] for i in (3, 0, 2, 1)] + [decoy],
                  pieces=pieces[::-1] + [dict(pieces[0], owner=0)],
                  seed_contract=seeds[::-1])
    inferred = dict(midpoint=np.array([[1., 2., 3.], [4., 5., 6.]]),
                    direction=np.array([[1., 0., 0.], [0., 1., 0.]]))
    reference = {"moves_10": {"initial_crossings": {"exhausted_keys": capkeys}}}
    prediction = dict(index=np.array([1, 0]), solution_index=np.array([0, 1]),
                      time=np.array([40., 90.]), moves=np.array([10, 10]),
                      exhausted=np.array([True, True]), usable=np.array([True, True]))
    pointer = dict(time=np.array([45., 80.]), usable=np.array([True, True]))
    calls = []

    def search(par, rng, counts, parent, mid, direction, hint):
        calls.append("search")
        np.testing.assert_array_equal(parent, [1, 2])
        np.testing.assert_array_equal(counts, [0, 2, 2, 0])
        np.testing.assert_array_equal(par[parent], parameters)
        np.testing.assert_array_equal(rng[parent], ranges)
        np.testing.assert_array_equal(mid, inferred["midpoint"])
        np.testing.assert_array_equal(direction, inferred["direction"])
        np.testing.assert_array_equal(hint, [4., 8.])
        return prediction

    def toca(par, mid, direction):
        calls.append("toca")
        # Resolve the analytic POINTER, not the retained solution or native pointer.
        np.testing.assert_array_equal(par, parameters[np.arange(2), prediction["index"]])
        np.testing.assert_array_equal(mid, inferred["midpoint"])
        np.testing.assert_array_equal(direction, inferred["direction"])
        return pointer["time"], pointer["usable"]

    monkeypatch.setattr(D.nativeref, "search", search)
    monkeypatch.setattr(D.nativeref, "toca", toca)
    return tables, inferred, reference, prediction, pointer, calls


def evidence(fixture):
    tables, inferred, reference = fixture[:3]
    return D.crossing_cap_evidence({key: ak.Array(value) for key, value in tables.items()},
                                   inferred, reference)


def test_cap_reports_actual_pointer_and_retained_time_separately(monkeypatch):
    fixture = cap_fixture(monkeypatch)
    report = evidence(fixture)
    expected = dict(constructor_offset_ns=[.125, .75],
                    native_particle_time_subtract_offset_ns=[40.5, 89.],
                    analytic_retained_time_ns=[40., 90.],
                    analytic_pointer_resolve_time_ns=[45., 80.],
                    retained_minus_native_time_ns=[-.5, 1.],
                    pointer_resolve_minus_native_time_ns=[4.5, -9.],
                    pointer_resolve_minus_retained_time_ns=[5., -10.],
                    native_pointer_index=[1, 1], analytic_pointer_index=[1, 0],
                    analytic_solution_index=[0, 1], analytic_moves=[10, 10])
    for field, values in expected.items():
        assert report[field] == values, field
    assert report["keys"] == fixture[2]["moves_10"]["initial_crossings"]["exhausted_keys"]
    assert report["key_fields"] == [*D.ITER_KEYS, "input_index"]
    assert report["pointer_agreement"] is False
    for flag in ("native_hint_calls_captured", "native_retained_vectors_compared",
                 "native_offset_time_roundtrip_exact"):
        assert report[flag] is False
    assert "NOT native crossing-vector replay" in report["scope"]
    assert fixture[-1] == ["search", "toca"]


@pytest.mark.parametrize("identity_field", D.KEYS)
def test_full_target_identity_joins_with_repeated_crossing_indices(monkeypatch, identity_field):
    fixture = cap_fixture(monkeypatch, identity_field, same_input=True)
    report = evidence(fixture)
    assert report["constructor_offset_ns"] == [.125, .75]
    assert report["native_particle_time_subtract_offset_ns"] == [40.5, 89.]
    assert fixture[-1] == ["search", "toca"]


@pytest.mark.parametrize("field", [*D.ITER_KEYS, "input_index"])
def test_missing_following_full_identity_is_not_substituted(monkeypatch, field):
    fixture = cap_fixture(monkeypatch)
    # First row is lane 1's following crossing; leave all other fields intact.
    fixture[0]["effects"][0][field] += 10
    with pytest.raises(ValueError, match="missing full identity join"):
        evidence(fixture)
    assert fixture[-1] == []


@pytest.mark.parametrize("table", ["effects", "iterations", "seed_contract"])
def test_duplicate_full_identities_fail_closed(monkeypatch, table):
    fixture = cap_fixture(monkeypatch)
    fixture[0][table].append(deepcopy(fixture[0][table][0]))
    with pytest.raises(ValueError, match="duplicate identities"):
        evidence(fixture)
    assert fixture[-1] == []


@pytest.mark.parametrize("mutation,message", [
    ("short_offsets", "cap lacks initial constructor offset"),
    ("zero_offset", "invalid cap construction offset"),
    ("nan_offset", "invalid cap construction offset"),
    ("missing_owner", "cap missing candidate owner"),
    ("no_swap", "crossing-cap owner did not swap"),
    ("stage", "crossing-cap stage changed"),
    ("gap", "crossing-cap iteration gap"),
    ("straw", "crossing-cap straw changed"),
    ("range", "cap reference range differs"),
    ("parameters", "cap reference parameters differ"),
])
def test_capture_and_constructor_guards(monkeypatch, mutation, message):
    fixture = cap_fixture(monkeypatch)
    tables = fixture[0]
    if mutation == "short_offsets":
        tables["seed_contract"][0]["sx_offset"].pop()
    elif mutation in ("zero_offset", "nan_offset"):
        tables["seed_contract"][0]["sx_offset"][-1] = 0. if mutation == "zero_offset" else np.nan
    elif mutation == "missing_owner":
        tables["pieces"] = [row for row in tables["pieces"] if row["iseed"] != 1]
    elif mutation == "no_swap":
        tables["iterations"][1]["swapped"] = False
    elif mutation in ("stage", "gap"):
        tables["iterations"][0]["miter" if mutation == "stage" else "iter"] += 1
    elif mutation == "straw":
        tables["effects"][0]["strawid"] += 1
    elif mutation == "range":
        tables["effects"][0]["reference_range"][0] += .25
    elif mutation == "parameters":
        tables["effects"][0]["reference_parameters"][0] += .25
    with pytest.raises(ValueError, match=message):
        evidence(fixture)
    assert fixture[-1] == (["search"] if mutation in ("range", "parameters") else [])


@pytest.mark.parametrize("field", ["exhausted", "usable"])
@pytest.mark.parametrize("lane", [0, 1])
def test_both_historical_caps_must_be_reproduced_and_usable(monkeypatch, field, lane):
    fixture = cap_fixture(monkeypatch)
    fixture[3][field][lane] = False
    with pytest.raises(ValueError, match="historical cap not reproduced"):
        evidence(fixture)
    assert fixture[-1] == ["search"]


@pytest.mark.parametrize("source,field,value", [
    (3, "time", np.nan), (3, "time", np.inf),
    (4, "time", np.nan), (4, "time", np.inf), (4, "usable", False),
])
def test_unusable_or_nonfinite_comparison_fails_closed(monkeypatch, source, field, value):
    fixture = cap_fixture(monkeypatch)
    fixture[source][field][1] = value
    with pytest.raises(ValueError, match="nonfinite cap comparison"):
        evidence(fixture)


@pytest.mark.parametrize("capkeys", [[], [[1, 2, 3, 4, 5, 6]], [[1.] * 6] * 2])
def test_missing_or_malformed_declared_caps_fail_closed(monkeypatch, capkeys):
    fixture = cap_fixture(monkeypatch)
    fixture[2]["moves_10"]["initial_crossings"]["exhausted_keys"] = capkeys
    with pytest.raises(ValueError, match="historical crossing-cap keys differ"):
        evidence(fixture)
    assert fixture[-1] == []


@pytest.mark.parametrize("time", [np.nan, np.inf, -np.inf])
def test_nonfinite_captured_after_time_must_not_be_reported_as_evidence(monkeypatch, time):
    fixture = cap_fixture(monkeypatch)
    fixture[0]["effects"][0]["time"] = time
    with pytest.raises(ValueError, match="nonfinite"):
        evidence(fixture)
