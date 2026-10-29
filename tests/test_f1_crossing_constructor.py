"""Synthetic full-key joins and independence checks for constructor-only replay."""
from copy import deepcopy
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_crossing_constructor as B


def sample(n=5):
    targets = [(1, 2, 3, 4), (9, 2, 3, 4), (1, 9, 3, 4), (1, 2, 9, 4), (1, 2, 3, 9)][:n]
    q = np.repeat(np.array([[[0.3, 0.4, 0.1, 0.7, 10., 1000.]]]), n, axis=0)
    ranges = np.repeat(np.array([[[0., 20.]]]), n, axis=0)
    parent = np.arange(n)
    cal = np.tile([0.2, 100., 3.], (n, 1))
    hv = np.tile([0.2, -100., 3.], (n, 1))
    args = (q, ranges, np.ones(n, dtype=int), parent, np.full(n, 105.6583755),
            cal, hv, np.full(n, 10.), np.full(n, 10.1), np.full(n, .0125))
    gen = B.generate(*args)
    records, hits, trajectories, bindings, tracker, straws = [], [], [], [], [], []
    source, hitkeys = [], []
    for i, target in enumerate(targets):
        key = dict(zip(B.KEYS, target))
        line, state = gen["line"], gen["state"]
        row = dict(key, object=8, stage=0, category=1, partner=19, strawid=17,
                   trajectory_index=4, sequence=0, input_index=-1, iteration_ordinal=-1, effect_index=-1,
                   ca_usable=True, ca_status=int(state["status"][i]), has_search=True,
                   precision=B.L.NATIVE_CA_PRECISION, search_precision=B.L.NATIVE_CA_PRECISION,
                   search_max_moves=10, search_particle_hint=10., search_sensor_hint=10.1,
                   search_initial_index=int(state["solution_history"][i, 0]),
                   search_final_index=int(state["index"][i]), search_solution_index=int(state["solution_index"][i]),
                   search_exhausted=bool(state["exhausted"][i]),
                   search_solution_indices=state["solution_history"][i, :state["moves"][i]+1].tolist(),
                   line_start=hv[i].tolist(), line_end=line["end"][i].tolist(),
                   line_direction=line["direction"][i].tolist(), line_length=float(line["length"][i]),
                   line_measurement_time=10.,
                   particle_poca=np.r_[state["particle_poca"][i], state["particle_time"][i]].tolist(),
                   sensor_poca=np.r_[state["sensor_poca"][i], state["sensor_time"][i]].tolist(),
                   particle_direction=state["direction"][i].tolist(), sensor_direction=state["sensor_direction"][i].tolist(),
                   reference_parameters=gen["reference_parameters"][i].tolist(),
                   reference_range=gen["reference_range"][i].tolist(), radius_operand=.0125,
                   speed_operand=float(gen["reference_speed"][i]), offset=float(gen["offset"][i]),
                   effect_time=float(gen["effect_time"][i]))
        records.extend([row, dict(row, stage=3, sequence=1, input_index=7, iteration_ordinal=5, effect_index=6),
                        dict(row, stage=3, sequence=2, input_index=7, iteration_ordinal=6, effect_index=8)])
        hits.append(dict(key, iteration_ordinal=5, effect_index=3, shindex=19, strawid=17))
        trajectories.append(dict(key, trajectory_index=4, category=1, piece_index=0,
                                 parameters=q[i, 0].tolist(), range=ranges[i, 0].tolist(), mass=105.6583755))
        bindings.append(dict(key, source_status=1, geometry_index=i))
        tracker.append(dict(geometry_index=i, wire_radius=.0125))
        straws.append(dict(geometry_index=i, fullsid=17, tube_cal=cal[i].tolist(), tube_hv=hv[i].tolist()))
        source.append(list(target)+( [7]))
        hitkeys.append(list(target)+[5, 3])
    hints = dict(source_keys=source, partner_hit_keys=hitkeys,
                 reconstructed_hit_constructor_particle_hint=[10.]*n,
                 reconstructed_hit_constructor_sensor_hint=[10.1]*n, straw_line_measurement_time=[10.]*n)
    return dict(args=args, gen=gen, records=records, hits=hits, trajectories=trajectories,
                bindings=bindings, tracker=tracker, straws=straws, hints=hints)


def prepare(f):
    selected, source, values = B.associated_inputs(ak.Array(f["records"]), ak.Array(f["hits"]), f["hints"])
    q, r, counts, parent, mass = B.pack_trajectories(ak.Array(f["trajectories"]), selected)
    cal, hv, radius = B.geometry_inputs(selected, ak.Array(f["bindings"]), ak.Array(f["tracker"]), ak.Array(f["straws"]))
    gen = B.generate(q, r, counts, parent, mass, cal, hv, values["particle_hint"], values["sensor_hint"], radius)
    return gen, selected, B.integer_keys(selected, B.OBJECT)


def test_synthetic_population_exact_and_no_mutation():
    f = sample()
    before = deepcopy(f["records"])
    gen, selected, ids = prepare(f)
    result = B.compare(gen, selected, ids)
    assert result["passed"] and result["exact"] and result["n_constructors"] == 5
    assert result["n_excluded"] == 0
    assert f["records"] == before


def test_fullkey_permutation_invariant():
    f = sample()
    original = prepare(f)
    for name in ("records", "hits", "trajectories", "bindings", "tracker", "straws"):
        f[name].reverse()
    gen, selected, ids = prepare(f)
    assert B.compare(gen, selected, ids)["exact"]
    join = B.match_rows(original[2], ids)
    np.testing.assert_array_equal(original[0]["offset"], gen["offset"][join])


@pytest.mark.parametrize("field", ["offset", "effect_time", "speed_operand", "radius_operand", "search_particle_hint",
                                   "search_sensor_hint", "line_measurement_time", "line_length"])
def test_captured_scalar_perturbations_never_change_generation(field):
    f = sample()
    expected = prepare(f)[0]
    f["records"][0][field] += 1.
    gen, selected, ids = prepare(f)
    np.testing.assert_array_equal(gen["offset"], expected["offset"])
    np.testing.assert_array_equal(gen["state"]["particle_time"], expected["state"]["particle_time"])
    assert not B.compare(gen, selected, ids)["passed"]


@pytest.mark.parametrize("field", ["line_start", "line_end", "line_direction", "particle_poca", "sensor_poca",
                                   "particle_direction", "sensor_direction", "reference_parameters", "reference_range"])
def test_captured_vector_perturbations_do_not_feed_back(field):
    f = sample()
    expected = prepare(f)[0]
    f["records"][0][field][0] += 1.
    gen, selected, ids = prepare(f)
    np.testing.assert_array_equal(gen["state"]["particle_poca"], expected["state"]["particle_poca"])
    assert not B.compare(gen, selected, ids)["passed"]


@pytest.mark.parametrize("field", ["ca_status", "search_initial_index", "search_final_index", "search_solution_index"])
def test_captured_indices_and_status_are_comparison_only(field):
    f = sample()
    f["records"][0][field] += 1
    gen, selected, ids = prepare(f)
    assert not B.compare(gen, selected, ids)["passed"]


def test_reversed_tube_line_is_detected_without_using_recorded_endpoints():
    f = sample()
    f["straws"][0]["tube_cal"], f["straws"][0]["tube_hv"] = f["straws"][0]["tube_hv"], f["straws"][0]["tube_cal"]
    gen, selected, ids = prepare(f)
    assert not B.compare(gen, selected, ids)["passed"]
    np.testing.assert_array_equal(gen["line"]["direction"][0], [0., -1., 0.])


@pytest.mark.parametrize("name", ["records", "hits", "trajectories", "bindings", "tracker", "straws"])
def test_duplicate_identity_rejected(name):
    f = sample()
    f[name].append(deepcopy(f[name][0]))
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("name", ["hits", "trajectories", "bindings", "tracker", "straws"])
def test_missing_parent_rejected(name):
    f = sample()
    f[name].pop()
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("field", B.KEYS)
def test_target_identity_mismatch(field):
    f = sample()
    f["hits"][0][field] += 30
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("field", ["partner", "strawid"])
def test_wrong_constructor_partner_or_straw_rejected(field):
    f = sample()
    f["records"][0][field] += 1
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("name", ["source_keys", "partner_hit_keys"])
@pytest.mark.parametrize("value", [-1, 1.5, True, 2**64-1])
def test_invalid_hint_identity(name, value):
    f = sample()
    f["hints"][name][0][0] = value
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("name", ["reconstructed_hit_constructor_particle_hint", "reconstructed_hit_constructor_sensor_hint", "straw_line_measurement_time"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad"])
def test_nonfinite_hint_rejected(name, value):
    f = sample()
    f["hints"][name][0] = value
    with pytest.raises((ValueError, TypeError)):
        prepare(f)


def test_line_anchor_contract():
    f = sample()
    f["hints"]["straw_line_measurement_time"][0] += 1
    with pytest.raises(ValueError, match="line anchor"):
        prepare(f)


def test_missing_first_record_rejected():
    f = sample()
    f["records"] = f["records"][:1]+f["records"][3:]
    with pytest.raises(ValueError, match="missing first"):
        prepare(f)


def test_sparse_piece_indices_rejected():
    f = sample()
    f["trajectories"][0]["piece_index"] = 2
    with pytest.raises(ValueError, match="noncontiguous"):
        prepare(f)


def test_radius_from_wire_geometry_not_recorded_operand():
    f = sample()
    before = prepare(f)[0]
    f["tracker"][0]["wire_radius"] *= 2
    after = prepare(f)[0]
    assert after["offset"][0] == 2*before["offset"][0]
    np.testing.assert_array_equal(after["state"]["particle_time"], before["state"]["particle_time"])


def test_capped_constructor_offset_uses_final_reference_not_solution(monkeypatch):
    f = sample(1)
    args = list(f["args"])
    args[0] = np.repeat(args[0], 2, axis=1)
    args[0][0, 1, 5] = 20.
    args[1] = np.array([[[0., 10.], [10., 20.]]])
    args[2] = np.array([2])
    state = deepcopy(f["gen"]["state"])
    state.update(index=np.array([1]), solution_index=np.array([0]), exhausted=np.array([True]))
    monkeypatch.setattr(B.L, "piecewise", lambda *a, **kw: state)
    result = B.generate(*args)
    expected_speed = B.C.particle(args[0][:, 1], state["particle_time"], args[4])[2]
    np.testing.assert_array_equal(result["offset"], args[-1]/expected_speed)
    assert not np.array_equal(result["offset"], args[-1]/state["speed"])


@pytest.mark.parametrize("index", [4, 5, 6, 7, 8, 9])
def test_generation_rejects_masked_operands(index):
    args = list(sample()["args"])
    args[index] = np.ma.array(args[index], mask=np.ones_like(args[index], dtype=bool))
    with pytest.raises(ValueError):
        B.generate(*args)


@pytest.mark.parametrize("field", ["parameters", "range", "mass"])
def test_nonfinite_counted_trajectory(field):
    f = sample()
    if field == "mass":
        f["trajectories"][0][field] = np.nan
    else:
        f["trajectories"][0][field][0] = np.nan
    with pytest.raises(ValueError):
        prepare(f)


def test_zero_duration_piece_is_not_discarded():
    f = sample(1)
    f["trajectories"][0]["range"] = [0., 0.]
    f["trajectories"].append(dict(f["trajectories"][0], piece_index=1, range=[0., 20.]))
    selected = ak.Array(f["records"])[np.array([True, False, False])]
    q, r, counts, parent, mass = B.pack_trajectories(ak.Array(f["trajectories"]), selected)
    assert counts.tolist() == [2]
    assert r[0, 0].tolist() == [0., 0.]


@pytest.mark.parametrize("field", ["precision", "search_max_moves", "has_search", "ca_usable"])
def test_unsupported_capture_contract_fails_without_exclusion(field):
    f = sample()
    f["records"][0][field] = False if field in ("has_search", "ca_usable") else 0
    gen, selected, ids = prepare(f)
    with pytest.raises(ValueError):
        B.compare(gen, selected, ids)


def test_extension_trajectory_without_accepted_crossing_is_retained():
    f = sample(1)
    f["trajectories"].append(dict(f["trajectories"][0], event=500))
    selected = ak.Array(f["records"])[np.array([True, False, False])]
    q, r, count, parent, mass = B.pack_trajectories(ak.Array(f["trajectories"]), selected)
    assert len(count) == 2 and len(np.unique(parent)) == 1
    gen, selected, ids = prepare(f)
    assert B.compare(gen, selected, ids)["exact"]


@pytest.mark.parametrize("field", ["ca_usable", "search_exhausted", "has_search"])
@pytest.mark.parametrize("value", [0, 1, 2, 1.0])
def test_nonboolean_flags_rejected(field, value):
    f = sample()
    for row in f["records"]:
        row[field] = value
    gen, selected, ids = prepare(f)
    with pytest.raises(ValueError):
        B.compare(gen, selected, ids)


@pytest.mark.parametrize("field", ["ca_status", "search_final_index", "search_solution_index", "search_initial_index", "search_max_moves"])
def test_float_discrete_index_rejected(field):
    f = sample()
    for row in f["records"]:
        row[field] = float(row[field])
    gen, selected, ids = prepare(f)
    with pytest.raises(ValueError):
        B.compare(gen, selected, ids)


def test_comparison_overflow_fails_closed():
    with pytest.raises(ValueError, match="overflow"):
        B.comparison(np.array([1e308]), np.array([-1e308]), np.ones((1, 5), dtype=int))


def test_wrong_search_precision_rejected():
    f = sample()
    f["records"][0]["search_precision"] = 1e-4
    gen, selected, ids = prepare(f)
    with pytest.raises(ValueError, match="search precision"):
        B.compare(gen, selected, ids)


def test_native_unsigned_history_padding_does_not_promote_to_float():
    gen, selected, ids = prepare(sample())
    selected = ak.with_field(selected, ak.values_astype(selected.search_solution_indices, np.uint64), "search_solution_indices")
    assert B.compare(gen, selected, ids)["exact"]


@pytest.mark.parametrize("value", [2**63, 2**64-1])
def test_unsigned_history_overflow_rejected(value):
    gen, selected, ids = prepare(sample())
    hist = np.zeros((len(selected), 1), dtype=np.uint64)
    hist[0, 0] = value
    selected = ak.with_field(selected, ak.Array(hist), "search_solution_indices")
    with pytest.raises(ValueError, match="overflow"):
        B.compare(gen, selected, ids)
