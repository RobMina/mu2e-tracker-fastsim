"""Synthetic comparison-only contracts, not captured-data or fit equivalence tests.

Only this test adapter manufactures captured outcomes from independent generation.
Object IDs are deliberately unrelated to straw IDs; all captured buffers are copies.
"""
from copy import deepcopy

import awkward as ak
import numpy as np
import pytest

import f1_material_discovery as M
import f1_material_discovery_compare as V
from test_f1_material_discovery import full_geometry, trajectory


TRACE_FIELDS = dict(particle_hint="search_input_particle_hints",
                    sensor_hint="search_input_sensor_hints",
                    particle_time="search_particle_times",
                    sensor_time="search_sensor_times",
                    status="search_statuses", usable="search_usable")


def ragged(value):
    """Preserve numeric dtype, including zero-width stage payloads."""
    return ak.unflatten(value.ravel().copy(), np.full(len(value), value.shape[1]))


def outcome_columns(t, out):
    g, s = out["generated"], out["generated"]["state"]
    n = len(s["index"])
    columns = dict(
        particle_poca=np.column_stack((s["particle_poca"], s["particle_time"])),
        sensor_poca=np.column_stack((s["sensor_poca"], s["sensor_time"])),
        particle_direction=s["direction"], sensor_direction=s["sensor_direction"],
        line_start=g["line_start"], line_end=g["line"]["end"],
        line_direction=g["line"]["direction"], line_length=g["line"]["length"],
        line_measurement_time=g["line"]["measurement_time"],
        reference_parameters=g["reference_parameters"], reference_range=g["reference_range"],
        d_dd_p=out["derivatives"]["H"][:, 0], d_td_p=-out["derivatives"]["H"][:, 1],
        doca=out["derivatives"]["doca"], doca_variance=out["docavar"],
        radius_operand=g["radius"], speed_operand=g["reference_speed"],
        offset=g["offset"], effect_time=g["effect_time"],
        search_particle_hint=g["particle_hint"], search_sensor_hint=g["sensor_hint"],
        ca_status=s["status"], ca_usable=s["usable"],
        search_initial_index=s["solution_history"][:, 0], search_final_index=s["index"],
        search_solution_index=s["solution_index"], search_exhausted=s["exhausted"],
        has_search=np.ones(n, dtype=bool), search_failed=np.zeros(n, dtype=bool),
        search_max_moves=np.full(n, 10, dtype=np.int64),
        precision=np.full(n, M.L.NATIVE_CA_PRECISION),
        search_precision=np.full(n, M.L.NATIVE_CA_PRECISION))
    columns = {name: value.copy() for name, value in columns.items()}
    valid = np.arange(11)[None, :] < (s["moves"] + 1)[:, None]
    columns["search_solution_indices"] = ak.unflatten(s["solution_history"][valid].copy(), s["moves"] + 1)
    parent = out["stages"][7]["parent"]
    _, trace = M.H.solve_trace(t["q"], t["ranges"], t["counts"], parent, g["line"],
                              g["particle_hint"], g["sensor_hint"], t["mass"][parent])
    for name, field in TRACE_FIELDS.items():
        columns[field] = ak.unflatten(trace[name][valid].copy(), s["moves"] + 1)
    return columns


def captured_tables(t, out):
    """Test-only adapter: never pass these tables into production generation."""
    common = outcome_columns(t, out)
    accepted = out["accepted"]
    attempt = out["stages"][7]
    object_ids = 100_000 + 3 * np.arange(np.count_nonzero(accepted))
    constructors = {name: deepcopy(value[accepted]) for name, value in common.items()}
    constructors.update({name: t["keys"][attempt["parent"][accepted], i].copy()
                         for i, name in enumerate(M.B.TRAJECTORY)})
    constructors.update(object=object_ids.copy(), strawid=attempt["straw"][accepted].copy(),
                        category=np.full(len(object_ids), 2, dtype=np.int64),
                        stage=np.zeros(len(object_ids), dtype=np.int64),
                        partner=np.full(len(object_ids), -1, dtype=np.int64))
    records, sort_keys = [], []
    for number in range(9):
        stage = out["stages"][number]
        n = len(stage["parent"])
        columns = {}
        for name, value in common.items():
            if isinstance(value, ak.Array):
                columns[name] = ak.unflatten(ak.to_numpy(ak.flatten(value))[:0].copy(),
                                             np.zeros(n, dtype=np.int64))
            else:
                columns[name] = np.zeros((n,) + value.shape[1:], dtype=value.dtype)
        if number == 7:
            columns.update(deepcopy(common))
        columns.update({name: t["keys"][stage["parent"], i].copy()
                        for i, name in enumerate(M.B.TRAJECTORY)})
        columns.update({name: stage[name].copy() for name in ("plane", "panel", "straw", "passed")})
        integers = ragged(stage["integers"])
        if number == 0:
            integers = ak.concatenate((integers, ak.copy(stage["old"])), axis=1)
        columns.update(stage=np.full(n, number, dtype=np.int64),
                       values=ragged(stage["values"]), integers=integers,
                       object=object_ids.copy() if number == 8 else np.full(n, -1, dtype=np.int64))
        records.append(ak.Array(columns))
        sort_keys.append(np.column_stack((stage["parent"], stage["plane"], stage["panel"],
                                          stage["straw"], np.full(n, number))))
    keys = np.concatenate(sort_keys)
    order = np.lexsort((keys[:, 4], keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))
    records = ak.concatenate(records)[order]
    position = np.arange(len(order))
    starts = np.r_[True, keys[order, 0][1:] != keys[order, 0][:-1]]
    sequence = position - np.maximum.accumulate(np.where(starts, position, 0))
    return ak.copy(ak.with_field(records, sequence, "sequence")), ak.copy(ak.Array(constructors))


@pytest.fixture(scope="module")
def comparison_world():
    t = trajectory(n=1)
    t["ranges"][:] = [-10., 10.]
    g = M.geometry(full_geometry(t))
    old = np.zeros((1, 65536), dtype=bool)
    old[0, 65535] = True  # Exercise stage-zero membership without changing reached windows.
    out = M.generate(t, g, old)
    discovery, constructors = captured_tables(t, out)
    return t, out, discovery, constructors


def changed(table, field, row, *, component=None, value=None, delta=1):
    """Replace one copied column; never mutate fixture or generated buffers."""
    column = table[field]
    if column.ndim == 1:
        data = ak.to_numpy(column).copy()
        index = row
        data[index] = value if value is not None else data[index] + delta
        replacement = data
    else:
        lengths = ak.to_numpy(ak.num(column, axis=1))
        data = ak.to_numpy(ak.flatten(column)).copy()
        index = int(lengths[:row].sum()) + (0 if component is None else component)
        data[index] = value if value is not None else data[index] + delta
        replacement = ak.unflatten(data, lengths)
    return ak.with_field(ak.copy(table), replacement, field)


def stage_row(discovery, number):
    return int(np.flatnonzero(ak.to_numpy(discovery.stage) == number)[0])


def assert_failed_metric(report, name, *, constructor=False):
    assert not report["passed"] and not report["exact"]
    metrics = report["constructors"]["metrics"] if constructor else report["metrics"]
    result = metrics[name]
    assert not result["passed"] and not result["exact"]
    assert result["n_failed_rows"] == result["n_inexact_rows"] == 1
    assert len(result["failed_keys"]) == 1


def test_full_synthetic_comparison_and_identity_join(comparison_world):
    t, out, discovery, constructors = comparison_world
    # Physical row order is irrelevant; chronology is the captured sequence field.
    report = V.compare(out, t, discovery[::-1], constructors[::-1])
    assert report["passed"] and report["exact"]
    assert report["n_attempts"] == report["constructors"]["n_constructors"] == 1080
    assert report["n_records"] == len(discovery)
    assert report["n_piece_solves"] == int((out["generated"]["state"]["moves"] + 1).sum())
    assert len(report["stages"]) == 9
    assert all(item["passed"] and item["exact"] for item in report["metrics"].values())
    assert np.all(ak.to_numpy(constructors.category) == 2)
    assert np.all(ak.to_numpy(constructors.partner) == -1)
    assert np.all(ak.to_numpy(constructors.object) != ak.to_numpy(constructors.strawid))
    np.testing.assert_array_equal(report["object_keys"][:, -1], ak.to_numpy(constructors.object))
    np.testing.assert_array_equal(ak.to_numpy(discovery.sequence), np.arange(len(discovery)))


def test_capture_and_comparison_results_do_not_alias_generation(comparison_world):
    t, original, discovery, constructors = comparison_world
    out = deepcopy(original)
    report = V.compare(out, t, discovery, constructors)
    captured = ak.to_numpy(constructors.reference_parameters)
    assert not np.shares_memory(captured, original["generated"]["reference_parameters"])
    assert not np.shares_memory(ak.to_numpy(discovery.line_end), original["generated"]["line"]["end"])
    assert not np.shares_memory(ak.to_numpy(ak.flatten(constructors.search_solution_indices)),
                                original["generated"]["state"]["solution_history"])
    for key, value in report["generated"].items():
        if isinstance(value, dict):
            for name, array in value.items():
                assert not np.shares_memory(array, out["generated"][key][name])
        else:
            assert not np.shares_memory(value, out["generated"][key])
    report["generated"]["reference_parameters"][0, 0] += 100.
    report["generated"]["state"]["particle_time"][0] += 100.
    np.testing.assert_array_equal(out["generated"]["reference_parameters"], original["generated"]["reference_parameters"])
    np.testing.assert_array_equal(out["generated"]["state"]["particle_time"], original["generated"]["state"]["particle_time"])


@pytest.mark.parametrize("number,field,value", [(0, "values", None), (1, "passed", False),
                                                (3, "integers", None), (7, "values", None)])
def test_corrupted_stage_payload_fails(comparison_world, number, field, value):
    t, out, discovery, constructors = comparison_world
    bad = changed(discovery, field, stage_row(discovery, number), value=value)
    assert_failed_metric(V.compare(out, t, bad, constructors), f"stage{number}/{field}")


def test_within_tolerance_is_passed_but_not_exact(comparison_world):
    t, out, discovery, constructors = comparison_world
    bad = changed(discovery, "values", stage_row(discovery, 0), delta=1e-9)
    report = V.compare(out, t, bad, constructors)
    assert report["passed"] and not report["exact"]
    metric = report["metrics"]["stage0/values"]
    assert metric["n_failed_rows"] == 0 and metric["n_inexact_rows"] == 1


@pytest.mark.parametrize("field,metric,value", [
    ("particle_poca", "CA/particle_poca", None),
    ("doca_variance", "CA/doca_variance", None),
    ("ca_status", "CA/ca_status", None),
    ("ca_usable", "CA/ca_usable", False),
    ("search_solution_indices", "CA/history", None),
    ("search_input_particle_hints", "trace/particle_hint", None),
    ("search_statuses", "trace/status", None),
    ("search_usable", "trace/usable", False),
])
def test_corrupted_attempt_state_or_trace_fails(comparison_world, field, metric, value):
    t, out, discovery, constructors = comparison_world
    bad = changed(discovery, field, stage_row(discovery, 7), value=value)
    assert_failed_metric(V.compare(out, t, bad, constructors), metric)


@pytest.mark.parametrize("field,metric", [("offset", "offset"),
                                          ("reference_parameters", "reference_parameters"),
                                          ("search_final_index", "index")])
def test_corrupted_constructor_outcome_fails(comparison_world, field, metric):
    t, out, discovery, constructors = comparison_world
    bad = changed(constructors, field, 0)
    report = V.compare(out, t, discovery, bad)
    assert all(item["passed"] for item in report["metrics"].values())
    assert_failed_metric(report, metric, constructor=True)


@pytest.mark.parametrize("problem,message", [
    ("chronology", "hierarchy order differs"),
    ("accepted_link", "accepted object link differs"),
    ("partner", "material constructor has partner"),
    ("old_identity", "oldstraw identities differ"),
    ("old_size", "oldstraw set size differs"),
    ("search_cap", "search cap differs"),
    ("trace_index", "invalid trace index"),
])
def test_corrupted_capture_contract_rejected(comparison_world, problem, message):
    t, out, discovery, constructors = comparison_world
    if problem == "chronology":
        discovery = changed(discovery, "sequence", stage_row(discovery, 7))
    elif problem == "accepted_link":
        discovery = changed(discovery, "object", stage_row(discovery, 8))
    elif problem == "partner":
        constructors = changed(constructors, "partner", 0, value=0)
    elif problem == "old_identity":
        discovery = changed(discovery, "integers", stage_row(discovery, 0), component=2, value=65534)
    elif problem == "old_size":
        integer = discovery.integers
        keep = (discovery.stage != 0) | (ak.local_index(integer, axis=1) < 2)
        discovery = ak.with_field(ak.copy(discovery), integer[keep], "integers")
    elif problem == "search_cap":
        discovery = changed(discovery, "search_max_moves", stage_row(discovery, 7), value=9)
    else:
        discovery = changed(discovery, "search_solution_indices", stage_row(discovery, 7), value=-1)
    with pytest.raises(ValueError, match=message):
        V.compare(out, t, discovery, constructors)
