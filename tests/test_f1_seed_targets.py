"""Synthetic target-adapter tests, not an independent numerical certificate."""
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_seed_contract_audit as D


CONFIG = dict(momentum=2000., errors=[15., .05, 10., .02, 2., 1.],
              constraints=[0., 0., 0., 0., 0., .0001])
SPEED = dict(bin_width=1., mean_half_velocity=100., scale=[1., 1.1, 1.2])


def key_fields(keys, names=D.B.KEYS):
    return dict(zip(names, keys.T))


def captured_crossing(crossing):
    # Expected capture is a separate generation, testing adapters rather than CA accuracy.
    state, line = crossing["state"], crossing["line"]
    n = len(state["particle_time"])
    history = state["solution_history"]
    history = ak.unflatten(history[history >= 0], state["moves"]+1)
    return dict(line_start=crossing["line_start"], line_end=line["end"],
                line_direction=line["direction"], line_length=line["length"],
                line_measurement_time=line["measurement_time"],
                search_particle_hint=crossing["particle_hint"],
                search_sensor_hint=crossing["sensor_hint"],
                particle_poca=np.column_stack((state["particle_poca"], state["particle_time"])),
                sensor_poca=np.column_stack((state["sensor_poca"], state["sensor_time"])),
                particle_direction=state["direction"], sensor_direction=state["sensor_direction"],
                reference_parameters=crossing["reference_parameters"],
                reference_range=crossing["reference_range"], radius_operand=crossing["radius"],
                speed_operand=crossing["reference_speed"], offset=crossing["offset"],
                effect_time=crossing["effect_time"], ca_status=state["status"],
                ca_usable=state["usable"], search_final_index=state["index"],
                search_solution_index=state["solution_index"], search_initial_index=state["solution_history"][:, 0],
                search_exhausted=state["exhausted"], search_solution_indices=history,
                has_search=np.ones(n, dtype=bool), search_max_moves=np.full(n, 10, dtype=np.int64),
                precision=np.full(n, D.B.L.NATIVE_CA_PRECISION),
                search_precision=np.full(n, D.B.L.NATIVE_CA_PRECISION))


@pytest.fixture(scope="module")
def population():
    # Three seeds per event; event numbers repeat across both runs and subruns.
    n, ne = 147, 49
    index, event = np.arange(n), np.arange(ne)
    event_keys = np.column_stack((11+event//21, 2+(event//7)%3, 17+event%7))
    keys = np.column_stack((np.repeat(event_keys, 3, axis=0), np.tile(np.arange(3), ne)))
    counts = np.full(n, 17, dtype=np.int64)+(index < 39)
    parent = np.repeat(index, counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    local = np.arange(len(parent))-starts[parent]
    event_counts = counts.reshape(ne, 3).sum(axis=1)
    event_starts = np.r_[0, np.cumsum(event_counts)[:-1]]
    source_index = np.arange(len(parent))-event_starts[parent//3]
    sid = 17+local
    raw = np.column_stack((.2+.01*index, .1+.02*index, np.zeros(n), np.zeros(n), 100.+.2*index))
    measured_time = (raw[parent, 4]-1.+.001*local).astype(np.float32)
    event_fields = key_fields(event_keys, D.B.KEYS[:3])
    event_fields.update({"sd_"+name: raw[:, i].reshape(ne, 3) for i, name in enumerate(D.S.SEED_FIELDS)})
    event_fields.update(sd_n=counts.reshape(ne, 3), sd_ind=ak.unflatten(source_index, event_counts),
                        s_sid=ak.unflatten(sid, event_counts), s_time=ak.unflatten(measured_time, event_counts),
                        s_et0=ak.unflatten(measured_time+np.float32(.1), event_counts),
                        s_et1=ak.unflatten(measured_time+np.float32(.2), event_counts),
                        s_edep=ak.unflatten(np.full(len(parent), .001, dtype=np.float32), event_counts),
                        s_eend=ak.unflatten(np.zeros(len(parent), dtype=np.int64), event_counts))
    events = ak.Array(event_fields)
    source = D.S.inputs(events, keys)
    mass = np.full(n, 105.658367)
    generated = D.S.generate(source["raw"], mass, CONFIG)
    y, z = -30.-4.*local, raw[parent, 1]+.5
    wire_cal = np.column_stack((np.full(len(parent), 100.), y, z))
    wire_hv = np.column_stack((np.full(len(parent), -100.), y, z))
    tube_cal, tube_hv = wire_cal+[0., 0., .2], wire_hv+[0., 0., .2]
    geometry = dict(cal=tube_cal, hv=tube_hv, wire_cal=wire_cal, wire_hv=wire_hv,
                    radius=np.full(len(parent), .0125))
    measured = D.I.measurements(events, source["hit_keys"], sid)
    rebuilt = D.I.generate(generated["parameters"][:, None, :], source["construction_range"][:, None, :],
                           np.ones(n, dtype=np.int64), parent, mass[parent], geometry, measured, SPEED)
    hit, crossing = rebuilt["hit"], rebuilt["crossing"]
    # Independent array reduction checks adapter aggregation, not the numerical generator.
    fitrange = np.column_stack((np.minimum.reduceat(np.minimum(hit["particle_time"], crossing["effect_time"]), starts),
                               np.maximum.reduceat(np.maximum(hit["particle_time"], crossing["effect_time"]), starts)))
    seeds = ak.Array(key_fields(keys) | dict(valid=np.ones(n, dtype=bool), mass=mass,
                     parameters=generated["parameters"], covariance=generated["covariance"].reshape(n, 36),
                     sh_shindex=ak.unflatten(source_index, counts), sh_strawid=ak.unflatten(sid, counts),
                     sh_time=ak.unflatten(hit["particle_time"], counts),
                     range_begin=fitrange[:, 0], range_end=fitrange[:, 1],
                     ph_time=fitrange.mean(axis=1)[:, None]))
    trajectory_index = 1000+index
    records = ak.Array(key_fields(keys[parent]) | dict(object=local, partner=source_index, strawid=sid,
                       trajectory_index=trajectory_index[parent], stage=np.zeros(len(parent), dtype=np.int64),
                       category=np.zeros(len(parent), dtype=np.int64)) | captured_crossing(crossing))
    trajectories = ak.Array(key_fields(keys) | dict(trajectory_index=trajectory_index,
                            piece_index=np.zeros(n, dtype=np.int64), category=np.zeros(n, dtype=np.int64),
                            parameters=generated["parameters"], covariance=generated["covariance"].reshape(n, 36),
                            range=source["construction_range"]))
    geometry_index = 2000+index
    bindings = ak.Array(key_fields(keys) | dict(geometry_index=geometry_index, source_status=np.ones(n, dtype=np.int64)))
    tracker = ak.Array(dict(geometry_index=geometry_index, wire_radius=np.full(n, .0125)))
    straws = ak.Array(dict(geometry_index=geometry_index[parent], fullsid=sid, tube_cal=tube_cal,
                          tube_hv=tube_hv, wire_cal=wire_cal, wire_hv=wire_hv))
    tables = dict(crossing_records=records, crossing_trajectories=trajectories,
                  crossing_geometry_bindings=bindings, crossing_geometry_tracker=tracker,
                  crossing_geometry_straws=straws, events=events)
    assert len(keys) == 147 and len(records) == 2538
    return SimpleNamespace(seeds=seeds, source=source, generated=generated, tables=tables,
                           keys=keys, fitrange=fitrange, rebuilt=rebuilt, starts=starts, parent=parent)


def compare(monkeypatch, data, *, tables=None, seeds=None):
    tables = data.tables if tables is None else tables
    seeds = data.seeds if seeds is None else seeds
    calls, root, cycles = [], object(), {}
    def read_tree(actual_root, directory, name, actual_cycles, fields=None):
        assert actual_root is root and actual_cycles is cycles
        assert directory == ("trkdump" if name == "events" else "kktrace")
        calls.append(name)
        return tables[name] if fields is None else tables[name][fields]
    monkeypatch.setattr(D.A, "read_tree", read_tree)
    # Source and generated arrays must share seed order, even when all tables are shuffled.
    keys = D.B.integer_keys(seeds, D.B.KEYS)
    source = D.S.inputs(tables["events"], keys)
    generated = D.S.generate(source["raw"], D.A.column(seeds, "mass"), CONFIG)
    result = D.compare_targets(root, seeds, source, generated, CONFIG, SPEED, cycles)
    assert set(calls) == set(tables) and len(calls) == len(tables)
    return result


def replace(table, field, value, *, row=0, component=None):
    values = np.array(ak.to_numpy(table[field]), copy=True)
    if component is None:
        values[row] = value
    else:
        values[row, component] = value
    return ak.with_field(table, values, field)


def assert_passed(result):
    assert result["n_targets"] == 147 and result["n_constructors"] == 2538
    assert all(item["passed"] and item["exact"] for item in result["comparison"].values())
    assert result["crossing_comparison"]["passed"] and result["crossing_comparison"]["exact"]
    assert result["crossing_comparison"]["n_excluded"] == 0


def test_full_census_real_generation(monkeypatch, population):
    result = compare(monkeypatch, population)
    assert_passed(result)
    join = D.A.match_rows(np.asarray(result["generated"]["keys"]), population.keys)
    np.testing.assert_array_equal(result["generated"]["parameters"], population.generated["parameters"][join])
    np.testing.assert_array_equal(result["generated"]["construction_range"], population.source["construction_range"][join])
    np.testing.assert_array_equal(result["generated"]["fit_range"], population.fitrange[join])
    np.testing.assert_array_equal(result["generated"]["constraint_time"], population.fitrange[join].mean(axis=1))
    assert np.all(population.fitrange[:, 0] > population.source["construction_range"][:, 1])


def test_full_key_independent_table_shuffles(monkeypatch, population):
    baseline = compare(monkeypatch, population)
    rng = np.random.default_rng(913)
    tables = {name: table[rng.permutation(len(table))] for name, table in population.tables.items()}
    seeds = population.seeds[rng.permutation(len(population.seeds))]
    shuffled = compare(monkeypatch, population, tables=tables, seeds=seeds)
    assert_passed(shuffled)
    assert shuffled == baseline


@pytest.mark.parametrize("table,field,component,value,failed", [
    ("crossing_trajectories", "parameters", 0, 900., "construction_parameters"),
    ("crossing_trajectories", "range", 0, -900., "construction_range"),
    ("crossing_trajectories", "covariance", 0, 900., "construction_covariance"),
    ("seeds", "parameters", 0, 900., None),
    ("seeds", "range_begin", None, -900., "generated_fit_range"),
    ("seeds", "range_end", None, 900., "generated_fit_range"),
    ("seeds", "ph_time", 0, 900., "generated_constraint_time"),
])
def test_captured_q_and_ranges_never_drive_generated_cas(monkeypatch, population, table, field, component, value, failed):
    baseline = compare(monkeypatch, population)
    tables, seeds = dict(population.tables), population.seeds
    if table == "seeds":
        seeds = replace(seeds, field, value, component=component)
    else:
        tables[table] = replace(tables[table], field, value, component=component)
    result = compare(monkeypatch, population, tables=tables, seeds=seeds)
    assert result["generated"] == baseline["generated"]
    assert result["crossing_comparison"] == baseline["crossing_comparison"]
    assert {name for name, item in result["comparison"].items() if not item["passed"]} == ({failed} if failed else set())
    if failed:
        assert result["comparison"][failed]["failed_keys"] == population.keys[:1].tolist()


@pytest.mark.parametrize("table", ["crossing_records", "crossing_trajectories"])
def test_wrong_trajectory_index_rejected(monkeypatch, population, table):
    tables = dict(population.tables)
    tables[table] = replace(tables[table], "trajectory_index", 999999)
    with pytest.raises(ValueError):
        compare(monkeypatch, population, tables=tables)


def test_fit_range_uses_generated_ca_plus_offset(monkeypatch, population):
    result = compare(monkeypatch, population)
    crossing = population.rebuilt["crossing"]
    hit_time = population.rebuilt["hit"]["particle_time"]
    ca_time = crossing["state"]["particle_time"]
    np.testing.assert_array_equal(crossing["effect_time"], ca_time+crossing["offset"])
    assert np.all(crossing["offset"] > 0)
    ca_only_end = np.maximum.reduceat(np.maximum(hit_time, ca_time), population.starts)
    assert np.all(population.fitrange[:, 1] > ca_only_end)
    tables = dict(population.tables)
    records = tables["crossing_records"]
    records = ak.with_field(records, np.full(len(records), 999.), "effect_time")
    records = ak.with_field(records, np.full(len(records), 999.), "offset")
    tables["crossing_records"] = records
    mutated = compare(monkeypatch, population, tables=tables)
    assert mutated["generated"] == result["generated"]
    assert mutated["comparison"] == result["comparison"]
    assert {name for name, item in mutated["crossing_comparison"]["metrics"].items() if not item["passed"]} == {"offset", "effect_time"}


def test_captured_hit_times_only_change_comparison(monkeypatch, population):
    baseline = compare(monkeypatch, population)
    seeds = ak.with_field(population.seeds, population.seeds.sh_time+1000., "sh_time")
    result = compare(monkeypatch, population, seeds=seeds)
    assert result["generated"] == baseline["generated"]
    assert result["crossing_comparison"] == baseline["crossing_comparison"]
    assert {name for name, item in result["comparison"].items() if not item["passed"]} == {"hit_time"}


@pytest.mark.parametrize("remove", ["target", "constructor"])
def test_fixed_census_not_bypassed(monkeypatch, population, remove):
    tables, seeds = dict(population.tables), population.seeds
    if remove == "target":
        tables["crossing_records"] = tables["crossing_records"][population.parent != 0]
        tables["crossing_trajectories"] = tables["crossing_trajectories"][1:]
    else:
        tables["crossing_records"] = tables["crossing_records"][1:]
        counts = np.asarray(ak.to_numpy(ak.num(seeds.sh_shindex, axis=1))).copy()
        counts[0] -= 1
        for field in ("sh_shindex", "sh_strawid", "sh_time"):
            seeds = ak.with_field(seeds, ak.unflatten(ak.flatten(seeds[field], axis=1)[1:], counts), field)
    with pytest.raises(ValueError, match="target census differs"):
        compare(monkeypatch, population, tables=tables, seeds=seeds)


def snapshot(table):
    form, length, buffers = ak.to_buffers(table)
    return form.to_json(), length, {name: value.tobytes() for name, value in buffers.items()}


def test_inputs_and_helper_bindings_unchanged(monkeypatch, population):
    tables = dict(population.tables, seeds=population.seeds)
    before = {name: snapshot(table) for name, table in tables.items()}
    source = {name: value.copy() for name, value in population.source.items()}
    generated = {name: value.copy() for name, value in population.generated.items()}
    helpers = (D.I.generate, D.B.generate, D.B.L.piecewise, D.B.C.constructor_hint,
               D.S.inputs, D.S.generate, D.S.bounds, D.A.require)
    assert_passed(compare(monkeypatch, population))
    assert before == {name: snapshot(table) for name, table in tables.items()}
    for name, value in source.items():
        np.testing.assert_array_equal(population.source[name], value)
    for name, value in generated.items():
        np.testing.assert_array_equal(population.generated[name], value)
    assert helpers == (D.I.generate, D.B.generate, D.B.L.piecewise, D.B.C.constructor_hint,
                       D.S.inputs, D.S.generate, D.S.bounds, D.A.require)
