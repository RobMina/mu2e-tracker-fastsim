"""Synthetic-only crossing capture audit; no ROOT/native runtime or fitted outputs."""
import copy
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_crossing_record_check as C


def piece(index=0):
    return dict(parameters=[float(index + n) for n in range(6)], covariance=np.eye(6).ravel().tolist(),
                range=[float(index * 10), float((index + 1) * 10)], bnom=[0., 0., 1.], mass=105., charge=-1)


def state(index=0, usable=True):
    return dict(ca_status=0 if usable else 3, ca_usable=usable, precision=C.PRECISION,
                **{"reference_" + k: v for k, v in piece(index).items()},
                **{k: [0.] * w if usable else [] for k, w in C.DERIVED.items()},
                **{k: [0., 0., 1.] for k in C.LINE},
                line_measurement_time=0., line_length=1., doca=1. if usable else np.nan,
                doca_variance=2. if usable else np.nan, toca_variance=3. if usable else np.nan,
                lsign=-1. if usable else np.nan)


def search(usable=True, cap=10, exhausted=False):
    return dict(has_search=True, search_failed=False, search_exhausted=exhausted,
                search_initial_index=0, search_final_index=0, search_solution_index=0, search_max_moves=cap,
                search_particle_hint=0., search_sensor_hint=0., search_precision=C.PRECISION,
                search_solution_indices=[0], search_input_particle_hints=[0.], search_input_sensor_hints=[0.],
                search_particle_times=[0.] if usable else [np.nan], search_sensor_times=[0.] if usable else [np.nan],
                search_statuses=[0] if usable else [3], search_usable=[usable])


def no_search():
    return dict(has_search=False, search_failed=False, search_exhausted=False,
                **{n: 0 for n in C.SEARCH_INT}, **{n: np.nan for n in C.SEARCH_FLOAT},
                **{n: [] for n in C.SEARCH_VECTORS})


def no_state():
    return dict(ca_status=6, ca_usable=False, precision=np.nan, reference_mass=np.nan, reference_charge=0,
                line_length=np.nan, line_measurement_time=np.nan, **{n: np.nan for n in C.CA_FLOAT},
                **{n: [] for n in (*C.DERIVED, *C.LINE)}, **{"reference_" + n: [] for n in C.PIECE})


def sample():
    records, trajectories, discovery, statuses, effects = [], [], [], [], []
    for subrun in (0, 1):
        base = dict(schema_version=1, run=8, subrun=subrun, event=42, iseed=3)
        for trajectory, category, count in ((0, 0, 1), (1, 3, 2), (2, 4, 2), (3, 2, 1)):
            for index in range(count):
                trajectories.append(dict(base, trajectory_index=trajectory, category=category,
                                         piece_index=index, **piece(index)))
        r = dict(base, object=0, sequence=0, stage=0, category=0, hint_kind=-1, strawid=100, partner=4,
                 trajectory_index=0, iteration_ordinal=-1, effect_index=-1, input_index=-1,
                 intrinsic_active=True, active=True, offset=.125, effect_time=.125,
                 radius_operand=.25, speed_operand=2., **state(), **search())
        local = [r]
        wrapper = dict(r, sequence=1, stage=2, hint_kind=1, trajectory_index=1,
                       radius_operand=np.nan, speed_operand=np.nan)
        local.append(wrapper)
        pre = dict(wrapper, sequence=2, stage=3, hint_kind=-1, iteration_ordinal=0,
                   effect_index=0, input_index=10, **no_search())
        local.append(pre)
        mutation = dict(pre, sequence=3, stage=4, effect_index=-1, input_index=-1, partner=9)
        local.append(mutation)
        local.append(dict(pre, sequence=4, partner=9, iteration_ordinal=1))
        unusable = dict(wrapper, sequence=5, stage=1, partner=9, trajectory_index=2, iteration_ordinal=1,
                        effect_time=np.nan, **state(index=1, usable=False), **search(usable=False))
        # Native pointer moves BEFORE unusability check: reference piece 1, last solved piece 0.
        unusable["search_final_index"] = 1
        local.append(unusable)
        local.append(dict(unusable, sequence=6, stage=3, hint_kind=-1, iteration_ordinal=2,
                          effect_index=0, input_index=10, **no_search()))
        material = dict(r, sequence=7, object=1, category=2, strawid=200, partner=-1, trajectory_index=3,
                        intrinsic_active=False, active=False)
        local.append(material)
        local.append(dict(material, sequence=8, stage=3, iteration_ordinal=3, effect_index=0, input_index=11,
                          radius_operand=np.nan, speed_operand=np.nan, **no_search()))
        records.extend(local)
        for row in local:
            if row["stage"] == 3:
                effects.append(dict(base, iteration_ordinal=row["iteration_ordinal"], effect_index=0,
                                    identity_kind=5, kind=2, input_index=row["input_index"], strawid=row["strawid"],
                                    partner_shindex=9 if row["object"] == 0 else -1,
                                    identity_from_final_collections=True, identity_valid=True, has_reference=True,
                                    has_change=True, payload_supported=True, active=row["active"],
                                    time=row["effect_time"] if row["ca_usable"] else 1234.,
                                    **{"reference_" + n: row["reference_" + n] for n in C.PIECE}))
        for stage in range(9):
            values = [0.] * [17, 0, 6, 11, 1, 0, 0, 5, 0][stage]
            integers = [0] * [2, 0, 0, 3, 1, 2, 1, 0, 0][stage]
            discovery.append(dict(base, sequence=stage, stage=stage, trajectory_index=3,
                                  plane=0 if stage >= 1 else -1, panel=0 if stage >= 3 else -1,
                                  straw=200 if stage >= 6 else -1, object=1 if stage == 8 else -1,
                                  passed=True, values=values, integers=integers,
                                  **(state() if stage == 7 else no_state()),
                                  **(search() if stage == 7 else no_search())))
        statuses.extend([dict(base, status=0, n_records=0, n_objects=0, n_trajectories=0, n_discovery=0, error=""),
                         dict(base, status=1, n_records=9, n_objects=2, n_trajectories=4, n_discovery=9, error="")])
    tables = dict(zip(C.SCHEMAS, (records, trajectories, discovery, statuses)))
    return tables, effects, np.array([[8, 0, 42, 3], [8, 1, 42, 3]], dtype=np.int64)


def arrays(tables, effects, targets):
    # Construct using an explicit branch schema, including typed empty vectors.
    def table(rows, schema):
        out = {}
        for name, (kind, ndim) in schema.items():
            dtype = {"iu": np.int64, "f": np.float64, "biu": np.bool_, "US": str}[kind]
            if ndim == 1:
                out[name] = np.array([r[name] for r in rows], dtype=dtype)
            else:
                lengths = np.array([len(r[name]) for r in rows], dtype=np.int64)
                flat = np.array([v for r in rows for v in r[name]], dtype=dtype)
                out[name] = ak.unflatten(flat, lengths)
        return ak.Array(out)
    return ({name: table(tables[name], schema) for name, schema in C.SCHEMAS.items()},
            table(effects, C.EFFECT), targets)


def check(data=None):
    return C.validate(*arrays(*(sample() if data is None else data)))


def test_positive_full_population_and_limits():
    with np.errstate(all="raise"):
        report = check()
    assert report["passed"] and report["n_targets"] == 2 and report["n_objects"] == 4
    assert report["n_prefilter"] == report["n_expected_prefilter"] == 8
    assert report["n_unusable_ca"] == 4 and report["n_excluded_prefilter"] == 0
    assert not report["detailed_physical_discovery_replay"] and not report["full_closure"]
    json.dumps(report, allow_nan=False)


def test_empty_selected_targets_and_typed_empty_sidecars():
    tables, effects, targets = sample()
    assert C.validate(*arrays({k: [] for k in tables}, [], targets[:0]))["passed"]


@pytest.mark.parametrize("table,row,field,value", [
    ("crossing_capture_status", 0, "status", 1),
    ("crossing_capture_status", 1, "status", -1),
    ("crossing_capture_status", 1, "error", "failed"),
    ("crossing_capture_status", 0, "n_objects", 1),
    ("crossing_capture_status", 1, "n_records", 8),
    ("crossing_capture_status", 1, "n_objects", 1),
    ("crossing_capture_status", 1, "n_trajectories", 6),
    ("crossing_capture_status", 1, "n_discovery", 8),
    ("crossing_records", 0, "sequence", 1),
    ("crossing_records", 0, "stage", 5),
    ("crossing_records", 0, "object", 2),
    ("crossing_records", 0, "category", 3),
    ("crossing_records", 0, "partner", -1),
    ("crossing_records", 0, "strawid", 65536),
    ("crossing_records", 0, "trajectory_index", 9),
    ("crossing_records", 0, "radius_operand", .251),
    ("crossing_records", 0, "speed_operand", 0.),
    ("crossing_records", 1, "radius_operand", .25),
    ("crossing_records", 2, "offset", np.nextafter(.125, 1.)),
    ("crossing_records", 2, "category", 1),
    ("crossing_records", 2, "strawid", 101),
    ("crossing_records", 2, "partner", 9),
    ("crossing_records", 2, "effect_index", -1),
    ("crossing_records", 2, "input_index", 13),
    ("crossing_records", 2, "iteration_ordinal", 1),
    ("crossing_records", 0, "precision", 1e-4),
    ("crossing_records", 0, "ca_status", 3),
    ("crossing_records", 0, "doca", np.nan),
    ("crossing_records", 0, "lsign", 0.),
    ("crossing_records", 0, "effect_time", .126),
    ("crossing_records", 0, "particle_poca", [0., 0., 0.]),
    ("crossing_records", 5, "particle_poca", [0., 0., 0., 0.]),
    ("crossing_records", 5, "doca", 0.),
    ("crossing_records", 6, "effect_time", 0.),
    ("crossing_records", 0, "search_failed", True),
    ("crossing_records", 0, "search_precision", 1e-4),
    ("crossing_records", 0, "search_initial_index", 1),
    ("crossing_records", 0, "search_solution_index", 1),
    ("crossing_records", 0, "search_final_index", 1),
    ("crossing_records", 0, "search_solution_indices", [0, 0]),
    ("crossing_records", 0, "search_solution_indices", [-1]),
    ("crossing_records", 0, "search_statuses", [7]),
    ("crossing_records", 0, "search_usable", [False]),
    ("crossing_records", 0, "search_particle_times", [np.nan]),
    ("crossing_records", 5, "search_particle_times", [0.]),
    ("crossing_records", 5, "search_final_index", 0),
    ("crossing_records", 0, "search_exhausted", True),
    ("crossing_records", 0, "search_max_moves", -1),
    ("crossing_records", 0, "search_max_moves", 2**32),
    ("crossing_records", 0, "search_particle_hint", .1),
    ("crossing_records", 0, "search_input_sensor_hints", [np.nan]),
    ("crossing_records", 2, "search_initial_index", 1),
    ("crossing_records", 2, "search_precision", C.PRECISION),
    ("crossing_records", 2, "search_statuses", [0]),
    ("crossing_records", 1, "hint_kind", -1),
    ("crossing_records", 2, "has_search", True),
    ("crossing_records", 4, "line_start", [0., 0., 2.]),
    ("crossing_records", 4, "reference_parameters", [10.] * 6),
    ("crossing_records", 8, "active", True),
    ("crossing_trajectories", 0, "parameters", [1.] * 5),
    ("crossing_trajectories", 0, "covariance", [np.nan] * 36),
    ("crossing_trajectories", 0, "range", [1., 0.]),
    ("crossing_trajectories", 0, "piece_index", 1),
    ("crossing_trajectories", 1, "trajectory_index", 8),
    ("crossing_trajectories", 2, "category", 4),
    ("crossing_discovery", 0, "passed", False),
    ("crossing_discovery", 0, "values", []),
    ("crossing_discovery", 0, "integers", []),
    ("crossing_discovery", 1, "sequence", 2),
    ("crossing_discovery", 1, "stage", 9),
    ("crossing_discovery", 1, "passed", False),
    ("crossing_discovery", 2, "plane", 1),
    ("crossing_discovery", 3, "panel", -1),
    ("crossing_discovery", 4, "values", [np.inf]),
    ("crossing_discovery", 6, "straw", 201),
    ("crossing_discovery", 7, "search_failed", True),
    ("crossing_discovery", 7, "passed", False),
    ("crossing_discovery", 7, "reference_mass", 104.),
    ("crossing_discovery", 8, "object", 0),
    ("crossing_discovery", 8, "trajectory_index", 0),
    ("crossing_discovery", 8, "precision", C.PRECISION),
])
def test_reject_corrupted_capture(table, row, field, value):
    data = sample()
    data[0][table][row][field] = value
    with pytest.raises(ValueError):
        check(data)


@pytest.mark.parametrize("field,value", [("input_index", 11), ("strawid", 101), ("active", False),
    ("time", np.nextafter(.125, 1.)), ("reference_parameters", [99.] * 6),
    ("reference_covariance", [0.] * 36), ("reference_range", [0., 11.]), ("reference_bnom", [0., 0., 2.]),
    ("identity_kind", 1), ("kind", 1), ("identity_valid", False), ("payload_supported", False),
    ("has_reference", False), ("has_change", False), ("subrun", 2), ("iteration_ordinal", 10)])
def test_reject_effect_mismatch(field, value):
    data = sample()
    data[1][0][field] = value
    with pytest.raises(ValueError):
        check(data)


def test_partner_final_stamping_is_not_history():
    data = sample()
    data[1][0]["partner_shindex"] = 987  # Final metadata, never assumed contemporaneous.
    assert check(data)["passed"]
    data[1][0]["identity_from_final_collections"] = False
    with pytest.raises(ValueError, match="contemporaneous"):
        check(data)


def test_cap_exhaustion_uses_final_pointer_not_solution_and_demotes_status():
    data = sample()
    r = data[0]["crossing_records"]
    for row in r[1:5]:
        row.update(state(index=1))
        row["ca_status"] = 1
    r[1].update(search(cap=1, exhausted=True), search_final_index=1,
                search_initial_index=1, search_solution_index=0, search_solution_indices=[1, 0],
                search_input_particle_hints=[0., 0.], search_input_sensor_hints=[0., 0.],
                search_particle_times=[0., 0.], search_sensor_times=[0., 0.],
                search_statuses=[0, 0], search_usable=[True, True])
    for e in data[1][:2]:
        e.update({"reference_" + n: piece(1)[n] for n in C.PIECE})
    assert check(data)["n_exhausted_searches"] == 1
    r[1]["ca_status"] = 0
    with pytest.raises(ValueError):
        check(data)


def test_zero_cap_and_discovery_native_nan_operands():
    data = sample()
    data[0]["crossing_records"][0]["search_max_moves"] = 0
    data[0]["crossing_discovery"][7]["values"] = [np.nan] * 5
    assert check(data)["passed"]


@pytest.mark.parametrize("name", list(C.SCHEMAS))
def test_missing_or_extra_schema_field(name):
    t, e, k = arrays(*sample())
    t[name] = ak.without_field(t[name], "schema_version")
    with pytest.raises(ValueError, match="missing required"):
        C.validate(t, e, k)
    t, e, k = arrays(*sample())
    t[name] = ak.with_field(t[name], np.zeros(len(t[name])), "unknown")
    with pytest.raises(ValueError, match="unexpected"):
        C.validate(t, e, k)


@pytest.mark.parametrize("field,value", [("object", np.uint64(2**63)), ("sequence", 0.5),
    ("ca_usable", 2), ("radius_operand", np.float32(.25)), ("partner", "1")])
def test_bad_types_and_overflow(field, value):
    t, e, k = arrays(*sample())
    t["crossing_records"] = ak.with_field(t["crossing_records"], np.full(len(t["crossing_records"]), value), field)
    with pytest.raises(ValueError):
        C.validate(t, e, k)


def test_missing_nested_and_scalar_values():
    t, e, k = arrays(*sample())
    for field in ("offset", "particle_poca"):
        original = t["crossing_records"]
        mask = np.arange(len(original)) != 0
        t["crossing_records"] = ak.with_field(original, ak.mask(original[field], mask), field)
        with pytest.raises(ValueError, match="missing/masked"):
            C.validate(t, e, k)
        t["crossing_records"] = original
    x = ak.Array([[None] * 4] + [[0.] * 4] * (len(original) - 1))
    t["crossing_records"] = ak.with_field(original, x, "particle_poca")
    with pytest.raises(ValueError, match="missing/masked"):
        C.validate(t, e, k)


@pytest.mark.parametrize("which", ["duplicate_ctor", "missing_record", "missing_effect", "duplicate_effect", "missing_status", "reverse_status", "foreign_target", "duplicate_targets", "overflow_targets"])
def test_missing_duplicate_and_full_target_keys(which):
    data = sample()
    t, e, k = data
    if which == "duplicate_ctor":
        t["crossing_records"][1]["stage"] = 0
    elif which == "missing_record":
        t["crossing_records"].pop(2)
    elif which == "missing_effect":
        e.pop(0)
    elif which == "duplicate_effect":
        e.append(copy.deepcopy(e[0]))
    elif which == "missing_status":
        t["crossing_capture_status"].pop(0)
    elif which == "reverse_status":
        t["crossing_capture_status"][:2] = t["crossing_capture_status"][:2][::-1]
    elif which == "foreign_target":
        k[0, 1] = 8
    elif which == "duplicate_targets":
        k[0] = k[1]
    elif which == "overflow_targets":
        k = k.astype(np.uint64); k[0, 0] = 2**63
    with pytest.raises(ValueError):
        check((t, e, k))


def test_vectorized_validator_no_python_record_iteration_or_ak_to_list():
    import ast
    source = Path(C.__file__).read_text()
    assert "ak.to_list" not in source
    tree = ast.parse(source)
    # Loops are schema/field/fixed-stage loops, never target/record data loops.
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)):
            assert ast.unparse(node.iter) not in {"r", "t", "d", "status", "effects", "targets", "records"}


def test_reject_reordered_record_stream_but_allow_interleaved_targets():
    t, e, k = arrays(*sample())
    t["crossing_records"] = t["crossing_records"][np.arange(18).reshape(2, 9).T.ravel()]
    assert C.validate(t, e[::-1], k[::-1])["passed"]
    t["crossing_records"] = t["crossing_records"][::-1]
    with pytest.raises(ValueError, match="sequence"):
        C.validate(t, e, k)


def test_untyped_empty_and_native_int_overflow_rejected():
    t, e, k = arrays(*sample())
    d = t["crossing_discovery"]
    t["crossing_discovery"] = ak.with_field(d, ak.Array([[]] * len(d)), "d_dd_p")
    with pytest.raises(ValueError, match="untyped"):
        C.validate(t, e, k)
    t, e, k = arrays(*sample())
    t["crossing_records"] = ak.with_field(t["crossing_records"], np.full(18, 2**31, dtype=np.int64), "run")
    with pytest.raises(ValueError, match="int32"):
        C.validate(t, e, k)


def test_trajectory_serialization_order_and_input_reuse():
    t, e, k = arrays(*sample())
    t["crossing_trajectories"] = t["crossing_trajectories"][[1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    with pytest.raises(ValueError, match="trajectory IDs"):
        C.validate(t, e, k)
    data = sample()
    data[0]["crossing_records"][8]["input_index"] = 10
    data[1][3]["input_index"] = 10
    with pytest.raises(ValueError, match="reused"):
        check(data)


@pytest.mark.parametrize("field,value", [("hint_kind", 0), ("search_particle_hint", 1.),
                                         ("search_input_particle_hints", [1.])])
def test_hint_source_validation(field, value):
    data = sample()
    data[0]["crossing_records"][1][field] = value
    with pytest.raises(ValueError):
        check(data)


def test_targets_with_no_objects_still_require_complete_status():
    t, e, k = arrays(*sample())
    t = {name: table[:0] for name, table in t.items()}
    base = sample()[0]["crossing_capture_status"][:2]
    base[1].update(n_records=0, n_objects=0, n_trajectories=0, n_discovery=0)
    t["crossing_capture_status"] = ak.Array(base)
    report = C.validate(t, e[:0], k[:1])
    assert report["passed"] and report["n_objects"] == 0 and report["n_targets"] == 1


def test_missing_discovery_header_and_unconditional_child_rejected():
    data = sample()
    data[0]["crossing_discovery"] = data[0]["crossing_discovery"][9:]
    data[0]["crossing_capture_status"][1]["n_discovery"] = 0
    # Remove material constructors too, to exercise empty accepted population.
    data[0]["crossing_records"] = data[0]["crossing_records"][:7] + data[0]["crossing_records"][9:]
    data[0]["crossing_capture_status"][1].update(n_records=7, n_objects=1)
    data[1].pop(3)
    with pytest.raises(ValueError, match="header"):
        check(data)
    data = sample()
    data[0]["crossing_discovery"][1]["stage"] = 3
    with pytest.raises(ValueError):
        check(data)


def test_all_unusable_search_times_cannot_bypass_nan_gate():
    data = sample()
    for table in (data[0]["crossing_records"], data[0]["crossing_discovery"]):
        for row in table:
            if "effect_time" in row or row["stage"] == 7:
                row.update(state(index=1 if row.get("trajectory_index") == 2 else 0, usable=False))
                if "effect_time" in row:
                    row["effect_time"] = np.nan
                if row["has_search"]:
                    row["search_statuses"] = [3]
                    row["search_usable"] = [False]
                    row["search_particle_times"] = [np.nan]
                    row["search_sensor_times"] = [np.nan]
                if row["stage"] in (1, 2) and "effect_time" in row:
                    row["hint_kind"] = 2
    assert check(data)["passed"]
    data[0]["crossing_records"][0]["search_particle_times"] = [0.]
    with pytest.raises(ValueError, match="NaN"):
        check(data)


def test_all_absent_discovery_defaults_checked():
    data = sample()
    # Keep only stage 0 and a failed plane-existence check per target; no CA.
    d = data[0]["crossing_discovery"]
    data[0]["crossing_discovery"] = [d[0], dict(d[1], passed=False), d[9], dict(d[10], passed=False)]
    r = data[0]["crossing_records"]
    data[0]["crossing_records"] = r[:7] + r[9:16]
    data[1][:] = data[1][:3] + data[1][4:7]
    for status in data[0]["crossing_capture_status"]:
        if status["status"] == 1:
            status.update(n_records=7, n_objects=1, n_discovery=2)
    assert check(data)["passed"]
    for field in ("line_length", "reference_mass", "search_precision", "search_sensor_hint"):
        data[0]["crossing_discovery"][0][field] = 1.
        with pytest.raises(ValueError, match="default"):
            check(data)
        data[0]["crossing_discovery"][0][field] = np.nan
