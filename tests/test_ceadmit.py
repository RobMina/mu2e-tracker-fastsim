"""Admission is fail-closed report validation, never a physics-closure verdict."""
from copy import deepcopy
import hashlib
import json

import pytest

from trkfastsim.validation import ceadmit as C


DIGEST = "a" * 64


def reports():
    return dict(
        census=dict(events=3, birth_keys_sha256=DIGEST, status_counts={"0": 3},
                    status_zero_events=3, validity_failure_events=0,
                    stock_selected_events=1, stock_rejected_events=2,
                    inclusive_path_pass_events=3, physics_eligible=True,
                    generation_saved_metadata_exact=True,
                    raw_products={"all_required_products_present": True},
                    event_cycles={"Events;1": 1, "Events;2": 3},
                    input_sha256={"art": "b" * 64, "generation_log": "c" * 64, "saved_log": "d" * 64}),
        source=dict(events=3, exact_sequential_order=True, source_window_complete=True,
                    source_keys_sha256="e" * 64, generated_keys_sha256=DIGEST,
                    input_sha256={"source_log": "f" * 64, "window_art": "0" * 64}),
        primary=dict(generated_starts=3, external_expected_events=3, primary_electrons=3,
                     primary_vd13=2, primary_steps=4, primary_step_events=2,
                     primary_step_events_without_vd13=0,
                     steps_before_first_vd13_in_crossing_events=0,
                     all_steps=5, other_particle_steps=1, birth_keys_sha256=DIGEST,
                     closed_census_keys_exact=True, census_events_sha256="1" * 64))


def paths(value, prefix=()):
    for key, child in value.items():
        path = prefix + (key,)
        yield path
        if type(child) is dict:
            yield from paths(child, path)


def parent(data, path):
    for key in path[:-1]:
        data = data[key]
    return data


ALL_PATHS = list(paths(reports()))
# A historical partial cycle is optional, not independently an admission requirement.
REQUIRED_PATHS = [p for p in ALL_PATHS if p != ("census", "event_cycles", "Events;1")]
INT_PATHS = [p for p in ALL_PATHS if type(parent(reports(), p)[p[-1]]) is int]
BOOL_PATHS = [p for p in ALL_PATHS if type(parent(reports(), p)[p[-1]]) is bool]
HASH_PATHS = [p for p in ALL_PATHS if p[-1].endswith("sha256") or "input_sha256" in p[:-1]]
HASH_PATHS = [p for p in HASH_PATHS if type(parent(reports(), p)[p[-1]]) is str]


def test_synthetic_allpass_is_read_only_and_not_physics_closure():
    inputs = reports()
    original = deepcopy(inputs)
    result = C.admit(**inputs)
    assert inputs == original
    assert result["admitted"] is True and result["events"] == 3
    assert result["birth_keys_sha256"] == DIGEST
    assert "not physics closure" in result["validity_verdict"]
    assert "POT" in result["limitations"]


@pytest.mark.parametrize("path", REQUIRED_PATHS)
def test_every_required_key_missing(path):
    inputs = reports()
    del parent(inputs, path)[path[-1]]
    with pytest.raises((ValueError, TypeError)):
        C.admit(**inputs)


@pytest.mark.parametrize("path", ALL_PATHS)
def test_every_evidence_field_rejects_null(path):
    inputs = reports()
    parent(inputs, path)[path[-1]] = None
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("path", INT_PATHS)
@pytest.mark.parametrize("bad", [True, False, 1.0, "1", -1, [], {}])
def test_counts_are_native_nonnegative_integers_not_boolean(path, bad):
    inputs = reports()
    parent(inputs, path)[path[-1]] = bad
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("path", BOOL_PATHS)
@pytest.mark.parametrize("bad", [False, 0, 1, "true", [], {}])
def test_required_flags_are_literal_true(path, bad):
    inputs = reports()
    parent(inputs, path)[path[-1]] = bad
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("path", HASH_PATHS)
@pytest.mark.parametrize("bad", ["", "a" * 63, "a" * 65, "g" * 64, "a" * 63 + "\n", 123, True, []])
def test_all_digest_fields_require_64_hex(path, bad):
    inputs = reports()
    parent(inputs, path)[path[-1]] = bad
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("label,key", [("census", "birth_keys_sha256"),
                                      ("primary", "birth_keys_sha256"),
                                      ("source", "generated_keys_sha256")])
def test_unequal_identity_digests(label, key):
    inputs = reports()
    inputs[label][key] = "2" * 64
    with pytest.raises(ValueError, match="digests differ"):
        C.admit(**inputs)


@pytest.mark.parametrize("label,key", [("census", "events"), ("source", "events"),
                                      ("primary", "generated_starts"), ("primary", "external_expected_events"),
                                      ("primary", "primary_electrons"), ("census", "status_zero_events"),
                                      ("census", "inclusive_path_pass_events")])
@pytest.mark.parametrize("value", [0, 2, 4])
def test_exact_positive_counts(label, key, value):
    inputs = reports()
    inputs[label][key] = value
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("label,key", [("census", "validity_failure_events"),
                                      ("primary", "primary_step_events_without_vd13"),
                                      ("primary", "steps_before_first_vd13_in_crossing_events")])
def test_validity_and_boundary_exceptions_never_admitted(label, key):
    inputs = reports()
    inputs[label][key] = 1
    with pytest.raises(ValueError, match="zero required"):
        C.admit(**inputs)


@pytest.mark.parametrize("status", [{}, {"0": 2}, {"1": 3}, {"0": 3, "1": 0},
                                    {"00": 3}, {0: 3}, {"0": True}, {"0": 3.0}])
def test_native_status_counts_self_consistent(status):
    inputs = reports()
    inputs["census"]["status_counts"] = status
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("selected,rejected", [(0, 2), (2, 2), (4, 0), (0, 4)])
def test_stock_partition(selected, rejected):
    inputs = reports()
    inputs["census"].update(stock_selected_events=selected, stock_rejected_events=rejected)
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("cycles", [{}, {"Events": 3}, {"Events;0": 3}, {"Events;01": 3},
                                    {"Other;1": 3}, {"Events;1": 3, "Events;2": 2},
                                    {"Events;1": 4, "Events;2": 3}, {"Events;1": True},
                                    {"Events;1": 3.0}, {"Events;1\n": 3}])
def test_event_cycles_require_valid_counts_and_complete_highest(cycles):
    inputs = reports()
    inputs["census"]["event_cycles"] = cycles
    with pytest.raises(ValueError):
        C.admit(**inputs)


def test_numeric_highest_cycle_and_partial_earlier_cycle():
    inputs = reports()
    inputs["census"]["event_cycles"] = {"Events;2": 0, "Events;10": 3}
    assert C.admit(**inputs)["admitted"]


@pytest.mark.parametrize("change", [dict(primary_vd13=4), dict(primary_step_events=3),
                                    dict(primary_steps=1), dict(primary_steps=0),
                                    dict(all_steps=4), dict(other_particle_steps=2),
                                    dict(primary_step_events=0)])
def test_primary_summary_count_consistency(change):
    inputs = reports()
    inputs["primary"].update(change)
    with pytest.raises(ValueError):
        C.admit(**inputs)


@pytest.mark.parametrize("selected", [0, 3])
def test_no_response_and_stock_partition_endpoints_are_allowed(selected):
    inputs = reports()
    inputs["census"].update(stock_selected_events=selected, stock_rejected_events=3 - selected)
    inputs["primary"].update(primary_vd13=0, primary_steps=0, primary_step_events=0,
                              all_steps=0, other_particle_steps=0)
    assert C.admit(**inputs)["admitted"]


def cli_inputs(tmp_path):
    args, files = [], {}
    for label, report in reports().items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(report) + "\n")
        files[label] = path
        args.extend([f"--{label}", str(path)])
    return args, files, tmp_path / "admission.json"


def test_cli_hashes_exact_input_bytes_and_exclusive_output(tmp_path, capsys):
    args, files, out = cli_inputs(tmp_path)
    originals = {label: path.read_bytes() for label, path in files.items()}
    C.main(args + ["--out", str(out)])
    result = json.loads(out.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["input_sha256"] == {label: hashlib.sha256(data).hexdigest()
                                       for label, data in originals.items()}
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        C.main(args + ["--out", str(out)])
    assert out.read_bytes() == before
    with pytest.raises(FileExistsError):
        C.main(args + ["--out", str(files["census"])])
    assert originals == {label: path.read_bytes() for label, path in files.items()}


@pytest.mark.parametrize("text", ["{", "[]", "null", "true", '"__import__(\'os\').system(\'false\')"',
                                  '{"events": 3, "events": 3}', '{"events": NaN}',
                                  '{"events": Infinity}', '{"events": -Infinity}'])
def test_bad_json_and_wrong_report_types_create_no_output(tmp_path, text):
    args, files, out = cli_inputs(tmp_path)
    files["census"].write_text(text)
    with pytest.raises(ValueError):
        C.main(args + ["--out", str(out)])
    assert not out.exists()


def test_failed_admission_leaves_output_absent(tmp_path):
    args, files, out = cli_inputs(tmp_path)
    report = reports()["source"]
    report["source_window_complete"] = False
    files["source"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        C.main(args + ["--out", str(out)])
    assert not out.exists()
