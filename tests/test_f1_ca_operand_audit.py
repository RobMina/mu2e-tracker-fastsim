"""Synthetic CA operand policy and endpoint joins; no captured ROOT data or fit reruns."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import f1_ca_operand_audit as A

C = A.C


def independent_information(q, H, residual, weight):
    """Outer-product normal equations, independent of the production contraction order."""
    W = np.sum(weight[..., None, None] * H[..., :, None] * H[..., None, :], axis=-3)
    b = (W @ q[..., None])[..., 0] - np.sum((weight * residual)[..., None] * H, axis=-2)
    return W, b


@pytest.fixture
def operands():
    q = np.array([[45., -1.1, 280., 1.4, 73000., 2000.],
                  [57., -1., 200., 1.1, 73005., 2030.]])
    mid = np.array([[250., 180., 390.], [290., 170., 480.]])
    direction = np.array([[.8, .6, 0.], [-.8, -.6, 0.]])
    line = C.sensor(mid + 500. * direction, mid - 500. * direction,
                    np.array([73003., 73010.]), np.array([180., 220.]))
    corrected = np.array([73000., 73006.])
    actual_H = (np.arange(36, dtype=float).reshape(2, 3, 6) - 13.) / 8.
    residual = np.array([[3., 8., -12.], [-5., 11., 7.]])
    weight = np.array([[.2, .03, .001], [.4, .07, .002]])
    native_W, native_b = independent_information(q, actual_H, residual, weight)
    native = np.zeros(2, dtype=[("time", "f8"), ("strawid", "i8")])
    native["time"] = q[:, 4]
    native["strawid"] = [0x2024, 0x1421]
    context = {"actual_H": actual_H, "residual": residual, "weight": weight,
               "native_W": native_W, "native_b": native_b, "native": native,
               "native_mass": C.N.MASS,
               "h": {"totdt": np.array([9., 14.]), "wdist": np.array([23., -41.]),
                     "esign": np.array([1., -1.]), "udir_dot_w": np.array([-.4, -.2])}}
    return q, line, corrected, context


def reconstructed(q, line, corrected, mass=C.N.MASS, initial=None):
    hints = C.constructor_hint(q, line, corrected, mass) if initial is None else initial
    built = C.solve(q, line, *hints, mass=mass)
    state = C.solve(q, line, built["particle_time"], built["sensor_time"], mass=mass)
    return built, state, C.rows_at_state(q, line, state, mass)


def arrays_in(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from arrays_in(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from arrays_in(item)


def test_ca_variant_isolates_rows_and_rebuilds_values_with_saved_weights(operands, monkeypatch):
    q, line, corrected, context = operands
    inputs = list(arrays_in(operands))
    before = [value.copy() for value in inputs]
    for value in inputs:
        value.setflags(write=False)
    calls = []
    substitute = A.substitutions

    def record(parameters, actual, source, residual, weight, native_W, native_b):
        assert parameters is q and actual is context["actual_H"]
        assert weight is context["weight"]
        assert native_W is context["native_W"] and native_b is context["native_b"]
        calls.append((source.copy(), residual.copy(), residual is context["residual"]))
        return substitute(parameters, actual, source, residual, weight, native_W, native_b)

    monkeypatch.setattr(A, "substitutions", record)
    result = A.ca_variant(*operands)
    assert len(calls) == 2
    H = np.asarray(result["H"])
    np.testing.assert_array_equal(calls[0][0], H)
    np.testing.assert_array_equal(calls[1][0], H)
    assert np.all(np.any(H != context["actual_H"], axis=-1))
    assert calls[0][2] and not calls[1][2]
    np.testing.assert_array_equal(calls[0][1], context["residual"])
    sensor_poca = C.sensor_position(line, np.asarray(result["sensor_time"]))
    longitudinal = np.sum((sensor_poca - line["mid"]) * line["direction"], axis=-1)
    expected_values = np.column_stack((
        -np.asarray(result["doca"]),
        np.asarray(result["sensor_time"]) - result["particle_time"] - context["h"]["totdt"],
        longitudinal - context["h"]["wdist"] * np.array([-1., 1.]),
    ))
    np.testing.assert_array_equal(calls[1][1], expected_values)
    assert np.all(expected_values != context["residual"])
    matrices, vectors = [], []
    for label, (_, residual, _) in zip(("row_only", "rows_and_values"), calls):
        W, b = C.N.information(q, H, residual, context["weight"])
        oracle_W, oracle_b = independent_information(q, H, residual, context["weight"])
        np.testing.assert_allclose(W, oracle_W, rtol=2e-13, atol=1e-10)
        np.testing.assert_allclose(b, oracle_b, rtol=2e-12, atol=1e-6)
        assert result[label]["weight_matrix"] == A._metric(W, context["native_W"])
        assert result[label]["weight_vector"] == A._metric(b, context["native_b"])
        matrices.append(W)
        vectors.append(b)
    np.testing.assert_array_equal(matrices[0], matrices[1])
    assert np.any(vectors[0] != vectors[1])
    assert result["value_vs_saved"] == A._metric(expected_values, context["residual"])
    for value, saved in zip(inputs, before):
        np.testing.assert_array_equal(value, saved)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("changed", ["residual", "weight", "actual_H", "totdt", "wdist"])
def test_saved_and_reconstructed_arms_depend_only_on_their_operands(operands, changed):
    baseline = A.ca_variant(*operands)
    q, line, corrected, context = deepcopy(operands)
    if changed in ("totdt", "wdist"):
        context["h"][changed] += 300.
    elif changed == "weight":
        context[changed] *= 3.
    else:
        context[changed] += 300.
    updated = A.ca_variant(q, line, corrected, context)
    np.testing.assert_array_equal(updated["H"], baseline["H"])
    np.testing.assert_array_equal(updated["particle_time"], baseline["particle_time"])
    if changed == "residual":
        assert updated["rows_and_values"] == baseline["rows_and_values"]
        assert updated["row_only"]["weight_matrix"] == baseline["row_only"]["weight_matrix"]
        assert updated["row_only"]["weight_vector"] != baseline["row_only"]["weight_vector"]
    elif changed in ("totdt", "wdist"):
        assert updated["row_only"] == baseline["row_only"]
        assert updated["rows_and_values"]["weight_matrix"] == baseline["rows_and_values"]["weight_matrix"]
        assert updated["rows_and_values"]["weight_vector"] != baseline["rows_and_values"]["weight_vector"]
    elif changed == "actual_H":
        assert updated["row_only"] == baseline["row_only"]
        assert updated["rows_and_values"] == baseline["rows_and_values"]
    else:
        for label in ("row_only", "rows_and_values"):
            assert updated[label]["weight_matrix"] != baseline[label]["weight_matrix"]
            assert updated[label]["weight_vector"] != baseline[label]["weight_vector"]


@pytest.mark.parametrize("explicit_hint", [False, True])
@pytest.mark.parametrize("mass", [C.N.MASS, 200.])
def test_constructor_then_update_replay_and_explicit_hint_mass(operands, monkeypatch, explicit_hint, mass):
    q, line, corrected, context = operands
    initial = (corrected - 2., corrected + 1.) if explicit_hint else None
    built, state, rows = reconstructed(q, line, corrected, mass, initial)
    expected_hint = C.constructor_hint(q, line, corrected, mass) if initial is None else initial
    solve = C.solve
    calls = []

    def record(parameters, sensor, ph, sh, *, mass):
        assert parameters is q and sensor is line
        calls.append((ph.copy(), sh.copy(), mass))
        return solve(parameters, sensor, ph, sh, mass=mass)

    def forbidden_constructor(*args, **kwargs):
        pytest.fail("explicit captured-time diagnostic must not use the measured constructor hint")

    monkeypatch.setattr(C, "solve", record)
    if explicit_hint:
        monkeypatch.setattr(C, "constructor_hint", forbidden_constructor)
    snapshots = [(a, a.copy()) for a in arrays_in((operands, initial))]
    result = A.ca_variant(*operands, initial=initial, mass=mass)
    assert len(calls) == 2
    for actual, expected in zip(calls[0][:2], expected_hint):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(calls[1][0], built["particle_time"])
    np.testing.assert_array_equal(calls[1][1], built["sensor_time"])
    assert [call[2] for call in calls] == [mass, mass]
    for field in ("particle_time", "sensor_time", "delta", "status"):
        np.testing.assert_array_equal(result[field], state[field])
    np.testing.assert_array_equal(result["H"], rows["H"])
    np.testing.assert_array_equal(result["constructor_updates"], built["updates"])
    np.testing.assert_array_equal(result["update_updates"], state["updates"])
    assert bool(result["one_ulp_sensitivity"]) == (mass == context["native_mass"])
    for value, saved in snapshots:
        np.testing.assert_array_equal(value, saved)


def test_one_ulp_reports_all_fixed_shifts_against_unshifted_not_native_or_selected(operands):
    q, line, corrected, context = operands
    _, state, rows = reconstructed(q, line, corrected)
    baseline_W, baseline_b = C.N.information(q, rows["H"], context["residual"], context["weight"])
    expected = {}
    for field in ("particle_time", "sensor_time"):
        for label, toward in (("minus", -np.inf), ("plus", np.inf)):
            shifted = deepcopy(state)
            shifted[field] = np.nextafter(state[field], toward)
            shifted["particle_poca"] = C.particle(q, shifted["particle_time"])[0]
            shifted["sensor_poca"] = C.sensor_position(line, shifted["sensor_time"])
            shifted["delta"] = shifted["sensor_poca"] - shifted["particle_poca"]
            H = C.rows_at_state(q, line, shifted)["H"]
            W, b = C.N.information(q, H, context["residual"], context["weight"])
            dt = shifted[field] - state[field]
            assert np.all(dt < 0) if label == "minus" else np.all(dt > 0)
            expected[field + "_" + label + "_one_ulp"] = {
                "time_delta": dt.tolist(), "row_vs_unshifted": A._metric(H, rows["H"]),
                "weight_matrix_vs_unshifted": A._metric(W, baseline_W),
                "weight_vector_vs_unshifted": A._metric(b, baseline_b),
            }
    result = A.ca_variant(*operands)
    assert result["one_ulp_sensitivity"] == expected
    assert len(expected) == 4
    assert any(item["row_vs_unshifted"]["max_absolute_residual"] > 0 for item in expected.values())
    context["native_W"], context["native_b"] = W, b
    context["native"]["time"] = np.nextafter(state["particle_time"], np.inf)
    targeted = A.ca_variant(*operands)
    assert targeted["one_ulp_sensitivity"] == expected
    for field in ("particle_time", "sensor_time"):
        np.testing.assert_array_equal(targeted[field], state[field])
    np.testing.assert_array_equal(targeted["H"], rows["H"])
    assert targeted["row_only"]["weight_matrix"] == A._metric(baseline_W, context["native_W"])
    assert targeted["row_only"]["weight_vector"] == A._metric(baseline_b, context["native_b"])
    assert targeted["time_vs_native"]["passed"]
    exact = targeted["time_vs_native_exact"]
    assert not exact["passed"] and exact["mismatches"] == len(q)
    assert exact == A._metric(state["particle_time"], context["native"]["time"], exact=True)


@pytest.fixture
def endpoint_file(tmp_path):
    path = tmp_path / "synthetic-endpoints.root"
    geometry = {"sid": np.array([[0x1C23, 0x2024, 0x1421, 0x18A2]], dtype=np.int64)}
    cal = np.array([[31., 32., 33.], [41., 42., 43.], [11., 12., 13.], [21., 22., 23.]])
    hv = cal + np.array([100., -200., 300.])
    for prefix, values in (("wcal", cal), ("whv", hv)):
        for i, axis in enumerate("xyz"):
            geometry[prefix + axis] = values[None, :, i]
    events = {
        "s_sid": ak.Array([[0x18A2], [0x1421, 0x18A2, 0x2024]]),
        "s_time": ak.Array([[999.], [101.25, 202.5, 303.75]]),
        "s_tmeas": ak.Array([[9999.], [1101., 1202., 1303.]]),
        "s_edep": ak.Array([[99.], [.01, .02, .03]]),
    }
    native = np.array([(0x2024,), (0x1421,)], dtype=[("strawid", "i8")])
    context = {"event_index": 1, "hit_ids": np.array([2, 0]), "native": native,
               "h": {"esign": np.array([1., -1.])}}

    def write_file():
        with uproot.recreate(path) as root:
            conditions = root.mktree("trkdump/conditions", {
                key: "4 * int64" if key == "sid" else "4 * float64" for key in geometry})
            conditions.extend(geometry)
            tree = root.mktree("trkdump/events", {
                key: "var * int64" if key == "s_sid" else "var * float64" for key in events})
            tree.extend(events)

    write_file()
    return path, context, geometry, write_file


def test_endpoint_inputs_joins_raw_straw_ids_in_hit_order_and_uses_s_time(endpoint_file):
    path, context, _, _ = endpoint_file
    before = deepcopy(context)
    with uproot.open(path) as root:
        assert root["trkdump/conditions"].classname == "TTree"
        assert root["trkdump/events"].classname == "TTree"
    early, late, corrected, energy, sid = A.endpoint_inputs(path, context)
    np.testing.assert_array_equal(early, [[41., 42., 43.], [111., -188., 313.]])
    np.testing.assert_array_equal(late, [[141., -158., 343.], [11., 12., 13.]])
    np.testing.assert_array_equal(corrected, [303.75, 101.25])
    np.testing.assert_array_equal(energy, [.03, .01])
    np.testing.assert_array_equal(sid, [0x2024, 0x1421])
    for value, saved in zip(arrays_in(context), arrays_in(before)):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize("failure,message", [
    ("duplicate", "duplicate endpoint straw identity"),
    ("missing", "missing wire endpoint"),
    ("native_mismatch", "native/source hit straw differs"),
    ("missing_event", "missing target event"),
])
def test_endpoint_inputs_fail_closed_on_identity_errors(endpoint_file, failure, message):
    path, context, geometry, write_file = endpoint_file
    if failure == "duplicate":
        geometry["sid"][0, 3] = geometry["sid"][0, 0]
    elif failure == "missing":
        geometry["sid"][0, 1] = 0x7777
    elif failure == "native_mismatch":
        context["native"]["strawid"][0] = 0x18A2
    else:
        context["event_index"] = 2
    write_file()
    with pytest.raises(ValueError, match=message):
        A.endpoint_inputs(path, context)


def test_run_never_overwrites_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    original = b'{"sentinel": "preserve these exact bytes"}\n'
    output.write_bytes(original)

    def forbidden(*args, **kwargs):
        pytest.fail("existing output must be rejected before loading evidence")

    monkeypatch.setattr(A, "load_evidence", forbidden)
    args = SimpleNamespace(out=output, trace_dir=[tmp_path / "unused-trace"],
                           check=tmp_path / "unused-check", native_source=tmp_path / "unused-native",
                           allow_loader_reconstruction=False)
    with pytest.raises(ValueError, match="refusing existing CA audit output"):
        A.run(args)
    assert output.read_bytes() == original


def test_fixed_state_arithmetic_isolated_and_all_predeclared_arms(operands):
    from trkfastsim.reco import nativearith as AR
    q, line, corrected, context = operands
    before = [(a, a.copy()) for a in arrays_in(operands)]
    prior = A.ca_variant(*operands)
    result = A.ca_variant(*operands, arithmetic=True)
    for field in ("row_only", "rows_and_values", "H", "delta", "particle_time", "sensor_time"):
        assert result[field] == prior[field]
    arithmetic = result["fixed_state_arithmetic"]
    assert not arithmetic["CA_state_changed"] and not arithmetic["values_or_weights_changed"]
    _, state, rows = reconstructed(q, line, corrected)
    ordered = AR.rows_at_state(q, line, state)
    for name, H, info in (("source_rows_only", ordered["H"], C.N.information),
                          ("source_information_only", rows["H"], AR.information),
                          ("source_rows_and_information", ordered["H"], AR.information)):
        W, b = info(q, H, context["residual"], context["weight"])
        assert arithmetic["variants"][name]["weight_matrix"] == A._metric(W, context["native_W"])
        assert arithmetic["variants"][name]["weight_vector"] == A._metric(b, context["native_b"])
    assert len(arithmetic["variants"]) == 3
    for value, saved in before:
        np.testing.assert_array_equal(value, saved)


def test_root_header_provenance_requires_unambiguous_build_include(tmp_path):
    build = tmp_path/"kinkal_build_p109"
    build.mkdir()
    inc = tmp_path/"install/include/root"
    headers = inc/"Math/GenVector"
    headers.mkdir(parents=True)
    for name in ("DisplacementVector3D.h", "Cartesian3D.h"):
        (headers/name).write_text(name)
    command = build/"compile_commands.json"
    command.write_text(json.dumps([{"command": f"c++ -I{inc} source.cc"}]))
    got = A.root_arithmetic_sources(tmp_path)
    assert len(got) == 3 and got[str(command)] == A.digest(command)
    command.write_text(json.dumps([{"command": f"c++ -I{inc} -I/other/include/root source.cc"}]))
    with pytest.raises(ValueError, match="ambiguous frozen ROOT"):
        A.root_arithmetic_sources(tmp_path)
