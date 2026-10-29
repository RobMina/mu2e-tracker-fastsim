"""Synthetic captured-state diagnostics; no captured ROOT files or fit/native jobs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_captured_ca_audit as D

A, C = D.A, D.C


def change(table, field, value, row=0):
    records = ak.to_list(table)  # Synthetic fixture construction only, never the audit hot path.
    records[row][field] = value
    return ak.Array(records)


def independent_information(q, H, residual, weight):
    W = np.sum(weight[..., None, None] * H[..., :, None] * H[..., None, :], axis=1)
    b = (W @ q[..., None])[..., 0] - np.sum((weight * residual)[..., None] * H, axis=1)
    return W, b


def supplied_direction_oracle(q, line, state, mass):
    """Ordinary contractions, independent of the audit's source-grouped row code."""
    _, _, speed, dx, dm = A.kinematics(q, state["particle_time"], mass)
    u, s, delta = (state[k] for k in ("direction", "sensor_direction", "delta"))
    dot = np.sum(s * u, axis=1)
    den = 1. - dot**2
    norm = np.linalg.norm(delta, axis=1)
    sign = np.copysign(1., np.sum(np.cross(s, u) * delta, axis=1))
    contract = lambda v, m: np.einsum("ni,nij->nj", v, m)
    dD = contract(sign[:, None] * delta / norm[:, None], dx)
    beta = (s * dot[:, None] - u) / (speed * den)[:, None]
    gamma = (s - dot[:, None] * u) / (line["speed"] * den)[:, None]
    ds, dp, ddm = contract(s, dx), contract(u, dx), contract(delta, dm)
    Hlong = ds + ((dot[:, None] * ds - dp - ddm) / den[:, None]) * dot[:, None]
    return dict(H=np.stack((dD, -contract(beta - gamma, dx), Hlong), axis=1),
                doca=sign * norm, longitudinal=np.sum((state["sensor_poca"] - line["mid"]) * s, axis=1))


@pytest.fixture
def sample():
    q = np.array([[45., -1.1, 280., 1.4, 73000., 2000.],
                  [57., -1., 200., 1.1, 73005., 2030.]])
    mid = np.array([[250., 180., 390.], [290., 170., 480.]])
    direction = np.array([[.8, .6, 0.], [-.8, -.6, 0.]])
    line = A.sensor(mid + 500. * direction, mid - 500. * direction,
                    np.array([73003., 73010.]), np.array([180., 220.]))
    corrected, mass = np.array([73000., 73006.]), C.N.MASS
    hints = C.constructor_hint(q, line, corrected, mass)
    built = C.solve(q, line, *hints, mass=mass, precision=D.NATIVE_CA_PRECISION)
    local = C.solve(q, line, built["particle_time"], built["sensor_time"],
                    mass=mass, precision=D.NATIVE_CA_PRECISION)
    # Deliberately retained synthetic snapshots, not claims about physical native results.
    cap = deepcopy(local)
    cap["particle_time"] += [.015, -.02]
    cap["sensor_time"] += [.025, .03]
    cap["particle_poca"] = C.particle(q, cap["particle_time"], mass)[0] + [.2, -.3, .4]
    cap["sensor_poca"] = C.sensor_position(line, cap["sensor_time"]) + [-.3, .1, .2]
    cap["delta"] = cap["sensor_poca"] - cap["particle_poca"]
    source = A.rows_at_state(q, line, cap, mass)
    H = source["H"]
    residual = np.column_stack((-source["doca"], [2.5, -3.], [-7., 4.]))
    mvar, scale = np.array([[2., 4., 8.], [3., 5., 9.]]), np.array([2., 3.])
    weight = 1. / (mvar * scale[:, None])
    native_W, native_b = independent_information(q, H, residual, weight)
    records, native = [], []
    for i in range(2):
        # Same event/effect ordinal, distinct subrun: partial-key joins are invalid.
        identity = dict(run=8, subrun=i, event=42, iseed=3, iteration_ordinal=0,
                        effect_index=4, shindex=[21, 7][i], strawid=100 + i,
                        input_index=10 + i, miter=2, iter=0, active=True, in_fit=True)
        g = H[i] * np.array([1., -1., -1.])[:, None]
        records.append(dict(identity, ca_usable=True, constrain_tot=True, constrain_long=True,
                            constrain_absdrift=False, hit_state=0, row_active=[1, 1, 1],
                            precision=D.NATIVE_CA_PRECISION, row_derivative=g.ravel().tolist(),
                            row_value=(residual[i] * [-1., 1., 1.]).tolist(), row_mvar=mvar[i].tolist(),
                            row_pvar=[1e20, 1e30, 1e40], variance_scale=scale[i],
                            dDdP=g[0].tolist(), dTdP=g[1].tolist(), doca=source["doca"][i],
                            particle_poca=np.r_[cap["particle_poca"][i], cap["particle_time"][i]].tolist(),
                            sensor_poca=np.r_[cap["sensor_poca"][i], cap["sensor_time"][i]].tolist(),
                            particle_direction=cap["direction"][i].tolist(),
                            sensor_direction=cap["sensor_direction"][i].tolist()))
        native.append(dict(identity, time=cap["particle_time"][i], reference_parameters=q[i].tolist()))
    context = dict(native=ak.Array(native), native_mass=mass, q=q.copy(), hit_ids=np.array([21, 7]),
                   actual_H=H.copy(), residual=residual.copy(), weight=weight.copy(),
                   native_W=native_W, native_b=native_b)
    return SimpleNamespace(q=q, line=line, corrected=corrected, mass=mass, local=local, cap=cap,
                           rows=ak.Array(records), H=H, residual=residual, weight=weight, context=context)


def test_join_order_signs_weights_and_unused_sidecar_rows(sample, monkeypatch):
    extra = change(sample.rows[:1], "event", 99)
    extra = change(extra, "hit_state", -3)
    rows = ak.concatenate((extra, sample.rows[::-1]))

    def forbidden(*args, **kwargs):
        pytest.fail("audit hot path must not convert records with ak.to_list")

    monkeypatch.setattr(ak, "to_list", forbidden)
    selected, H, r, w = D.join_null_rows(rows, sample.context["native"])
    np.testing.assert_array_equal(D.keys(selected, D.HIT_KEYS), D.keys(sample.rows, D.HIT_KEYS))
    np.testing.assert_array_equal(H, sample.H)
    np.testing.assert_array_equal(r, sample.residual)
    np.testing.assert_array_equal(w, sample.weight)
    # row_pvar is deliberately huge: it is not part of the measurement weight.
    np.testing.assert_array_equal(H[:, 0], D._dense(selected, "dDdP", (6,)))
    np.testing.assert_array_equal(H[:, 1], -D._dense(selected, "dTdP", (6,)))
    np.testing.assert_array_equal(r[:, 0], -D._col(selected, "doca"))


@pytest.mark.parametrize("field", D.HIT_KEYS)
def test_join_uses_every_full_key_component(sample, field):
    decoy = change(sample.rows[:1], field, 99)
    rows = ak.concatenate((decoy, sample.rows[::-1]))
    selected, *_ = D.join_null_rows(rows, sample.context["native"])
    np.testing.assert_array_equal(D.keys(selected, D.HIT_KEYS), D.keys(sample.rows, D.HIT_KEYS))
    with pytest.raises(ValueError, match="missing.*identity"):
        D.join_null_rows(ak.concatenate((decoy, sample.rows[1:])), sample.context["native"])


@pytest.mark.parametrize("duplicate", ["selected", "unselected", "native"])
def test_join_rejects_duplicate_keys_even_outside_target(sample, duplicate):
    rows, native = sample.rows, sample.context["native"]
    if duplicate == "native":
        native = ak.concatenate((native, native[:1]))
    else:
        extra = rows[:1] if duplicate == "selected" else change(rows[:1], "event", 99)
        rows = ak.concatenate((rows, extra, extra))
    with pytest.raises(ValueError, match="duplicate"):
        D.join_null_rows(rows, native)


@pytest.mark.parametrize("field", ["shindex", "strawid", "input_index", "miter", "iter", "in_fit"])
def test_join_rechecks_effect_identity(sample, field):
    value = False if field == "in_fit" else 999
    with pytest.raises(ValueError, match="snapshot/effect differs: " + field):
        D.join_null_rows(change(sample.rows, field, value), sample.context["native"])


@pytest.mark.parametrize("field,value,message", [
    ("active", False, "active"), ("ca_usable", False, "ca_usable"),
    ("constrain_tot", False, "constrain_tot"), ("constrain_long", False, "constrain_long"),
    ("constrain_absdrift", True, "absolute drift"), ("hit_state", 1, "null hits"),
    ("hit_state", -1, "null hits"), ("row_active", [1, 0, 1], "all three"),
    ("precision", 1e-4, "precision"), ("precision", np.nan, "precision"),
    ("row_derivative", [np.nan] * 18, "nonfinite native derivative"),
    ("row_value", [1., np.inf, 2.], "nonfinite native residual"),
    ("row_mvar", [1., 0., 2.], "measurement variance"),
    ("row_mvar", [1., -1., 2.], "measurement variance"),
    ("row_mvar", [1., np.nan, 2.], "measurement variance"),
    ("variance_scale", 0., "scale"), ("variance_scale", np.inf, "scale"),
    ("dDdP", [0.] * 6, "cached dDdP"), ("dTdP", [0.] * 6, "cached dTdP"),
    ("doca", 0., "cached doca"),
])
def test_join_rejects_unsupported_or_inconsistent_null_rows(sample, field, value, message):
    with pytest.raises(ValueError, match=message):
        D.join_null_rows(change(sample.rows, field, value), sample.context["native"])


def test_snapshot_preserves_supplied_positions_times_and_directions(sample):
    state = D.snapshot_state(sample.rows[::-1])
    for field in ("particle_time", "sensor_time", "particle_poca", "sensor_poca", "direction",
                  "sensor_direction", "usable"):
        np.testing.assert_array_equal(state[field], sample.cap[field][::-1])
    np.testing.assert_array_equal(state["delta"], state["sensor_poca"] - state["particle_poca"])
    assert state["particle_poca"].shape == (2, 3)
    assert state["particle_time"].shape == (2,)


def test_fixed_state_matches_nativearith_when_directions_are_reconstructed(sample, monkeypatch):
    expected = A.rows_at_state(sample.q, sample.line, sample.cap, sample.mass)

    def forbidden(*args, **kwargs):
        pytest.fail("fixed-state evaluation must not solve or reconstruct CA")

    monkeypatch.setattr(C, "solve", forbidden)
    monkeypatch.setattr(C, "constructor_hint", forbidden)
    monkeypatch.setattr(C, "particle", forbidden)
    monkeypatch.setattr(C, "sensor_position", forbidden)
    actual = D.fixed_state_rows(sample.q, sample.line, sample.cap, sample.mass)
    for field in ("H", "doca", "longitudinal"):
        np.testing.assert_array_equal(actual[field], expected[field])


@pytest.mark.parametrize("field", ["direction", "sensor_direction"])
def test_fixed_state_actually_uses_each_supplied_direction(sample, monkeypatch, field):
    state = deepcopy(sample.cap)
    state[field] = np.roll(state[field], 1, axis=1)
    expected = supplied_direction_oracle(sample.q, sample.line, state, sample.mass)
    baseline = D.fixed_state_rows(sample.q, sample.line, sample.cap, sample.mass)

    def forbidden(*args, **kwargs):
        pytest.fail("fixed controls cannot call CA solve")

    monkeypatch.setattr(C, "solve", forbidden)
    result = D.fixed_state_rows(sample.q, sample.line, state, sample.mass)
    for key in ("H", "doca", "longitudinal"):
        np.testing.assert_allclose(result[key], expected[key], rtol=2e-13, atol=1e-10)
    assert not np.allclose(result["H"], baseline["H"], rtol=1e-5, atol=1e-7)
    if field == "sensor_direction":
        assert not np.allclose(result["longitudinal"], baseline["longitudinal"])


@pytest.mark.parametrize("failure,message", [("unusable", "unusable"), ("shape", "shape"),
                                               ("nonfinite", "nonfinite"), ("zero", "undefined"),
                                               ("parallel", "undefined")])
def test_fixed_state_rejects_invalid_controls(sample, failure, message):
    state = deepcopy(sample.cap)
    if failure == "unusable":
        state["usable"][0] = False
    elif failure == "shape":
        state["direction"] = state["direction"][:, :2]
    elif failure == "nonfinite":
        state["sensor_direction"][0, 1] = np.nan
    elif failure == "zero":
        state["delta"][0] = 0.
    else:
        state["direction"][0] = state["sensor_direction"][0] = [1., 0., 0.]
    with pytest.raises(ValueError, match=message):
        D.fixed_state_rows(sample.q, sample.line, state, sample.mass)


def test_compare_target_fixed_controls_and_captured_information_anchor(sample, monkeypatch):
    # Distinct saved reference/operands ensure captured arms do not silently use saved ones.
    context = deepcopy(sample.context)
    context["q"][:, 0] += 2.
    context["actual_H"] *= 1.2
    context["residual"] += 5.
    context["weight"] *= 1.7
    calls, fixed_calls, information_calls = [], [], []
    solve, fixed, information = C.solve, D.fixed_state_rows, D.information_report

    def record_solve(q, line, ph, sh, **kwargs):
        assert len(calls) < 6, "only three constructor/update pairs; no CA solve in fixed controls"
        calls.append((q.copy(), kwargs["precision"]))
        return solve(q, line, ph, sh, **kwargs)

    def record_fixed(q, line, state, mass):
        fixed_calls.append((q.copy(), deepcopy(state), len(calls)))
        return fixed(q, line, state, mass)

    def record_information(q, H, residual, weight, ctx):
        information_calls.append((q.copy(), H.copy(), residual.copy(), weight.copy()))
        return information(q, H, residual, weight, ctx)

    def forbidden(*args, **kwargs):
        pytest.fail("comparison hot path must not use ak.to_list")

    monkeypatch.setattr(C, "solve", record_solve)
    monkeypatch.setattr(D, "fixed_state_rows", record_fixed)
    monkeypatch.setattr(D, "information_report", record_information)
    monkeypatch.setattr(ak, "to_list", forbidden)
    result = D.compare_target(context, sample.rows[::-1], sample.line, sample.corrected)
    assert len(calls) == 6 and len(fixed_calls) == 8
    assert [x[1] for x in calls] == [1e-4, 1e-4, D.NATIVE_CA_PRECISION,
                                     D.NATIVE_CA_PRECISION, D.NATIVE_CA_PRECISION, D.NATIVE_CA_PRECISION]
    for q, _ in calls[:4]:
        np.testing.assert_array_equal(q, sample.q)
    for q, _ in calls[4:]:
        np.testing.assert_array_equal(q, context["q"])
    assert result["n_hits"] == 2 and result["source_shindex"] == [21, 7]
    np.testing.assert_array_equal(result["captured_H"], sample.H)
    np.testing.assert_array_equal(result["captured_residual"], sample.residual)
    np.testing.assert_array_equal(result["captured_weight"], sample.weight)
    local = fixed_calls[1][1]
    controls = result["captured_operand_controls"]
    settings = [
        ("captured_times_recomputed_positions", False, False, False, True),
        ("captured_particle_poca_only", True, False, False, False),
        ("captured_sensor_poca_only", False, True, False, False),
        ("captured_both_pocas", True, True, False, False),
        ("captured_complete_state", True, True, True, True),
    ]
    for (label, pp, sp, directions, times), (q, state, nsolves) in zip(settings, fixed_calls[3:]):
        assert nsolves == 6
        np.testing.assert_array_equal(q, sample.q)
        expected = deepcopy(local)
        if times:
            for field in ("particle_time", "sensor_time"):
                expected[field] = sample.cap[field]
            expected["particle_poca"] = C.particle(sample.q, expected["particle_time"], sample.mass)[0]
            expected["sensor_poca"] = C.sensor_position(sample.line, expected["sensor_time"])
        for field, use_capture in (("particle_poca", pp), ("sensor_poca", sp),
                                   ("direction", directions), ("sensor_direction", directions)):
            if use_capture:
                expected[field] = sample.cap[field]
        expected["delta"] = expected["sensor_poca"] - expected["particle_poca"]
        for field in D.snapshot_state(sample.rows):
            np.testing.assert_array_equal(state[field], expected[field])
        H = supplied_direction_oracle(q, sample.line, expected, sample.mass)["H"]
        np.testing.assert_allclose(controls[label]["H"], H, rtol=2e-13, atol=1e-10)
    # Every fixed row arm must consume captured r/w and the native reference,
    # including the dDdP-only splice, not the deliberately different saved data.
    expected_rows = [np.asarray(controls[label]["H"]) for label, *_ in settings]
    partial = fixed(sample.q, sample.line, local, sample.mass)["H"].copy()
    partial[:, 0] = sample.H[:, 0]
    expected_rows.extend((partial, sample.H))
    for H in expected_rows:
        assert any(np.array_equal(q, sample.q) and np.array_equal(actual_H, H)
                   and np.array_equal(r, sample.residual) and np.array_equal(w, sample.weight)
                   for q, actual_H, r, w in information_calls)
    for label in ("captured_complete_state", "captured_all_rows"):
        anchor = controls[label]["information"]
        assert anchor["weight_matrix"]["passed"] and anchor["weight_vector"]["passed"]
        assert anchor["failing_source_shindex"] == []
    assert controls["captured_all_rows"]["H_vs_captured"]["exact"]["passed"]
    assert not result["captured_rows_saved_values_weights"]["weight_matrix"]["passed"]
    assert not result["saved_rows_captured_values_weights"]["weight_matrix"]["passed"]
    assert not result["native_reference_vs_saved"]["criterion"]["passed"]
    # The comparison must not mutate the captured values.
    np.testing.assert_array_equal(sample.rows.row_value, sample.residual * [-1., 1., 1.])
    json.dumps(result, allow_nan=False)


def test_complete_control_uses_captured_directions_not_reconstructed_ones(sample):
    state = deepcopy(sample.cap)
    for field in ("direction", "sensor_direction"):
        state[field] = np.roll(state[field], 1, axis=1)
    oracle = supplied_direction_oracle(sample.q, sample.line, state, sample.mass)
    residual = sample.residual.copy()
    residual[:, 0] = -oracle["doca"]
    records = ak.to_list(sample.rows)
    for i, row in enumerate(records):
        g = oracle["H"][i] * np.array([1., -1., -1.])[:, None]
        row.update(particle_direction=state["direction"][i].tolist(),
                   sensor_direction=state["sensor_direction"][i].tolist(),
                   row_derivative=g.ravel().tolist(), dDdP=g[0].tolist(), dTdP=g[1].tolist(),
                   doca=oracle["doca"][i], row_value=(residual[i] * [-1., 1., 1.]).tolist())
    context = deepcopy(sample.context)
    context["native_W"], context["native_b"] = independent_information(
        sample.q, oracle["H"], residual, sample.weight)
    result = D.compare_target(context, ak.Array(records), sample.line, sample.corrected)
    controls = result["captured_operand_controls"]
    complete = controls["captured_complete_state"]
    np.testing.assert_allclose(complete["H"], oracle["H"], rtol=2e-13, atol=1e-10)
    assert complete["information"]["weight_matrix"]["passed"]
    assert complete["information"]["weight_vector"]["passed"]
    assert complete["information"]["failing_source_shindex"] == []
    assert controls["captured_all_rows"]["information"]["failing_source_shindex"] == []
    assert not controls["captured_both_pocas"]["H_vs_captured"]["criterion"]["passed"]
    assert not result["reconstructed_sensor_direction_vs_captured"]["criterion"]["passed"]


@pytest.mark.parametrize("operand", ["H", "residual", "weight"])
def test_information_report_keeps_failing_source_hit_ids(sample, operand):
    H, residual, weight = sample.H.copy(), sample.residual.copy(), sample.weight.copy()
    if operand == "H":
        H[1] *= 2.
    elif operand == "residual":
        residual[1] += 100.
    else:
        weight[1] *= 3.
    report = D.information_report(sample.q, H, residual, weight, sample.context)
    assert report["failing_source_shindex"] == [7]
    assert report["weight_matrix"]["passed"] == (operand == "residual")
    assert not report["weight_vector"]["passed"]


def test_compare_target_requires_exact_effect_time(sample, monkeypatch):
    sample.context["native"] = change(sample.context["native"], "time",
                                        np.nextafter(sample.cap["particle_time"][0], np.inf))

    def forbidden(*args, **kwargs):
        pytest.fail("bad effect time must fail before reconstruction")

    monkeypatch.setattr(C, "solve", forbidden)
    with pytest.raises(ValueError, match="particle time/effect differs"):
        D.compare_target(sample.context, sample.rows, sample.line, sample.corrected)


@pytest.fixture
def gate(tmp_path):
    check, frozen, evidence = (tmp_path / name for name in ("check.json", "frozen.root", "evidence.bin"))
    for path in (check, frozen, evidence):
        path.write_text("synthetic " + path.name)
    path = tmp_path / "operand-check.json"
    report = dict(passed=True, hashes_unchanged=True, all_four_evidence_hashes_verified=True,
                  capture_off_hit_operands_empty=True, check_sha256=D.digest(check),
                  reference_capture={"sha256": D.digest(frozen)},
                  original_tree_passivity={name: {"passed": True} for name in D.ORIGINAL_KEYS},
                  n_excluded_hits=0, n_hits=2, n_expected_hits=2, rtol=1e-5, atol=1e-7,
                  end_sha256={str(p.resolve()): D.digest(p) for p in (check, frozen, evidence)})
    path.write_text(json.dumps(report))
    return SimpleNamespace(path=path, check=check, frozen=frozen, evidence=evidence, report=report)


def test_receipt_binds_evidence_and_its_own_hash(gate):
    report, watched = D.receipt(gate.path, gate.check, gate.frozen)
    assert report["passed"]
    assert watched[str(gate.path.resolve())] == D.digest(gate.path)
    for path in (gate.check, gate.frozen, gate.evidence):
        assert watched[str(path.resolve())] == D.digest(path)


@pytest.mark.parametrize("flag", ["passed", "hashes_unchanged", "all_four_evidence_hashes_verified",
                                  "capture_off_hit_operands_empty"])
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_receipt_requires_explicit_true_flags(gate, flag, value):
    if value is None:
        del gate.report[flag]
    else:
        gate.report[flag] = value
    gate.path.write_text(json.dumps(gate.report))
    with pytest.raises(ValueError, match=flag):
        D.receipt(gate.path, gate.check, gate.frozen)


@pytest.mark.parametrize("failure,message", [
    ("check", "new gate differs"), ("frozen", "frozen reference differs"),
    ("evidence", "hash changed"), ("missing_tree", "original-tree"),
    ("extra_tree", "original-tree"), ("failed_tree", "original-tree"),
    ("excluded", "census"), ("count", "census"), ("rtol", "tolerances"),
    ("atol", "tolerances"), ("no_manifest", "hash manifest"),
])
def test_receipt_fails_closed_on_hashes_census_and_gate_policy(gate, failure, message):
    if failure in ("check", "frozen", "evidence"):
        getattr(gate, failure).write_text("changed bytes")
    elif failure == "missing_tree":
        gate.report["original_tree_passivity"].pop(next(iter(D.ORIGINAL_KEYS)))
    elif failure == "extra_tree":
        gate.report["original_tree_passivity"]["unexpected"] = {"passed": True}
    elif failure == "failed_tree":
        gate.report["original_tree_passivity"][next(iter(D.ORIGINAL_KEYS))]["passed"] = False
    elif failure == "excluded":
        gate.report["n_excluded_hits"] = 1
    elif failure == "count":
        gate.report["n_hits"] = 1
    elif failure in ("rtol", "atol"):
        gate.report[failure] *= 10
    else:
        gate.report.pop("end_sha256")
    gate.path.write_text(json.dumps(gate.report))
    with pytest.raises(ValueError, match=message):
        D.receipt(gate.path, gate.check, gate.frozen)


@pytest.mark.parametrize("value", [None, True, 2., "2", 0])
def test_receipt_requires_explicit_positive_integer_counts(gate, value):
    for field in ("n_hits", "n_expected_hits"):
        if value is None:
            gate.report.pop(field)
        else:
            gate.report[field] = value
    gate.path.write_text(json.dumps(gate.report))
    with pytest.raises(ValueError, match="census"):
        D.receipt(gate.path, gate.check, gate.frozen)


@pytest.fixture
def speed_fcl(tmp_path):
    path = tmp_path/"expanded.fcl"
    path.write_text("""services: { strawResponse: {
        eBinWidth: 0.5
        eBins: 4
        defaultHalfPropVelocity: 80
        eHalfPVScale: [0.7, 0.9, 1.0, 1.2]
        strawHalfPropVelocity: []
        useDb: false
    } }
""")
    return path


def test_source_speed_configuration_and_independent_interpolation(speed_fcl):
    config = D.source_speed_configuration(speed_fcl)
    assert config["sha256"] == D.digest(speed_fcl)
    energy = np.array([0., .00025, .0005, .00125, .002])
    actual = D.source_signal_speed(energy, config)
    np.testing.assert_allclose(actual, 160.*np.array([.7, .8, .9, 1.1, 1.4]), rtol=2e-16)


@pytest.mark.parametrize("before,after", [
    ("useDb: false", "useDb: true"), ("strawHalfPropVelocity: []", "strawHalfPropVelocity: [80]"),
    ("eBins: 4", "eBins: 3"), ("eBinWidth: 0.5", "eBinWidth: 0"),
    ("defaultHalfPropVelocity: 80", "defaultHalfPropVelocity: nan"),
    ("eBins: 4", "eBins: 4\neBins: 4"),
    ("eHalfPVScale: [0.7, 0.9, 1.0, 1.2]", "eHalfPVScale: [0.7, nan, 1.0, 1.2]")])
def test_speed_configuration_refuses_unsupported_inputs(speed_fcl, before, after):
    speed_fcl.write_text(speed_fcl.read_text().replace(before, after))
    with pytest.raises(ValueError):
        D.source_speed_configuration(speed_fcl)


@pytest.mark.parametrize("energy", [[np.nan], [np.inf], [-.001], [[.001]]])
def test_source_speed_rejects_invalid_measured_energy(speed_fcl, energy):
    with pytest.raises(ValueError, match="energy"):
        D.source_signal_speed(energy, D.source_speed_configuration(speed_fcl))


def test_speed_uses_source_slope_before_multiplication(speed_fcl):
    config = D.source_speed_configuration(speed_fcl)
    energy = np.array([.000123456789, .000787654321, .00162345987])
    grid = np.arange(4)*config["bin_width"]
    expected = []
    for value in energy:
        x = value*1000.
        width = (grid[-1]-grid[0])/3
        i = min(2, max(0, int(np.floor((x-grid[0])/width))))
        slope = (config["scale"][i+1]-config["scale"][i])/width
        y = config["scale"][i]+(x-grid[i])*slope
        expected.append(2.*(y*config["mean_half_velocity"]))
    np.testing.assert_array_equal(D.source_signal_speed(energy, config), expected)


def test_speed_controls_isolate_speed_and_reference_origins(sample, speed_fcl, monkeypatch):
    context = deepcopy(sample.context)
    context["q"][:, 0] += 1.
    context["h"] = dict(edep=np.array([.0007, .0013]), totdt=np.array([4., 6.]),
                        wdist=np.array([1., -1.]), esign=np.array([1., -1.]), udir_dot_w=np.ones(2))
    cond = dict(straw={"halfvp": np.array([70., 80.])}, halfvpscale=np.array([1., 1.2, 1.1, 1.3]))
    before_line = deepcopy(sample.line)
    calls = []
    solve = C.solve

    def record(q, line, ph, sh, **kwargs):
        calls.append((q.copy(), line["speed"].copy(), kwargs["precision"]))
        return solve(q, line, ph, sh, **kwargs)

    monkeypatch.setattr(C, "solve", record)
    result = D.speed_controls(context, sample.rows, sample.line, sample.corrected, context["h"]["edep"],
                             np.array([0, 1]), cond, D.source_speed_configuration(speed_fcl))
    assert len(calls) == 10 and len(result["variants"]) == 5
    for q, speed, precision in calls[:8]:
        np.testing.assert_array_equal(q, sample.q)
        assert precision == D.NATIVE_CA_PRECISION
    for q, speed, precision in calls[8:]:
        np.testing.assert_array_equal(q, context["q"])
    np.testing.assert_array_equal(calls[6][1], calls[8][1])
    for field in sample.line:
        np.testing.assert_array_equal(sample.line[field], before_line[field])
    json.dumps(result, allow_nan=False)


def test_integer_absolute_drift_flag_is_rejected(sample):
    rows = ak.with_field(sample.rows, np.array([1, 0], dtype=np.int32), "constrain_absdrift")
    with pytest.raises(ValueError, match="absolute drift"):
        D.join_null_rows(rows, sample.context["native"])


def test_fixed_state_rejects_finite_overflow(sample):
    state = deepcopy(sample.cap)
    state["delta"][:] = 1e308
    with pytest.raises(ValueError, match="nonfinite"):
        D.fixed_state_rows(sample.q, sample.line, state, sample.mass)


def test_speed_clips_segment_before_integer_cast(speed_fcl):
    config = D.source_speed_configuration(speed_fcl)
    energy = np.array([1e20])
    value = D.source_signal_speed(energy, config)
    slope = (1.2-1.)/.5
    expected = 2.*((1.+(energy*1000.-1.)*slope)*80.)
    np.testing.assert_array_equal(value, expected)
    with pytest.raises(ValueError, match="nonfinite"):
        D.source_signal_speed(np.array([1e308]), config)


def run_args(tmp_path):
    return SimpleNamespace(out=tmp_path / "audit.json", trace_dir=[tmp_path / "trace"],
                           check=tmp_path / "check.json", operand_check=tmp_path / "operand-check.json",
                           native_source=tmp_path / "native", allow_loader_reconstruction=False)


def test_run_exclusively_creates_report_before_loading_evidence(tmp_path, monkeypatch):
    args = run_args(tmp_path)
    original = b'{"sentinel": "keep exact bytes"}\n'
    args.out.write_bytes(original)

    def forbidden(*args, **kwargs):
        pytest.fail("existing output must be rejected before evidence loading")

    monkeypatch.setattr(D, "load_evidence", forbidden)
    with pytest.raises((FileExistsError, ValueError)):
        D.run(args)
    assert args.out.read_bytes() == original


def test_run_writes_explicit_failure_without_claiming_closure(tmp_path, monkeypatch):
    args = run_args(tmp_path)
    monkeypatch.setenv("F1_SYNTHETIC_NONDEFAULT", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("nondefault environment must fail before evidence loading")

    monkeypatch.setattr(D, "load_evidence", forbidden)
    result = D.run(args)
    saved = json.loads(args.out.read_text())
    for report in (result, saved):
        assert not report["completed"] and not report["full_closure"]
        assert not report["full_numerical_validation"]
        assert "default F1 environment" in report["error"]
        assert report["targets"] == []


def test_run_detects_hash_change_even_when_work_already_failed(tmp_path, monkeypatch):
    args = run_args(tmp_path)
    for key in list(D.os.environ):
        if key.startswith("F1_"):
            monkeypatch.delenv(key)
    original_digest = D.digest
    counts = {}

    def changed_digest(path):
        key = str(Path(path).resolve())
        counts[key] = counts.get(key, 0) + 1
        # Simulate mutation; never modify real repository sources.
        return original_digest(path) if counts[key] == 1 else "0" * 64

    def fail_evidence(*args, **kwargs):
        raise ValueError("synthetic evidence failure")

    monkeypatch.setattr(D, "digest", changed_digest)
    monkeypatch.setattr(D, "load_evidence", fail_evidence)
    report = D.run(args)
    assert not report["completed"] and not report["hashes_unchanged"]
    assert "synthetic evidence failure" in report["error"]
    assert "changed during audit" in report["integrity_error"]
    assert not report["full_closure"] and not report["full_numerical_validation"]
    assert json.loads(args.out.read_text())["hashes_unchanged"] is False
