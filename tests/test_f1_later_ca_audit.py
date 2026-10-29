"""Synthetic later-CA parent audit contracts; no captures, builds or remote IO."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_later_ca_audit as D


def adjacent_fixture():
    # Every target-key component matters; input identities repeat across targets.
    targets = [(1, 2, 3, 4), (9, 2, 3, 4), (1, 9, 3, 4),
               (1, 2, 9, 4), (1, 2, 3, 9)]
    stages = [(0, 0, True), (0, 1, False), (0, 2, True),
              (1, 0, True), (1, 2, True), (1, 3, True)]
    iterations, effects, rows = [], [], []
    for target in targets:
        for ordinal, (miter, iteration, swapped) in enumerate(stages):
            base = dict(zip(D.KEYS, target), iteration_ordinal=ordinal,
                        miter=miter, iter=iteration)
            iterations.append(dict(base, swapped=swapped))
            for source in (7, 8):
                row = dict(base, identity_kind=1, effect_index=source + 20 * ordinal,
                           input_index=source, shindex=100 + source,
                           strawid=200 + source,
                           has_reference=not (ordinal == 5 and source == 8),
                           marker=1000 * len(targets) + len(rows))
                effects.append(row)
                rows.append(dict(row))
            # Non-hit effects must never enter the sidecar census or join.
            effects.append(dict(base, identity_kind=2, effect_index=99,
                                input_index=99, shindex=99, strawid=99,
                                has_reference=True, marker=-1))
    return effects, iterations, rows


def call_adjacent(effects, iterations, rows):
    return D.adjacent_hits(ak.Array(effects), ak.Array(iterations), ak.Array(rows))


def test_adjacent_fullkey_reordered_sidecar_and_effect_order():
    effects, iterations, rows = adjacent_fixture()
    # Neither owner-table order, effect order nor sidecar order is a join key.
    effects = effects[::-1]
    iterations = iterations[::2] + iterations[1::2]
    before, after, row0, row1, parent, census = call_adjacent(effects, iterations, rows[::-1])
    expected = [r for r in effects if r["identity_kind"] == 1 and
                (r["iteration_ordinal"] == 0 or
                 (r["iteration_ordinal"] == 4 and r["input_index"] == 7))]
    assert ak.to_list(before) == expected
    assert ak.to_list(row0.marker) == [r["marker"] for r in expected]
    assert ak.to_list(row1.marker) == ak.to_list(after.marker)
    for old, new, owner in zip(ak.to_list(before), ak.to_list(after), parent):
        assert all(old[k] == new[k] == iterations[owner][k] for k in D.KEYS)
        assert new["iteration_ordinal"] == old["iteration_ordinal"] + 1
        assert new["miter"] == old["miter"]
        assert new["iter"] == old["iter"] + 1
        assert new["input_index"] == old["input_index"]
        assert new["effect_index"] != old["effect_index"]
    assert {k: census[k] for k in ("n_all_snapshots", "n_adjacent_updates",
            "n_not_selected", "n_no_next", "n_next_other_stage", "n_same_stage_no_swap")} == dict(
        n_all_snapshots=60, n_adjacent_updates=15, n_not_selected=45,
        n_no_next=10, n_next_other_stage=20, n_same_stage_no_swap=10)
    assert "outside" in census["scope"]


@pytest.mark.parametrize("mutation,message", [
    ("duplicate_effect", "duplicate identities"),
    ("duplicate_sidecar", "duplicate identities"),
    ("duplicate_iteration", "duplicate identities"),
    ("duplicate_source", "duplicate available identities"),
    ("missing_sidecar", "census differs"),
    ("missing_iteration", "missing full identity join"),
    ("wrong_sidecar_key", "missing full identity join"),
    ("changed_shindex", "adjacent source-hit identity changed"),
    ("changed_strawid", "adjacent straw identity changed"),
])
def test_adjacent_rejects_ambiguous_missing_or_changed_identity(mutation, message):
    effects, iterations, rows = adjacent_fixture()
    if mutation == "duplicate_effect":
        effects[1] = deepcopy(effects[0])
    elif mutation == "duplicate_sidecar":
        rows[1] = deepcopy(rows[0])
    elif mutation == "duplicate_iteration":
        iterations.append(deepcopy(iterations[0]))
    elif mutation == "duplicate_source":
        effects[1]["input_index"] = rows[1]["input_index"] = effects[0]["input_index"]
    elif mutation == "missing_sidecar":
        rows.pop()
    elif mutation == "missing_iteration":
        iterations.pop(0)
    elif mutation == "wrong_sidecar_key":
        rows[0]["run"] = 1000
    else:
        field = mutation.removeprefix("changed_")
        # Both captures agree on the changed value; the adjacency must still fail.
        effects[3][field] += 1
        rows[2][field] += 1
    with pytest.raises(ValueError, match=message):
        call_adjacent(effects, iterations, rows)


@pytest.mark.parametrize("field", ["input_index", "shindex", "strawid", "miter", "iter"])
def test_adjacent_sidecar_identity_fields_are_checked(field):
    effects, iterations, rows = adjacent_fixture()
    rows[0][field] += 1
    with pytest.raises(ValueError, match="sidecar identity differs: " + field):
        call_adjacent(effects, iterations, rows)


def piece_fixture():
    iterations = [dict(run=1, subrun=2, event=e, iseed=0, iteration_ordinal=0)
                  for e in (3, 4, 5)]
    pieces = [dict(iterations[owner], owner=1, piece_index=index,
                   parameters=[0., 0., 0., 0., float(index), 2000.],
                   range=[float(index), float(index + 1)])
              for owner, index in [(0, 1), (1, 0), (0, 0)]]
    # Another owner family is not a candidate, even with matching keys/bad operands.
    pieces.append(dict(pieces[0], owner=0, parameters=[float("nan")] * 6))
    return pieces, iterations


def test_pad_candidates_fullkey_owner_filter_and_index_placement():
    pieces, iterations = piece_fixture()
    par, ranges, counts = D.pad_candidates(ak.Array(pieces), ak.Array(iterations[::-1]))
    np.testing.assert_array_equal(counts, [0, 1, 2])
    assert par.shape == (3, 2, 6) and ranges.shape == (3, 2, 2)
    np.testing.assert_array_equal(par[2, 1], pieces[0]["parameters"])
    np.testing.assert_array_equal(par[1, 0], pieces[1]["parameters"])
    np.testing.assert_array_equal(par[0], 0.)
    np.testing.assert_array_equal(ranges[2], [[0., 1.], [1., 2.]])


@pytest.mark.parametrize("mutation,message", [
    ("duplicate", "duplicate identities"), ("missing_owner", "piece without iteration"),
    ("hole", "piece index outside owner"), ("negative", "invalid integer identity"),
    ("parameters_nan", "nonfinite"), ("range_inf", "nonfinite"),
    ("parameters_mask", "missing"), ("range_width", "shape"),
])
def test_pad_candidates_rejects_bad_candidates(mutation, message):
    pieces, iterations = piece_fixture()
    if mutation == "duplicate":
        pieces.append(deepcopy(pieces[0]))
    elif mutation == "missing_owner":
        pieces[0]["event"] = 999
    elif mutation in ("hole", "negative"):
        pieces[0]["piece_index"] = 2 if mutation == "hole" else -1
    elif mutation == "parameters_nan":
        pieces[0]["parameters"][0] = float("nan")
    elif mutation == "range_inf":
        pieces[0]["range"][0] = float("inf")
    elif mutation == "parameters_mask":
        pieces[0]["parameters"][0] = None
    else:
        pieces[0]["range"] = [0.]
    with pytest.raises(ValueError, match=message):
        D.pad_candidates(ak.Array(pieces), ak.Array(iterations))


def z_operands():
    return dict(parameters=np.array([[[0., 0., 4., 0., 2., 2000.]]]),
                ranges=np.array([[[-10., 10.]]]), counts=np.array([1]),
                parent=np.array([0]), zpos=np.array([100.]), hint=np.array([3.]),
                mass=np.array([105.6583755]))


def test_ztime_real_particle_source_arithmetic_and_no_input_mutation():
    args = z_operands()
    original = deepcopy(args)
    p, direction, speed = D.C.particle(args["parameters"][0], args["hint"], args["mass"])
    expected = args["hint"] + (args["zpos"] - p[:, 2]) / (direction[:, 2] * speed)
    result = D.ztime(**args)
    np.testing.assert_array_equal(result["time"], expected)
    np.testing.assert_array_equal(result["updates"], [1])
    np.testing.assert_array_equal(result["index"], [0])
    assert not result["exhausted"].any()
    for key in args:
        np.testing.assert_array_equal(args[key], original[key])


def test_ztime_oscillation_exact_100_update_cap_and_inactive_lanes(monkeypatch):
    # Each piece's q[0] is the root of an exact linear z(t), independent of hints.
    par = np.zeros((3, 2, 6))
    par[:, :, 5] = 2000.
    par[:, :, 0] = [[1.5, .5], [.25, 1.5], [1.5, 1.75]]
    calls = []
    def particle(q, time, mass):
        calls.append(len(time))
        position = np.zeros((len(time), 3))
        position[:, 2] = time - q[:, 0]
        direction = np.zeros_like(position)
        direction[:, 2] = 1.
        return position, direction, np.ones(len(time))
    monkeypatch.setattr(D.C, "particle", particle)
    result = D.ztime(par, np.tile([[0., 1.], [1., 2.]], (3, 1, 1)),
                     np.full(3, 2), np.arange(3), np.zeros(3), np.full(3, .5), np.ones(3))
    np.testing.assert_array_equal(result["time"], [.5, .25, 1.75])
    np.testing.assert_array_equal(result["index"], [0, 0, 1])
    np.testing.assert_array_equal(result["updates"], [100, 1, 2])
    np.testing.assert_array_equal(result["exhausted"], [True, False, False])
    assert calls == [3, 2] + [1] * 98


def test_ztime_zero_longitudinal_velocity_is_explicit_failure(monkeypatch):
    def particle(q, time, mass):
        return np.zeros((len(time), 3)), np.zeros((len(time), 3)), np.ones(len(time))
    monkeypatch.setattr(D.C, "particle", particle)
    with pytest.raises(ValueError, match="zero longitudinal velocity"):
        D.ztime(**z_operands())


@pytest.mark.parametrize("field", ["zpos", "hint", "mass"])
@pytest.mark.parametrize("bad", ["mask", "nan", "shape"])
def test_ztime_rejects_missing_nonfinite_and_shape(field, bad):
    args = z_operands()
    args[field] = (np.ma.array(args[field], mask=[True]) if bad == "mask" else
                   np.array([np.nan]) if bad == "nan" else np.array([1., 2.]))
    with pytest.raises(ValueError, match="missing|finite|shape"):
        D.ztime(**args)


@pytest.mark.parametrize("mass", [0., -1.])
def test_ztime_rejects_nonpositive_mass(mass):
    args = z_operands()
    args["mass"][:] = mass
    with pytest.raises(ValueError, match="positive"):
        D.ztime(**args)


@pytest.mark.parametrize("field", ["parameters", "ranges", "counts", "parent"])
def test_ztime_rejects_masked_piece_and_owner_operands(field):
    args = z_operands()
    mask = np.zeros(args[field].shape, dtype=bool)
    mask.flat[0] = True
    args[field] = np.ma.array(args[field], mask=mask)
    with pytest.raises(ValueError, match="missing|mask"):
        D.ztime(**args)


def hit_operands(n):
    args = z_operands()
    args.pop("hint")
    args.pop("zpos")
    args.update(parent=np.zeros(n, int), mass=np.full(n, 105.6583755),
                line=D.C.sensor(np.tile([100., 0., 0.], (n, 1)),
                                np.tile([-100., 0., 0.], (n, 1)),
                                np.full(n, 4.), np.full(n, 200.)),
                previous=dict(usable=np.ones(n, bool), particle_time=np.arange(n) + 10.,
                              sensor_time=np.arange(n) + 20.),
                drift=np.full(n, .25), straw_z=np.zeros(n))
    return args


def fake_state(dz, usable):
    dz = np.asarray(dz)
    poca = np.zeros((len(dz), 3))
    poca[:, 2] = -dz
    return dict(particle_poca=poca, usable=np.array(usable, bool),
                status=np.where(usable, 0, 2), index=np.arange(len(dz)))


def test_hit_update_strict_100_retry_fallback_and_second_giveup(monkeypatch):
    args = hit_operands(6)
    args["previous"]["usable"][-1] = False
    first = fake_state([100., -100., np.nextafter(100., np.inf),
                        np.nextafter(-100., -np.inf), 0., 0.],
                       [True, True, True, True, False, False])
    second = fake_state([100., np.nextafter(-100., -np.inf), 0., 0.],
                        [True, True, False, True])
    saved = deepcopy(first)
    calls, zcalls = [], []
    def piecewise(parameters, ranges, counts, parent, line, ph, sh, mass):
        calls.append((parent.copy(), deepcopy(line), ph.copy(), sh.copy(), mass.copy()))
        return first if len(calls) == 1 else second
    def ztime(parameters, ranges, counts, parent, zpos, hint, mass):
        zcalls.append((parent.copy(), zpos.copy(), hint.copy(), mass.copy()))
        return dict(time=hint + 2., index=np.zeros(len(hint), int),
                    updates=np.ones(len(hint), int), exhausted=np.zeros(len(hint), bool))
    monkeypatch.setattr(D.L, "piecewise", piecewise)
    monkeypatch.setattr(D, "ztime", ztime)
    initial, final, policy = D.hit_update(**args)
    assert len(calls) == 2 and len(zcalls) == 1
    np.testing.assert_array_equal(calls[0][2], [10., 11., 12., 13., 14., 3.25])
    np.testing.assert_array_equal(calls[0][3], [20., 21., 22., 23., 24., 3.5])
    np.testing.assert_array_equal(policy["retry"], [False, False, True, True, True, True])
    np.testing.assert_array_equal(policy["give_up"], [False, False, False, True, True, False])
    np.testing.assert_array_equal(zcalls[0][2], np.full(4, 3.25))
    np.testing.assert_array_equal(calls[1][2], np.full(4, 5.25))
    np.testing.assert_array_equal(calls[1][3], np.full(4, 3.5))
    assert calls[1][1]["mid"].shape == (4, 3)
    for key in first:
        np.testing.assert_array_equal(initial[key], saved[key])
        np.testing.assert_array_equal(first[key], saved[key])
        np.testing.assert_array_equal(final[key][:2], saved[key][:2])
        np.testing.assert_array_equal(final[key][2:], second[key])
        assert not np.shares_memory(final[key], first[key])


def test_hit_update_no_retry_does_not_call_ztime(monkeypatch):
    first = fake_state([100., -100.], [True, True])
    monkeypatch.setattr(D.L, "piecewise", lambda *a, **k: first)
    def forbidden(*a, **k):
        pytest.fail("boundary-only usable states must not retry")
    monkeypatch.setattr(D, "ztime", forbidden)
    _, final, policy = D.hit_update(**hit_operands(2))
    assert policy["ztime"] is None
    assert not policy["retry"].any() and not policy["give_up"].any()
    np.testing.assert_array_equal(final["particle_poca"], first["particle_poca"])


@pytest.fixture
def predecessor(tmp_path):
    evidence = tmp_path / "synthetic_input.txt"
    evidence.write_text("local synthetic evidence only\n")
    manifest = {str(evidence): D.digest(evidence)}
    data = dict(completed=True, hashes_unchanged=True, before_sha256=manifest,
                end_sha256=manifest.copy(), rtol=D.R.RTOL, atol=D.R.ATOL)
    path = tmp_path / "cad.json"
    def save():
        path.write_text(json.dumps(data))
        return D.digest(path)
    save()
    return path, evidence, data, save


def test_receipt_valid_provenance_and_pinned_hash_before_parsing(predecessor):
    path, evidence, data, save = predecessor
    pin = save()
    watched = {}
    assert D.receipt(path, watched, expected=pin) == data
    assert watched == {str(path.resolve()): pin, str(evidence): D.digest(evidence)}
    path.write_text("not JSON")
    watched = {}
    with pytest.raises(ValueError, match="predecessor receipt changed"):
        D.receipt(path, watched, expected=pin)
    assert watched == {str(path.resolve()): pin}


@pytest.mark.parametrize("field,value,message", [
    ("completed", False, "incomplete"), ("completed", 1, "incomplete"),
    ("hashes_unchanged", 1, "incomplete"),
    ("before_sha256", {}, "missing predecessor manifest"),
    ("before_sha256", [], "missing predecessor manifest"),
    ("end_sha256", {}, "incomplete"),
    ("rtol", 1e-4, "tolerances differ"), ("atol", 1e-6, "tolerances differ"),
])
def test_receipt_malformed_contract(predecessor, field, value, message):
    path, _, data, save = predecessor
    data[field] = value
    pin = save()
    with pytest.raises(ValueError, match=message):
        D.receipt(path, {}, expected=pin)


@pytest.mark.parametrize("sha", ["g" * 64, "A" * 64, "a" * 63, 123, None, []])
def test_receipt_malformed_digest_retains_declared_manifest(predecessor, sha):
    path, evidence, data, save = predecessor
    data["before_sha256"] = data["end_sha256"] = {str(evidence): sha}
    pin = save()
    watched = {}
    with pytest.raises(ValueError, match="invalid manifest digest"):
        D.receipt(path, watched, expected=pin)
    assert watched == {str(path): pin, str(evidence): sha}


@pytest.mark.parametrize("mutation", ["changed", "deleted"])
def test_receipt_changed_or_deleted_evidence(predecessor, mutation):
    path, evidence, data, save = predecessor
    pin = save()
    if mutation == "changed":
        evidence.write_text("changed")
    else:
        evidence.unlink()
    watched = {}
    with pytest.raises((ValueError, FileNotFoundError)):
        D.receipt(path, watched, expected=pin)
    assert watched == {str(path): pin, **data["before_sha256"]}


def test_receipt_must_not_replace_preexisting_provenance(predecessor):
    path, evidence, _, save = predecessor
    pin = save()
    watched = {str(evidence): "0" * 64}
    with pytest.raises(ValueError):
        D.receipt(path, watched, expected=pin)
    assert watched[str(evidence)] == "0" * 64


@pytest.fixture
def default_environment(monkeypatch):
    for key in list(D.os.environ):
        if key.startswith("F1_"):
            monkeypatch.delenv(key)


def run_args(tmp_path, path=None):
    return SimpleNamespace(cad_report=path or tmp_path / "absent.json",
                           native_source=tmp_path / "native", out=tmp_path / "out.json")


def assert_failure_json(args, report):
    assert report["completed"] is False
    assert json.loads(args.out.read_text()) == report
    for field in ("full_closure", "full_history_closure", "full_state_policy_closure",
                  "native_hint_calls_captured", "crossing_retained_vectors_captured"):
        assert report[field] is False
    assert report["limits"] == D.LIMITS
    assert (report["rtol"], report["atol"]) == (1e-5, 1e-7)


def test_run_nondefault_environment_exclusive_failure_json(tmp_path, monkeypatch, default_environment):
    args = run_args(tmp_path)
    monkeypatch.setenv("F1_SYNTHETIC_NONDEFAULT", "1")
    def forbidden(*a, **k):
        pytest.fail("must reject environment before reading predecessor")
    monkeypatch.setattr(D, "receipt", forbidden)
    report = D.run(args)
    assert_failure_json(args, report)
    assert "requires default F1 environment" in report["error"]
    assert report["before_sha256"] == report["end_sha256"] == {}
    assert report["hashes_unchanged"] is False
    saved = args.out.read_bytes()
    with pytest.raises(FileExistsError):
        D.run(args)
    assert args.out.read_bytes() == saved


def test_run_missing_predecessor_records_pin(tmp_path, default_environment):
    args = run_args(tmp_path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert report["before_sha256"] == {str(args.cad_report): D.CAD_SHA256}
    assert report["end_sha256"] == {str(args.cad_report): None}
    assert report["hashes_unchanged"] is False
    assert "FileNotFoundError" in report["error"]


@pytest.mark.parametrize("mutation", ["unchanged", "changed", "deleted"])
def test_run_failure_rehashes_all_known_provenance(tmp_path, monkeypatch, default_environment, mutation):
    args = run_args(tmp_path)
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original")
    original = D.digest(evidence)
    def fail_receipt(path, watched):
        watched[str(evidence)] = original
        if mutation == "changed":
            evidence.write_text("changed")
        elif mutation == "deleted":
            evidence.unlink()
        raise ValueError("synthetic predecessor failure")
    monkeypatch.setattr(D, "receipt", fail_receipt)
    report = D.run(args)
    assert_failure_json(args, report)
    assert "synthetic predecessor failure" in report["error"]
    assert report["before_sha256"] == {str(evidence): original}
    assert report["end_sha256"] == {str(evidence): None if mutation == "deleted" else D.digest(evidence)}
    assert report["hashes_unchanged"] is (mutation == "unchanged")
    assert ("integrity_error" in report) is (mutation != "unchanged")


@pytest.mark.parametrize("sha", [None, [], "bad", float("nan")])
def test_run_malformed_manifest_still_writes_strict_failure_json(
        predecessor, tmp_path, monkeypatch, default_environment, sha):
    path, evidence, data, save = predecessor
    data["before_sha256"] = data["end_sha256"] = {str(evidence): sha}
    pin = save()
    real_receipt = D.receipt
    monkeypatch.setattr(D, "receipt", lambda p, watched: real_receipt(p, watched, expected=pin))
    args = run_args(tmp_path, path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert "invalid manifest digest" in report["error"]
    assert report["hashes_unchanged"] is False


def test_hit_update_absent_previous_uses_midpoint_drift_hint(monkeypatch):
    args = hit_operands(2)
    args["previous"] = None
    calls = []
    def piecewise(parameters, ranges, counts, parent, line, ph, sh, mass):
        calls.append((ph.copy(), sh.copy()))
        return fake_state([0., 0.], [True, True])
    monkeypatch.setattr(D.L, "piecewise", piecewise)
    _, _, policy = D.hit_update(**args)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [3.25, 3.25])
    np.testing.assert_array_equal(calls[0][1], [3.5, 3.5])
    assert not policy["retry"].any()


@pytest.mark.parametrize("which", ["first", "final"])
def test_hit_update_nonfinite_dz_fails_without_dropping_lane(monkeypatch, which):
    calls = []
    def piecewise(*args, **kwargs):
        calls.append(1)
        return fake_state([np.nan if which == "first" or len(calls) == 2 else 101.], [True])
    monkeypatch.setattr(D.L, "piecewise", piecewise)
    monkeypatch.setattr(D, "ztime", lambda *a, **k: dict(time=np.array([0.])))
    with pytest.raises(ValueError, match="nonfinite " + which + " hit-z difference"):
        D.hit_update(**hit_operands(1))
    assert len(calls) == (1 if which == "first" else 2)


@pytest.mark.parametrize("text", ["not JSON", "[]", "null"])
def test_run_malformed_predecessor_json_is_failure_report(
        tmp_path, monkeypatch, default_environment, text):
    path = tmp_path / "cad.json"
    path.write_text(text)
    pin = D.digest(path)
    real_receipt = D.receipt
    monkeypatch.setattr(D, "receipt", lambda p, watched: real_receipt(p, watched, expected=pin))
    args = run_args(tmp_path, path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert report["before_sha256"] == report["end_sha256"] == {str(path): pin}
    assert report["hashes_unchanged"] is True
    assert "error" in report


def test_run_malformed_manifest_path_still_writes_failure_json(
        predecessor, tmp_path, monkeypatch, default_environment):
    path, _, data, save = predecessor
    data["before_sha256"] = data["end_sha256"] = {"invalid\x00path": "a" * 64}
    pin = save()
    real_receipt = D.receipt
    monkeypatch.setattr(D, "receipt", lambda p, watched: real_receipt(p, watched, expected=pin))
    args = run_args(tmp_path, path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert "error" in report
    assert report["hashes_unchanged"] is False
