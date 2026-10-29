"""Saved-operand diagnostics only: synthetic arrays/ROOT, never execute a fit."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import f1_momentum_first_operands as O
from test_f1_hit_operand_check import change, sample


def forbidden(*args, **kwargs):
    pytest.fail("invalid evidence reached downstream comparison")


def fixture():
    """One real native algebra fixture, scrambled source slots, three crossings."""
    rows, effects, iteration = sample()
    rows, effects, iteration = rows[:5], effects[:6], iteration[:1]
    effects = change(change(effects, "time", -40., 0), "time", 45., 1)
    effects = change(effects, "time", 30., 5)
    effects = ak.with_field(effects, np.full(len(effects), -1), "partner_shindex")
    extra = ak.to_list(effects[5:]) * 3
    # Fixed synthetic records, not a per-event analysis loop.
    extra = [dict(record, effect_index=index, kind=2, strawid=strawid,
                  partner_shindex=partner, active=active, in_fit=in_fit, time=time)
             for record, index, strawid, partner, active, in_fit, time in zip(
                 extra, (6, 7, 8), (502, 503, 501), (22, 24, 20),
                 (True, False, True), (True, True, False), (27., 7., 17.))]
    effects = ak.concatenate([effects, ak.Array(extra)])
    iteration = change(change(change(iteration, "neffects", 9), "range_begin", 12.), "range_end", 30.)
    identities = dict(hit_ids=np.array([24, 999, 20, 23, 21, 22]),
                      crossing_ids=np.array([503, 501, 502]),
                      seed_hit_mask=np.array([True, False, True, True, True, True]))
    active = np.array([True, False, True, False, False, True])
    mvar = np.array([[20., 40., 80.], [0., 0., 0.], [2., 4., 8.],
                     [12., 24., 48.], [6., 12., 24.], [8., 16., 32.],
                     [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 1., 1.]])
    weight = np.zeros((10, 3))
    weight[:6] = np.divide(.5, mvar[:6], out=np.zeros((6, 3)), where=active[:, None])
    material = np.array([False]*6 + [True, False, False, False])
    order = np.array([8, 3, 0, 7, 5, 1, 9, 2, 6, 4])
    t = dict(order=order, eff_w=weight[order], eff_mvar=mvar[order],
             is_material=material[order], hit_active=active,
             hit_lr=np.array([1., 0., -1., 0., 0., -1.]),
             physical_time=np.array([18., -999., -10., 50., 90., 22., 5., 15., 25., 30.]),
             bounds=np.array([-10., 30.]), crossing_active=np.array([True, True, False]),
             crossing_inbounds=np.array([True, False, False]))
    return t, identities, iteration, effects, rows, 2.


def diagnostic(args=None):
    return O.state_variance(*(fixture() if args is None else args))


@pytest.mark.parametrize("times,active,values,slots", [
    ([7., -5., 3., 20.], [True, False, True, False], [3., 7.], [2, 0]),
    ([2., 2., 2.], [False, True, True], [2., 2.], [1, 1]),
    ([-4.], [True], [-4., -4.], [0, 0]),
])
def test_bounds_values_and_source_slots(times, active, values, slots):
    assert O.bounds(times, np.array(active)) == dict(values=values, source_slots=slots)


@pytest.mark.parametrize("times,active", [
    ([], np.array([], dtype=bool)), ([1., 2.], [False, False]),
    ([1., np.nan], [True, False]), ([np.inf, 2.], [False, True]),
    ([1., 2.], [1, 0]), ([1., 2.], [True]),
    ([[1., 2.]], [[True, True]]), (1., True),
])
def test_bounds_rejects_invalid_operands_including_inactive_nonfinite(times, active):
    with pytest.raises(ValueError, match="bound operands"):
        O.bounds(times, active)


def test_real_native_algebra_full_population_and_row_activity_counts():
    report = diagnostic()
    check, state = report["native_row_algebra"], report["hit_states"]
    assert check["passed"] and check["n_hits"] == check["n_expected_hits"] == 5
    assert check["n_excluded_hits"] == 0 and check["n_active_residual_rows"] == 6
    assert state["source_shindex"] == [24, 20, 23, 21, 22]
    assert state["native_hit_state"] == [1, -3, 0, -2, -1]
    assert state["native_active"] == [True, False, True, False, True]
    assert state["f1_active"] == [True, True, False, False, True]
    assert state["activity_different_shindex"] == [20, 23]
    assert state["native_row_active"] == [[True, True, True], [False]*3,
                                          [True, True, False], [False]*3, [True, False, False]]
    assert state["row_mismatch_indices"] == [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [4, 1], [4, 2]]
    expected = {"distance": (2, 1, 1, 1, [24, 22]),
                "time": (1, 2, 1, 1, [24]), "longitudinal": (1, 2, 0, 2, [24])}
    for name, (common, f1only, nativeonly, neither, sources) in expected.items():
        metric = state["row_comparisons"][name]
        assert [metric[k] for k in ("n_common", "n_f1_only", "n_native_only", "n_neither")] == [common, f1only, nativeonly, neither]
        assert common + f1only + nativeonly + neither == 5
        assert metric["source_shindex"] == sources
        assert metric["mvar"]["n_components"] == metric["weight"]["n_components"] == common
    assert state["f1_mvar"]["values"] == [[20., 40., 80.], [2., 4., 8.], [12., 24., 48.], [6., 12., 24.], [8., 16., 32.]]
    assert state["native_weight"]["values"] == [[.25, .125, .0625], [0., 0., 0.], [.25, .125, 0.], [0., 0., 0.], [.25, 0., 0.]]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("permutation", [np.arange(10), np.arange(9, -1, -1),
                                           np.array([4, 9, 2, 0, 8, 5, 1, 6, 3, 7])])
def test_sorted_effect_payload_permutations_and_native_row_order(permutation):
    args = list(fixture())
    baseline = diagnostic(args)
    t = args[0]
    inverse = np.argsort(t["order"])
    for name in ("eff_w", "eff_mvar", "is_material"):
        t[name] = t[name][inverse][permutation]
    t["order"] = permutation
    args[3] = args[3][np.array([4, 2, 0, 5, 1, 3, 6, 7, 8])]
    args[4] = args[4][np.array([2, 0, 4, 1, 3])]
    report = diagnostic(args)
    for name in ("hit_states", "bounds", "crossings"):
        assert report[name] == baseline[name]
    # Native checker intentionally preserves incoming row-key order.
    assert report["native_row_algebra"]["passed"]
    assert sorted(report["native_row_algebra"]["hit_keys"]) == sorted(baseline["native_row_algebra"]["hit_keys"])


@pytest.mark.parametrize("field", ["run", "subrun", "event", "iseed", "iteration_ordinal", "effect_index"])
def test_full_key_row_join_rejects_wrong_component_even_with_validator_isolated(monkeypatch, field):
    args = list(fixture())
    args[4] = change(args[4], field, 99)
    monkeypatch.setattr(O.H, "validate", lambda *a: {"passed": True})
    monkeypatch.setattr(O.T, "_metric", forbidden)
    with pytest.raises(ValueError, match="identity"):
        diagnostic(args)


def test_full_key_join_not_shindex_only_with_foreign_subrun_rows(monkeypatch):
    args = list(fixture())
    expected = diagnostic(args)["hit_states"]
    foreign = ak.with_field(args[4], np.ones(5, dtype=int), "subrun")
    foreign = ak.with_field(foreign, np.full((5, 3), 1234.), "row_mvar")
    args[4] = ak.concatenate([foreign, args[4][::-1]])
    monkeypatch.setattr(O.H, "validate", lambda *a: {"passed": True})
    assert diagnostic(args)["hit_states"] == expected


@pytest.mark.parametrize("field", ["run", "subrun", "event", "iseed", "iteration_ordinal", "effect_index"])
def test_real_native_validator_rejects_wrong_full_key_before_metrics(monkeypatch, field):
    args = list(fixture())
    args[4] = change(args[4], field, 99)
    monkeypatch.setattr(O.T, "_metric", forbidden)
    with pytest.raises(ValueError, match="identity"):
        diagnostic(args)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_common_active_nonfinite_variance_kept_in_denominator(value):
    args = list(fixture())
    t = args[0]
    t["eff_mvar"][np.flatnonzero(t["order"] == 0)[0], 0] = value
    report = diagnostic(args)
    state = report["hit_states"]
    metric = state["row_comparisons"]["distance"]
    assert metric["n_common"] == metric["mvar"]["n_components"] == 2
    assert metric["source_shindex"] == [24, 22]
    assert not metric["mvar"]["passed"] and metric["mvar"]["nonfinite"] == 1
    assert metric["mvar"]["mismatches"] == 2
    assert state["f1_mvar"]["values"][0][0] is None
    assert state["f1_mvar"]["nonfinite"][0][0]
    json.dumps(report, allow_nan=False)


def test_empty_common_activity_is_explicit_not_dropped():
    args = list(fixture())
    args[0]["hit_active"][:] = False
    args[0]["eff_w"][:] = 0.
    args[0]["bounds"] = np.array([30., 30.])
    report = diagnostic(args)
    for name, native_count in (("distance", 3), ("time", 2), ("longitudinal", 1)):
        metric = report["hit_states"]["row_comparisons"][name]
        assert metric["n_common"] == metric["n_f1_only"] == 0
        assert metric["n_native_only"] == native_count and metric["n_neither"] == 5-native_count
        assert metric["source_shindex"] == [] and metric["mvar"]["n_components"] == 0


@pytest.mark.parametrize("scale", [1., 2. + 1e-12, 0., -1., np.nan, np.inf])
def test_native_f1_variance_scale_mismatch_fails_before_comparisons(monkeypatch, scale):
    args = list(fixture())
    args[-1] = scale
    monkeypatch.setattr(O.T, "_metric", forbidden)
    with pytest.raises(ValueError, match="variance scale differs"):
        diagnostic(args)


def test_crossed_mask_bounds_are_labelled_with_values_and_sources():
    b = diagnostic()["bounds"]
    assert b["source_shindex_or_constraint"] == [24, 20, 23, 21, 22, -1]
    assert b["f1_mask"] == [True, True, False, False, True, True]
    assert b["native_mask"] == [True, False, True, False, True, True]
    assert b["f1_times"] == [18., -10., 50., 90., 22., 30.]
    assert b["native_times"] == [12., -40., 12., 45., 12., 30.]
    assert b["f1_times_f1_mask"] == dict(values=[-10., 30.], source_slots=[1, 5])
    assert b["native_times_native_mask"] == dict(values=[12., 30.], source_slots=[0, 5])
    assert b["diagnostic_native_times_f1_mask"] == dict(values=[-40., 30.], source_slots=[1, 5])
    assert b["diagnostic_f1_times_native_mask"] == dict(values=[18., 50.], source_slots=[0, 2])
    assert b["saved_f1"] == [-10., 30.] and b["saved_native"] == [12., 30.]
    assert b["f1_reconstruction"]["passed"] and b["native_reconstruction"]["passed"]


def test_crossing_activity_bounds_and_time_join_use_straw_identity():
    x = diagnostic()["crossings"]
    assert x["strawid"] == [502, 503, 501] and x["partner_shindex"] == [22, 24, 20]
    assert x["f1_active"] == [False, True, True]
    assert x["native_active"] == [True, False, True]
    assert x["f1_inbounds"] == [False, True, False]
    assert x["native_in_fit"] == [True, True, False]
    assert x["f1_participates"] == [False, True, False]
    assert x["native_participates"] == [True, False, False]
    assert x["disagreement_strawids"] == [502, 503]
    assert x["f1_time"] == [25., 5., 15.] and x["native_time"] == [27., 7., 17.]
    assert "not a pure geometric-bounds flag" in x["note"]


def test_real_native_row_algebra_failure_precedes_all_comparisons(monkeypatch):
    args = list(fixture())
    args[3] = change(args[3], "weight_matrix", [123.]*36, 4)
    assert not O.H.validate(args[4], args[3], args[2])["passed"]
    monkeypatch.setattr(O.T, "_metric", forbidden)
    monkeypatch.setattr(O, "bounds", forbidden)
    with pytest.raises(ValueError, match="captured native row algebra failed"):
        diagnostic(args)


@pytest.mark.parametrize("mutation,message", [
    ("order", "effect order"), ("hit_ids", "ambiguous source identities"),
    ("crossing_ids", "ambiguous source identities"), ("weight_nan", "F1 weights"),
    ("weight_negative", "F1 weights"), ("activity", "row/hit activity"),
    ("material", "material participation"), ("f1_bounds", "bound reconstruction"),
    ("native_bounds", "bound reconstruction"), ("constraint", "constraint activity"),
])
def test_invalid_saved_operands_and_bounds_fail_closed(mutation, message):
    args = list(fixture())
    t = args[0]
    if mutation == "order":
        t["order"][0] = t["order"][1]
    elif mutation in ("hit_ids", "crossing_ids"):
        args[1][mutation][0] = args[1][mutation][1]
    elif mutation in ("weight_nan", "weight_negative"):
        t["eff_w"][np.flatnonzero(t["order"] == 0)[0], 0] = np.nan if mutation == "weight_nan" else -1.
    elif mutation == "activity":
        t["hit_active"][0] = False
    elif mutation == "material":
        t["is_material"][np.flatnonzero(t["order"] == 6)[0]] = False
    elif mutation == "f1_bounds":
        t["bounds"][0] = -11.
    elif mutation == "native_bounds":
        args[2] = change(args[2], "range_begin", 11.)
    else:
        args[3] = change(args[3], "active", False, 5)
    with pytest.raises(ValueError, match=message):
        diagnostic(args)


def evidence(tmp_path, monkeypatch):
    """Real receipt, NPZ, ROOT/highest-cycle reads; only replace sealed digests."""
    t, identities, iteration, effects, rows, _ = fixture()
    # Exactly representable native/F1 annealing scale (T=1 -> scale=4).
    rows = ak.with_field(rows, np.full(len(rows), 4.), "variance_scale")
    effects = ak.with_field(effects, effects.weight_matrix / 2., "weight_matrix")
    effects = ak.with_field(effects, effects.weight_vector / 2., "weight_vector")
    data = tmp_path / "data"
    trace_dir = data / "momentum_trace_7033_fixed_v1"
    trace_dir.mkdir(parents=True)
    capture = data / "crossing_3k_20261028_v2" / "capture.root"
    capture.parent.mkdir()
    # Decoys share event/seed/effect indices but differ in subrun or iteration.
    def decoys(table):
        return ak.concatenate([ak.with_field(table, np.ones(len(table), dtype=int), "subrun"),
                               ak.with_field(table, np.ones(len(table), dtype=int), "iteration_ordinal"), table])
    pieces = ak.Array([dict(run=8, subrun=0, event=42, iseed=3, iteration_ordinal=0, owner=0)])
    with uproot.recreate(capture) as root:
        root["kktrace/iterations"] = {f: change(iteration, "capture_valid", False)[f] for f in iteration.fields}
        for name, table in dict(iterations=decoys(iteration), effects=decoys(effects),
                                pieces=decoys(pieces), hit_operands=decoys(rows)).items():
            root["kktrace/" + name] = {f: table[f] for f in table.fields}
    saved = {name: np.zeros((1,) + shape, dtype=O.fittrace.dtype(name))
             for name, shape in O.fittrace.shapes(6, 3).items()}
    saved.update({name: value[None] for name, value in t.items()})
    saved.update(identities, key=np.array([8, 0, 42, 3]), valid=np.array([True]), stage_index=np.array([0]))
    trace = trace_dir / "trace.npz"
    np.savez(trace, **saved)
    helper = tmp_path / "helper-source.txt"
    helper.write_text("synthetic watched helper\n")
    hashes = {str(helper): O.A.digest(helper)}
    manifest = dict(completed=True, passed=True, controls_passed=True, hashes_unchanged=True,
                    before_sha256=hashes, end_sha256=hashes,
                    trace_passivity=dict(exact=True), population_compatibility=dict(passed=True),
                    artifacts=dict(trace=dict(path=str(trace), sha256=O.A.digest(trace))),
                    key=[8, 0, 42, 3], n_iterations=1, schedule=dict(T=[1.]))
    receipt = trace_dir / "manifest.json"
    receipt.write_text(json.dumps(manifest, allow_nan=False))
    monkeypatch.setattr(O, "TRACE_SHA", O.A.digest(receipt))
    monkeypatch.setattr(O.C, "CAPTURE_SHA", O.A.digest(capture))
    monkeypatch.setattr(O.M, "helpers", lambda watched: O.D.watch(helper, watched))
    return SimpleNamespace(data=data, out=tmp_path / "report.json", manifest=manifest,
                           receipt=receipt, trace=trace, saved=saved, helper=helper, capture=capture)


def replace_comparison(monkeypatch, replacement):
    """Isolate run failure/finalization, retaining all other real origin checks."""
    original = O.D.code_origin
    monkeypatch.setattr(O, "state_variance", replacement)
    def origin(function, path):
        if function is replacement:
            return
        return original(function, path)
    monkeypatch.setattr(O.D, "code_origin", origin)


def reseal(e, monkeypatch):
    e.receipt.write_text(json.dumps(e.manifest, allow_nan=False))
    monkeypatch.setattr(O, "TRACE_SHA", O.A.digest(e.receipt))


def strict_report(path):
    def reject(value):
        pytest.fail("non-strict JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=reject)


def test_run_real_saved_artifacts_highest_cycles_sources_hashes_and_exclusive_output(tmp_path, monkeypatch):
    e = evidence(tmp_path, monkeypatch)
    report = O.run(e.data, e.out)
    assert report["completed"] and report["passed"], report
    assert report["hashes_unchanged"] and report["before_sha256"] == report["end_sha256"]
    for path in (Path(O.__file__).resolve(), Path(O.H.__file__).resolve(), e.receipt, e.trace, e.capture, e.helper):
        assert report["before_sha256"][str(path)] == O.A.digest(path)
    assert report["highest_cycles"]["kktrace/iterations"].endswith(";2")
    assert report["diagnostic"]["native_row_algebra"]["n_hits"] == 5
    assert report["key"] == [8, 0, 42, 3]
    assert not any(report[k] for k in ("full_fit_closure", "full_f1_closure", "full_numerical_validation"))
    assert strict_report(e.out) == report
    before = e.out.read_bytes()
    with pytest.raises(FileExistsError):
        O.run(e.data, e.out)
    assert e.out.read_bytes() == before


@pytest.mark.parametrize("kind", ["receipt_hash", "artifact_hash", "capture_hash", "controls", "passivity", "compatibility"])
def test_run_invalid_evidence_fails_before_comparisons(tmp_path, monkeypatch, kind):
    e = evidence(tmp_path, monkeypatch)
    if kind == "receipt_hash":
        monkeypatch.setattr(O, "TRACE_SHA", "0"*64)
    elif kind == "capture_hash":
        monkeypatch.setattr(O.C, "CAPTURE_SHA", "0"*64)
    else:
        if kind == "artifact_hash":
            e.manifest["artifacts"]["trace"]["sha256"] = "0"*64
        elif kind == "controls":
            e.manifest["controls_passed"] = False
        elif kind == "passivity":
            e.manifest["trace_passivity"]["exact"] = False
        else:
            e.manifest["population_compatibility"]["passed"] = False
        reseal(e, monkeypatch)
    replace_comparison(monkeypatch, forbidden)
    report = O.run(e.data, e.out)
    assert not report["completed"] and not report["passed"]
    expected = "hash mismatch" if kind.endswith("hash") else "missing literal receipt flag" if kind == "controls" else "incompatible/nonpassive trace"
    assert expected in report["error"]
    assert report["hashes_unchanged"] and strict_report(e.out) == report


@pytest.mark.parametrize("field,value", [("key", np.array([8, 1, 42, 3])),
                                         ("valid", np.array([False])),
                                         ("stage_index", np.array([1]))])
def test_run_saved_identity_validity_and_stage_fail_closed(tmp_path, monkeypatch, field, value):
    e = evidence(tmp_path, monkeypatch)
    e.saved[field] = value
    np.savez(e.trace, **e.saved)
    e.manifest["artifacts"]["trace"]["sha256"] = O.A.digest(e.trace)
    reseal(e, monkeypatch)
    replace_comparison(monkeypatch, forbidden)
    report = O.run(e.data, e.out)
    assert not report["passed"] and "identity/validity" in report["error"]
    assert strict_report(e.out) == report


@pytest.mark.parametrize("remove", [False, True])
def test_run_rehash_on_failure_detects_watched_change_or_missing_file(tmp_path, monkeypatch, remove):
    e = evidence(tmp_path, monkeypatch)
    def fail(*args):
        if remove:
            e.helper.unlink()
        else:
            e.helper.write_text("changed\n")
        raise ValueError("deliberate comparison failure")
    replace_comparison(monkeypatch, fail)
    report = O.run(e.data, e.out)
    assert not report["passed"] and not report["hashes_unchanged"]
    assert "deliberate comparison failure" in report["error"]
    assert bool(report["hash_errors"]) is remove
    assert strict_report(e.out) == report


def test_run_source_code_origin_violation_is_reported(tmp_path, monkeypatch):
    e = evidence(tmp_path, monkeypatch)
    def foreign_bounds(*args):
        raise AssertionError("must not execute")
    foreign_bounds.__module__ = O.__name__
    monkeypatch.setattr(O, "bounds", foreign_bounds)
    report = O.run(e.data, e.out)
    assert not report["passed"] and "function code from different checkout" in report["error"]
    assert report["hashes_unchanged"] and strict_report(e.out) == report


def test_run_strict_json_failure_keeps_hashes_and_fails_closed(tmp_path, monkeypatch):
    e = evidence(tmp_path, monkeypatch)
    replace_comparison(monkeypatch, lambda *a: {"unserializable": np.nan})
    report = O.run(e.data, e.out)
    assert not report["passed"] and not report["completed"]
    assert "strict JSON serialization failed" in report["error"]
    assert report["hashes_unchanged"] and strict_report(e.out) == report


def test_run_rejects_protected_checkout_output_before_creation(tmp_path):
    out = SCRIPTS / "__forbidden_operand_report_for_test__.json"
    assert not out.exists()
    with pytest.raises(ValueError, match="protected checkout"):
        O.run(tmp_path, out)
    assert not out.exists()


@pytest.mark.parametrize("mode", ["missing_hit", "extra_hit", "duplicate_row", "missing_crossing_identity"])
def test_membership_and_identity_guards_with_isolated_validator(monkeypatch, mode):
    args = list(fixture())
    if mode == "missing_hit":
        args[3] = args[3][1:]
    elif mode == "extra_hit":
        args[3] = ak.concatenate([args[3], args[3][:1]])
    elif mode == "duplicate_row":
        args[4] = ak.concatenate([args[4], args[4][:1]])
    else:
        args[3] = change(args[3], "strawid", 99999, 6)
    monkeypatch.setattr(O.H, "validate", lambda *a: {"passed": True})
    with pytest.raises(ValueError, match="membership|duplicate|identity"):
        diagnostic(args)


def test_run_rehash_after_successful_comparison_still_fails_closed(tmp_path, monkeypatch):
    e = evidence(tmp_path, monkeypatch)
    original = O.state_variance
    def mutate(*args):
        result = original(*args)
        e.helper.write_text("changed after comparison\n")
        return result
    replace_comparison(monkeypatch, mutate)
    report = O.run(e.data, e.out)
    assert report["completed"] and not report["passed"] and not report["hashes_unchanged"]
    assert report["diagnostic"]["native_row_algebra"]["passed"]
    assert strict_report(e.out) == report


def test_run_iteration_count_mismatch_fails_before_comparison(tmp_path, monkeypatch):
    e = evidence(tmp_path, monkeypatch)
    e.manifest["n_iterations"] = 2
    reseal(e, monkeypatch)
    replace_comparison(monkeypatch, forbidden)
    report = O.run(e.data, e.out)
    assert not report["passed"] and "identity/validity" in report["error"]
    assert strict_report(e.out) == report


@pytest.mark.parametrize("suffix", [', "passed": true}', ', "unexpected": NaN}'])
def test_run_receipt_strict_json_rejects_duplicate_keys_and_nan(tmp_path, monkeypatch, suffix):
    e = evidence(tmp_path, monkeypatch)
    e.receipt.write_text(e.receipt.read_text()[:-1] + suffix)
    monkeypatch.setattr(O, "TRACE_SHA", O.A.digest(e.receipt))
    replace_comparison(monkeypatch, forbidden)
    report = O.run(e.data, e.out)
    assert not report["passed"] and not report["completed"]
    assert "duplicate" in report["error"] or "nonfinite" in report["error"]
    assert report["hashes_unchanged"] and strict_report(e.out) == report
