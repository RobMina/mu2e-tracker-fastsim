"""Local seed-audit contracts: provenance, failure receipts and independent toy capture.

No native build, remote data, receipt pin changes or production-helper substitutes.
The synthetic capture is computed analytically, not with the audited generators.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_seed_contract_audit as D


def args(tmp_path):
    return SimpleNamespace(**{k: tmp_path/k for k in
                              ("out", "full_check", "initial", "capture", "expanded", "native_source")})


def test_helpers_bound():
    watched = {}
    modules = D.helpers(watched)
    root = Path(D.__file__).resolve().parents[1]
    for name in ("scripts/f1_seed_contract.py", "scripts/f1_seed_contract_audit.py",
                 "scripts/f1_initial_crossing.py", "scripts/f1_initial_crossing_audit.py"):
        path = str(root/name)
        assert modules[name] == path
        assert watched[path] == D.A.digest(path)
    assert all(Path(path).is_absolute() for path in watched)


@pytest.mark.parametrize("module,field", [
    (D.S, name) for name in ("configuration", "inputs", "bounds", "dot", "unit", "generate", "constraint")
] + [(D, name) for name in ("flat", "compare_all", "compare_targets", "population", "run", "main")])
def test_foreign_callable_rejected(monkeypatch, module, field):
    monkeypatch.setattr(module, field, lambda *a, **kw: None)
    with pytest.raises(ValueError):
        D.helpers({})


@pytest.mark.parametrize("field", ["array", "column", "match_rows", "require"])
def test_foreign_alias_rejected(monkeypatch, field):
    monkeypatch.setattr(D.S, field, lambda *a, **kw: None)
    with pytest.raises(ValueError, match="alias"):
        D.helpers({})


def test_spoofed_function_metadata_rejected(monkeypatch):
    def replacement(*a, **kw):
        return None
    replacement.__module__ = D.S.__name__
    replacement.__name__ = "generate"
    monkeypatch.setattr(D.S, "generate", replacement)
    with pytest.raises(ValueError, match="code from different checkout"):
        D.helpers({})


def test_transitive_seed_binding_rejected(monkeypatch):
    monkeypatch.setattr(D.S, "B", SimpleNamespace())
    with pytest.raises(ValueError, match="binding"):
        D.helpers({})


def test_module_path_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(D.S, "__file__", str(tmp_path/"foreign.py"))
    with pytest.raises(ValueError, match="module path"):
        D.helpers({})


def test_failure_receipt_is_exclusive_and_complete(tmp_path):
    a = args(tmp_path)
    report = D.run(a)
    assert not report["passed"] and not report["completed"]
    assert "FileNotFoundError" in report["error"]
    assert report["hashes_unchanged"]
    assert report["before_sha256"] == report["end_sha256"]
    assert report["before_sha256"] and report["hash_errors"] == []
    assert report["rtol"] == 1e-5 and report["atol"] == 1e-7
    for flag in ("full_fit_closure", "full_f1_closure", "full_numerical_validation", "independent_seed_finder"):
        assert report[flag] is False
    before = a.out.read_bytes()
    assert json.loads(before) == report
    with pytest.raises(FileExistsError):
        D.run(a)
    assert a.out.read_bytes() == before


def test_existing_output_rejected_before_helpers(tmp_path, monkeypatch):
    a = args(tmp_path)
    a.out.write_bytes(b"existing evidence\x00")
    def unexpected(*a):
        pytest.fail("exclusive output must be checked before helper execution")
    monkeypatch.setattr(D, "helpers", unexpected)
    with pytest.raises(FileExistsError):
        D.run(a)
    assert a.out.read_bytes() == b"existing evidence\x00"


@pytest.mark.parametrize("key", ["full_check", "initial", "capture", "expanded", "native_source", "checkout"])
def test_protected_paths(tmp_path, key):
    a = args(tmp_path)
    if key == "checkout":
        a.out = Path(D.__file__).resolve().parents[1]/"seed-audit-must-not-be-written.json"
    elif key == "native_source":
        a.out = a.native_source/"report.json"
    else:
        a.out = getattr(a, key)
        a.out.write_text("protected input")
    before = a.out.read_bytes() if a.out.exists() else None
    with pytest.raises(ValueError, match="protected"):
        D.run(a)
    assert (a.out.read_bytes() if a.out.exists() else None) == before


@pytest.mark.parametrize("target", ["capture", "native_source", "checkout"])
def test_protected_paths_resolve_symlinks(tmp_path, target):
    a = args(tmp_path)
    link = tmp_path/"alias"
    if target == "capture":
        a.capture.write_text("capture")
        link.symlink_to(a.capture)
        a.out = link
    else:
        directory = a.native_source if target == "native_source" else Path(D.__file__).resolve().parents[1]
        if target == "native_source":
            directory.mkdir()
        link.symlink_to(directory, target_is_directory=True)
        a.out = link/"seed-audit-must-not-be-written.json"
    with pytest.raises(ValueError, match="protected"):
        D.run(a)
    if target == "capture":
        assert a.capture.read_text() == "capture"
    else:
        assert not a.out.exists()


@pytest.mark.parametrize("change", ["edit", "delete"])
def test_source_hash_drift_retained_in_failure_manifest(tmp_path, monkeypatch, change):
    a = args(tmp_path)
    source = tmp_path/"watched-source.py"
    source.write_text("before")
    before = D.A.digest(source)
    def helpers(watched):
        D.D.watch(source, watched)
        if change == "edit":
            source.write_text("after")
        else:
            source.unlink()
        raise ValueError("deliberate failure after watch")
    monkeypatch.setattr(D, "helpers", helpers)
    report = D.run(a)
    assert report["before_sha256"] == {str(source): before}
    assert report["passed"] is False and report["completed"] is False
    assert report["hashes_unchanged"] is False
    assert "deliberate failure" in report["error"]
    assert "changed" in report["integrity_error"]
    if change == "edit":
        assert report["end_sha256"] == {str(source): D.A.digest(source)}
        assert report["hash_errors"] == []
    else:
        assert report["end_sha256"] == {}
        assert len(report["hash_errors"]) == 1
        assert "FileNotFoundError" in report["hash_errors"][0]
    assert json.loads(a.out.read_text()) == report


def test_empty_watch_failure_does_not_claim_integrity(tmp_path, monkeypatch):
    def helpers(watched):
        raise ValueError("before first watch")
    monkeypatch.setattr(D, "helpers", helpers)
    a = args(tmp_path)
    report = D.run(a)
    assert report["before_sha256"] == report["end_sha256"] == {}
    assert report["passed"] is False and report["hashes_unchanged"] is False
    assert json.loads(a.out.read_text()) == report


# Literal configuration deliberately does not borrow the helper's constants.
CONFIG = {"momentum": 2000., "errors": [15., .05, 10., .02, 2., 1.],
          "constraints": [0., 0., 0., 0., 0., .0001]}
KEYS = ("run", "subrun", "event", "iseed")
MEMBERSHIP = ("source_shindices", "sh_shindex", "sx_partner_shindex",
              "sh_strawid", "sx_strawid", "sh_index", "sx_index", "sx_partner_initial_index")
COMPARISONS = {"parameters", "covariance", "bnom", "charge", *MEMBERSHIP,
               "range_from_captured_times", "constraint_parameters", "constraint_covariance",
               "constraint_mask", "constraint_time_from_captured_times", "constraint_index"}


def analytic_q(raw):
    """Closed form for zero slopes or (A1,B1)=(1,0), independent of S."""
    a0, b0, a1, b1, t0 = raw
    assert b1 == 0. and a1 in (0., 1.)
    if a1 == 0.:
        return [a0, -math.pi/2, b0, math.pi/2, t0, 2000.]
    # Native momentum starts as float32. Only this scalar rounding is shared
    # with the source specification; POCA/angles/time are analytic here.
    component = float(np.float32(np.float32(1)/np.sqrt(np.float32(2)))*np.float32(2000))
    momentum = math.sqrt(2*component*component)
    speed = momentum/math.sqrt(momentum*momentum + 105.658367**2)*299.792458
    return [a0/math.sqrt(2), -math.pi/4, b0, math.pi/2,
            t0-a0/math.sqrt(2)/speed, momentum]


@pytest.fixture
def synthetic():
    """Three full event identities, four seeds, unequal unsorted memberships.

    Two events deliberately share their event number, and two seeds share an
    event. Captured fit times differ from measured construction times.
    """
    event_rows = [
        dict(run=11, subrun=2, event=17, sd_a0=[12., 8.], sd_b0=[30., -4.],
             sd_a1=[0., 1.], sd_b1=[0., 0.], sd_t0=[100., 200.],
             sd_n=[2, 3], sd_ind=[2, 0, 3, 1, 4],
             s_time=[12., 24., 36., 48., 60.], s_sid=[60001, 60002, 60003, 60004, 60005]),
        dict(run=11, subrun=3, event=17, sd_a0=[-7.], sd_b0=[19.],
             sd_a1=[0.], sd_b1=[0.], sd_t0=[300.], sd_n=[2], sd_ind=[1, 0],
             s_time=[70., 80.], s_sid=[10, 11]),
        dict(run=11, subrun=2, event=18, sd_a0=[5.], sd_b0=[-8.],
             sd_a1=[0.], sd_b1=[0.], sd_t0=[400.], sd_n=[1], sd_ind=[0],
             s_time=[9.], s_sid=[1]),
    ]
    # Seed capture order deliberately differs from event/source order.
    identities = [(11, 2, 17, 1), (11, 3, 17, 0), (11, 2, 18, 0), (11, 2, 17, 0)]
    raw = [[8., -4., 1., 0., 200.], [-7., 19., 0., 0., 300.],
           [5., -8., 0., 0., 400.], [12., 30., 0., 0., 100.]]
    indices = [[3, 1, 4], [1, 0], [0], [2, 0]]
    strawids = [[60004, 60002, 60005], [11, 10], [1], [60003, 60001]]
    hit_times = [[102., 98., 110.], [201., 205.], [301.], [402., 401.]]
    crossing_times = [[103., 97., 111.], [200., 206.], [302.], [399., 403.]]
    fit_ranges = [[97., 111.], [200., 206.], [301., 302.], [399., 403.]]
    construction = [[24., 60.], [70., 80.], [9., 9.], [12., 36.]]
    q = [analytic_q(row) for row in raw]
    covariance = np.diag(np.square([15., .05, 10., .02, 2., 1.])).reshape(36).tolist()
    ph_covariance = np.diag([1., 1., 1., 1., 1., .0001**2]).reshape(36).tolist()
    rows = []
    for i, key in enumerate(identities):
        local = list(range(len(indices[i])))
        rows.append(dict(zip(KEYS, key), schema_version=1, valid=True, mass=105.658367,
                         parameters=q[i], covariance=covariance.copy(), bnom=[0., 0., .001], charge=-1,
                         source_shindices=indices[i].copy(), sh_shindex=indices[i].copy(),
                         sx_partner_shindex=indices[i].copy(), sh_strawid=strawids[i].copy(),
                         sx_strawid=strawids[i].copy(), sh_index=local.copy(), sx_index=local.copy(),
                         sx_partner_initial_index=local.copy(), sh_time=hit_times[i], sx_time=crossing_times[i],
                         range_begin=fit_ranges[i][0], range_end=fit_ranges[i][1],
                         ph_parameters=q[i].copy(), ph_covariance=ph_covariance.copy(),
                         ph_mask=[0, 0, 0, 0, 0, 1], ph_time=[sum(fit_ranges[i])/2], ph_index=[0]))
    events = ak.Array(event_rows)
    events = ak.with_field(events, ak.values_astype(events.s_time, np.float32), "s_time")
    return SimpleNamespace(events=events, seeds=ak.Array(rows), config=deepcopy(CONFIG),
                           keys=np.array(identities), raw=np.array(raw), q=np.array(q),
                           construction=np.array(construction), fit_ranges=np.array(fit_ranges),
                           hit_keys=np.array([(*key, index) for key, members in zip(identities, indices)
                                              for index in members]),
                           strawids=np.array(sum(strawids, [])), parent=np.array([0, 0, 0, 1, 1, 2, 3, 3]))


def changed(table, field, value, row=0, component=None):
    """Own a new field buffer; never mutate the baseline capture/oracle."""
    values = deepcopy(ak.to_list(table[field]))
    if component is None:
        values[row] = value
    else:
        values[row][component] = value
    return ak.with_field(table, ak.Array(values), field)


def compare(data, seeds=None, events=None):
    return D.compare_all(data.events if events is None else events,
                         data.seeds if seeds is None else seeds, data.config)


def test_compare_all_independent_multiple_seeds(synthetic):
    s = synthetic
    before_events, before_seeds = ak.to_list(s.events), ak.to_list(s.seeds)
    source, generated, comparisons, diagnostic = compare(s)
    assert set(comparisons) == COMPARISONS
    assert all(result["passed"] for result in comparisons.values())
    np.testing.assert_array_equal(source["raw"], s.raw)
    np.testing.assert_array_equal(source["hit_keys"], s.hit_keys)
    np.testing.assert_array_equal(source["parent"], s.parent)
    np.testing.assert_array_equal(source["strawid"], s.strawids)
    np.testing.assert_array_equal(source["construction_range"], s.construction)
    np.testing.assert_allclose(generated["parameters"], s.q, rtol=1e-14, atol=1e-13)
    assert not np.array_equal(s.construction, s.fit_ranges)
    for name, result in comparisons.items():
        assert result["n_rows"] == (8 if name in MEMBERSHIP else 4)
        assert result["failed_keys"] == []
    assert diagnostic["n_seeds"] == 4 and diagnostic["n_initial_hits"] == 8
    assert diagnostic["n_parallel"] == 0 and diagnostic["n_non_nominal_momentum"] == 1
    momentum = s.q[:, 5]
    np.testing.assert_allclose(diagnostic["momentum_minus_2000_quantiles"],
                               np.quantile(momentum-2000., [0, .1, .5, .9, 1]), rtol=0, atol=1e-12)
    assert diagnostic["generated_momentum_exact"] == np.array_equal(generated["parameters"][:, 5], momentum)
    assert ak.to_list(s.events) == before_events and ak.to_list(s.seeds) == before_seeds
    assert s.config == CONFIG


@pytest.mark.parametrize("event_order,seed_order", [([2, 0, 1], [0, 1, 2, 3]),
                                                   ([0, 1, 2], [3, 1, 0, 2]),
                                                   ([1, 2, 0], [2, 3, 1, 0])])
def test_compare_all_full_key_shuffles(synthetic, event_order, seed_order):
    s = synthetic
    source, generated, comparisons, diagnostic = compare(s, seeds=s.seeds[seed_order], events=s.events[event_order])
    assert all(item["passed"] for item in comparisons.values())
    np.testing.assert_array_equal(source["raw"], s.raw[seed_order])
    np.testing.assert_array_equal(source["construction_range"], s.construction[seed_order])
    np.testing.assert_allclose(generated["parameters"], s.q[seed_order], rtol=1e-14, atol=1e-13)
    expected_keys = np.concatenate([s.hit_keys[s.parent == i] for i in seed_order])
    np.testing.assert_array_equal(source["hit_keys"], expected_keys)
    assert diagnostic["n_seeds"] == 4 and diagnostic["n_initial_hits"] == 8


# Each captured output has a single independent corruption and one expected
# comparison failure. In particular native q/C are never generator operands.
@pytest.mark.parametrize("field,component,value,failed", [
    ("parameters", 0, 900., "parameters"),
    ("covariance", 1, 9., "covariance"),
    ("bnom", 2, .1, "bnom"),
    ("charge", None, 1, "charge"),
    *((name, 0, 123, name) for name in MEMBERSHIP),
    ("range_begin", None, -500., "range_from_captured_times"),
    ("range_end", None, 500., "range_from_captured_times"),
    ("ph_parameters", 4, 900., "constraint_parameters"),
    ("ph_covariance", 0, 7., "constraint_covariance"),
    ("ph_mask", 0, 1, "constraint_mask"),
    ("ph_time", 0, 999., "constraint_time_from_captured_times"),
    ("ph_index", 0, 1, "constraint_index"),
])
def test_compare_all_captured_field_mutation_is_isolated(synthetic, field, component, value, failed):
    s = synthetic
    baseline = ak.to_list(s.seeds)
    _, original, original_comps, _ = compare(s)
    seeds = changed(s.seeds, field, value, component=component)
    source, generated, comparisons, _ = compare(s, seeds=seeds)
    assert {name for name, result in comparisons.items() if not result["passed"]} == {failed}
    result = comparisons[failed]
    assert result["n_failed_components"] == 1 and result["n_failed_rows"] == 1
    assert result["failed_keys"] == [s.hit_keys[0].tolist() if field in MEMBERSHIP else s.keys[0].tolist()]
    for name, result in original_comps.items():
        if name != failed:
            assert comparisons[name] == result
    for name in original:
        np.testing.assert_array_equal(generated[name], original[name])
    np.testing.assert_array_equal(source["raw"], s.raw)
    assert ak.to_list(s.seeds) == baseline
    assert all(item["passed"] for item in compare(s)[2].values())


@pytest.mark.parametrize("field", MEMBERSHIP)
def test_compare_all_membership_order_matters(synthetic, field):
    s = synthetic
    members = ak.to_list(s.seeds[field][0])
    members[0], members[1] = members[1], members[0]
    comparisons = compare(s, seeds=changed(s.seeds, field, members))[2]
    assert {name for name, result in comparisons.items() if not result["passed"]} == {field}
    assert comparisons[field]["n_failed_components"] == 2
    assert comparisons[field]["failed_keys"] == s.hit_keys[:2].tolist()


@pytest.mark.parametrize("field", [*MEMBERSHIP, "sh_time", "sx_time"])
def test_compare_all_jagged_collection_lengths_rejected(synthetic, field):
    values = ak.to_list(synthetic.seeds[field][0])[:-1]
    with pytest.raises(ValueError, match="collection lengths differ"):
        compare(synthetic, seeds=changed(synthetic.seeds, field, values))


@pytest.mark.parametrize("field,component,value", [("schema_version", None, 2), ("valid", None, False),
                                                  ("mass", None, 100.), ("parameters", 2, float("nan")),
                                                  ("covariance", 0, float("inf")), ("ph_time", 0, float("nan")),
                                                  ("sh_time", 0, float("nan")), ("sx_time", 0, float("inf"))])
def test_compare_all_invalid_capture_rejected(synthetic, field, component, value):
    with pytest.raises(ValueError):
        compare(synthetic, seeds=changed(synthetic.seeds, field, value, component=component))


@pytest.mark.parametrize("corruption", ["duplicate_seed", "missing_event", "duplicate_event", "missing_seed"])
def test_compare_all_full_identity_errors(synthetic, corruption):
    s = synthetic
    seeds, events = s.seeds, s.events
    if corruption == "duplicate_seed":
        seeds = seeds[[0, 0, 2, 3]]
    elif corruption == "missing_event":
        events = events[[0, 2]]
    elif corruption == "duplicate_event":
        events = events[[0, 0, 1, 2]]
    else:
        seeds = changed(seeds, "iseed", 20)
    with pytest.raises(ValueError):
        compare(s, seeds=seeds, events=events)


@pytest.mark.parametrize("field,which", [("sh_time", -100.), ("sx_time", 999.)])
def test_captured_times_only_drive_conditional_range_and_constraint_time(synthetic, field, which):
    s = synthetic
    source0, generated0, _, _ = compare(s)
    source, generated, comparisons, _ = compare(s, seeds=changed(s.seeds, field, which, component=0))
    assert {name for name, item in comparisons.items() if not item["passed"]} == {
        "range_from_captured_times", "constraint_time_from_captured_times"}
    for name in source0:
        np.testing.assert_array_equal(source[name], source0[name])
    for name in generated0:
        np.testing.assert_array_equal(generated[name], generated0[name])


def test_measured_times_affect_construction_range_not_all_seed_comparisons(synthetic):
    s = synthetic
    before, generated0, comps0, _ = compare(s)
    # Preserve required float32 precision after changing one selected ComboHit.
    events = changed(s.events, "s_time", -900., component=3)
    events = ak.with_field(events, ak.values_astype(events.s_time, np.float32), "s_time")
    source, generated, comps, _ = compare(s, events=events)
    assert source["construction_range"][0].tolist() == [-900., 60.]
    np.testing.assert_array_equal(source["construction_range"][1:], before["construction_range"][1:])
    assert comps == comps0
    for name in generated0:
        np.testing.assert_array_equal(generated[name], generated0[name])


def test_numeric_pass_distinct_from_exact(synthetic):
    s = synthetic
    tiny = float(s.seeds.bnom[0][0])+1e-9
    comparisons = compare(s, seeds=changed(s.seeds, "bnom", tiny, component=0))[2]
    assert all(item["passed"] for item in comparisons.values())
    assert comparisons["bnom"]["exact"] is False
    assert comparisons["bnom"]["inexact_keys"] == [s.keys[0].tolist()]
