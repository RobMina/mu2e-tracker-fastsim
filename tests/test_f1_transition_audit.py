"""Synthetic transition audit contracts; no captures, native jobs or remote IO."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_transition_audit as D


SEED = [[5, "CADSHU"], [5, "CADSHU"], [2, "CADSHU"], [1, "CADSHU"]]
EXTENSION = [[2, "CADSHU"], [1, "CADSHU"], [.3, "CADSHU"], [0, "CADSHU:BkgANNSHU"],
             [1, "CADSHU:DriftANNSHU"], [.5, "CADSHU:DriftANNSHU"],
             [.2, "CADSHU:DriftANNSHU"], [0, "CADSHU:DriftANNSHU:BkgANNSHU"],
             [0, "CADSHU:DriftANNSHU"]]


def expanded_blocks():
    blocks = {}
    for name, schedule in (("FitSettings", SEED), ("ExtensionSettings", EXTENSION)):
        blocks[name] = ("  ProcessEnds: false\n  BFieldCorrection: false\n"
                        "  MetaIterationSettings: " + json.dumps(schedule) + "\n")
    blocks["KKFitSettings"] = ('  StrawHitClusterLevel: "none"\n  UseCaloCluster: false\n'
        '  AddHits: true\n  AddMaterial: true\n  MaterialCorrection: true\n'
        '  MaxStrawHitDt: 50\n  MaxStrawHitDOCA: 10\n  TPOCAPrecision: 0.0001\n')
    return blocks


def expanded_text(blocks):
    return "\n".join(name + ": {\n" + body + "}\n" for name, body in blocks.items())


def configure(tmp_path, text):
    path = tmp_path / "expanded.fcl"
    path.write_text(text)
    return D.configuration(path)


def test_configuration_identical_expanded_scopes_defaults_and_no_mutation(tmp_path):
    text = expanded_text(expanded_blocks())
    # Expanded copies at distinct nesting/indentation levels are expected.
    nested = "outer: {\n" + "\n".join("  " + line for line in text.splitlines()) + "\n}\n"
    result = configure(tmp_path, text + text)
    assert result["FitSettings"]["schedule"] == SEED
    assert result["ExtensionSettings"]["schedule"] == EXTENSION
    assert result["seed_stages"] == 4 and result["extension_stages"] == 9
    assert result["max_piece_moves"] == 10 and result["cluster_level"] == "none"
    assert result["TPOCAPrecision"] == float(np.float32(1e-4))
    for name in ("FitSettings", "ExtensionSettings"):
        assert result[name]["process_ends"] is False
        assert result[name]["bfield_correction"] is False
    assert configure(tmp_path, nested) == result
    assert "separate CA" in result["lifecycle"]


@pytest.mark.parametrize("name", ["FitSettings", "ExtensionSettings", "KKFitSettings"])
@pytest.mark.parametrize("mutation", ["missing", "empty", "conflicting"])
def test_configuration_rejects_missing_empty_or_conflicting_blocks(tmp_path, name, mutation):
    blocks = expanded_blocks()
    original = expanded_text(blocks)
    if mutation == "missing":
        del blocks[name]
    else:
        blocks[name] = "" if mutation == "empty" else blocks[name] + "  other: true\n"
    text = expanded_text(blocks)
    if mutation == "conflicting":
        text = original + text
    with pytest.raises(ValueError):
        configure(tmp_path, text)


@pytest.mark.parametrize("scope", ["FitSettings", "ExtensionSettings"])
@pytest.mark.parametrize("field", ["ProcessEnds", "BFieldCorrection"])
@pytest.mark.parametrize("mutation", ["true", "duplicate", "other_scope"])
def test_configuration_false_is_required_once_in_its_own_scope(tmp_path, scope, field, mutation):
    blocks = expanded_blocks()
    line = "  " + field + ": false\n"
    if mutation == "true":
        blocks[scope] = blocks[scope].replace(line, line.replace("false", "true"))
    elif mutation == "duplicate":
        blocks[scope] += line
    else:
        blocks[scope] = blocks[scope].replace(line, "")
        blocks["KKFitSettings"] += line
    with pytest.raises(ValueError, match="unsupported " + scope + "." + field):
        configure(tmp_path, expanded_text(blocks))


@pytest.mark.parametrize("scope", ["FitSettings", "ExtensionSettings"])
@pytest.mark.parametrize("extra", ["HitReferencePiece: 0", "XingReferencePiece: 0", "MaxPieceIter: 10",
                                   "Chi2SHUSettings: {}", "PanelDiagSHUSettings: {}"])
def test_configuration_rejects_explicit_reference_strategy_and_other_updaters(tmp_path, scope, extra):
    blocks = expanded_blocks()
    blocks[scope] += "  " + extra + "\n"
    with pytest.raises(ValueError, match="reference strategy|unsupported updater"):
        configure(tmp_path, expanded_text(blocks))


@pytest.mark.parametrize("scope", ["FitSettings", "ExtensionSettings"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "empty", "changed", "wrong_scope"])
def test_configuration_exact_scoped_schedule(tmp_path, scope, mutation):
    blocks = expanded_blocks()
    schedule = SEED if scope == "FitSettings" else EXTENSION
    line = "  MetaIterationSettings: " + json.dumps(schedule) + "\n"
    if mutation == "duplicate":
        blocks[scope] += line
    elif mutation in ("missing", "wrong_scope"):
        blocks[scope] = blocks[scope].replace(line, "")
        if mutation == "wrong_scope":
            blocks["KKFitSettings"] += line
    else:
        replacement = [] if mutation == "empty" else schedule[::-1]
        blocks[scope] = blocks[scope].replace(line, "  MetaIterationSettings: " + json.dumps(replacement) + "\n")
    with pytest.raises(ValueError, match="schedule"):
        configure(tmp_path, expanded_text(blocks))


@pytest.mark.parametrize("old,new", [('StrawHitClusterLevel: "none"', 'StrawHitClusterLevel: "panel"'),
    ('UseCaloCluster: false', 'UseCaloCluster: true'), ('AddHits: true', 'AddHits: false'),
    ('AddMaterial: true', 'AddMaterial: false'), ('MaterialCorrection: true', 'MaterialCorrection: false'),
    ('MaxStrawHitDt: 50', 'MaxStrawHitDt: 51'), ('MaxStrawHitDOCA: 10', 'MaxStrawHitDOCA: 11'),
    ('TPOCAPrecision: 0.0001', 'TPOCAPrecision: 0.001')])
def test_configuration_kk_contract(tmp_path, old, new):
    with pytest.raises(ValueError):
        configure(tmp_path, expanded_text(expanded_blocks()).replace(old, new))


def link_fixture():
    targets = [(1, 2, 3, 4), (9, 2, 3, 4), (1, 9, 3, 4), (1, 2, 9, 4), (1, 2, 3, 9)]
    stages = [(0, 0), (0, 1), (1, 0), (3, 2), (4, 0), (5, 0)]
    effects, rows, iterations = [], [], []
    for target in targets:
        for ordinal, (miter, iteration) in enumerate(stages):
            base = dict(zip(D.KEYS, target), iteration_ordinal=ordinal, miter=miter, iter=iteration)
            iterations.append(dict(base, swapped=True))
            for source in (7, 8):
                row = dict(base, identity_kind=1, effect_index=source + 20 * ordinal,
                           input_index=source, shindex=100 + source, strawid=200 + source,
                           has_reference=True, marker=len(rows))
                effects.append(row)
                rows.append(dict(row))
            effects.append(dict(base, identity_kind=2, effect_index=99, input_index=99,
                                shindex=99, strawid=99, has_reference=False, marker=-1))
    return effects, rows, iterations


def call_links(effects, rows, iterations):
    return D.hit_links(ak.Array(effects), ak.Array(rows), ak.Array(iterations))


def test_hit_links_full_identity_all_ordinal_transitions_not_only_same_miter():
    effects, rows, iterations = link_fixture()
    effects, iterations = effects[::-1], iterations[::2] + iterations[1::2]
    hits, sidecar, parent, before, after = call_links(effects, rows[::-1], iterations)
    assert len(hits) == 60 and len(before) == len(after) == 50
    assert ak.to_list(hits.marker) == ak.to_list(sidecar.marker)
    assert len(hits) - len(before) == 10
    assert np.count_nonzero(D.column(hits, "miter")[before] != D.column(hits, "miter")[after]) == 40
    for b, a in zip(before, after):
        old, new, owner = ak.to_list(hits[b]), ak.to_list(hits[a]), iterations[parent[b]]
        assert all(old[k] == new[k] == owner[k] for k in D.KEYS)
        assert new["iteration_ordinal"] == old["iteration_ordinal"] + 1
        assert new["input_index"] == old["input_index"]
        assert new["effect_index"] != old["effect_index"]


@pytest.mark.parametrize("mutation,message", [
    ("duplicate_effect", "duplicate identities"), ("duplicate_sidecar", "duplicate identities"),
    ("duplicate_iteration", "duplicate identities"), ("duplicate_source", "duplicate identities"),
    ("missing_sidecar", "census differs"), ("missing_iteration", "missing full identity join"),
    ("wrong_key", "missing full identity join"), ("no_reference", "without reference"),
    ("shindex", "adjacent hit identity changed"), ("strawid", "adjacent hit identity changed"),
    ("iteration_miter", "effect/iteration identity differs"), ("iteration_iter", "effect/iteration identity differs")])
def test_hit_links_rejects_incomplete_or_changed_identities(mutation, message):
    effects, rows, iterations = link_fixture()
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
    elif mutation == "wrong_key":
        rows[0]["run"] = 999
    elif mutation == "no_reference":
        effects[0]["has_reference"] = False
    elif mutation.startswith("iteration_"):
        iterations[0][mutation.removeprefix("iteration_")] += 1
    else:
        effects[3][mutation] += 1
        rows[2][mutation] += 1
    with pytest.raises(ValueError, match=message):
        call_links(effects, rows, iterations)


@pytest.mark.parametrize("field", ["input_index", "shindex", "strawid", "miter", "iter"])
def test_hit_links_sidecar_identity_fields(field):
    effects, rows, iterations = link_fixture()
    rows[0][field] += 1
    with pytest.raises(ValueError, match="sidecar identity differs: " + field):
        call_links(effects, rows, iterations)


def pointer_fixture():
    par = np.arange(24, dtype=float).reshape(2, 2, 6)
    ranges = np.array([[[0., 1.], [1., 2.]], [[3., 4.], [4., 5.]]])
    parent = np.array([1, 0, 1])
    chosen = np.array([1, 0, 0])
    hits = [dict(reference_range=ranges[p, i].tolist(), reference_parameters=par[p, i].tolist())
            for p, i in zip(parent, chosen)]
    return hits, par, ranges, np.array([2, 2]), parent


def test_observed_pointer_positive_width_boundary_and_owner_identity():
    hits, *args = pointer_fixture()
    np.testing.assert_array_equal(D.observed_pointer(ak.Array(hits), *args), [1, 0, 0])


@pytest.mark.parametrize("mutation,message", [("zero", "positive-width"), ("reverse", "positive-width"),
    ("range", "range outside"), ("parameters", "parameters outside"), ("nan", "nonfinite"),
    ("mask", "missing"), ("shape", "shape")])
def test_observed_pointer_rejects_bad_or_nonowned_reference(mutation, message):
    hits, *args = pointer_fixture()
    if mutation in ("zero", "reverse"):
        hits[0]["reference_range"] = [4., 4. if mutation == "zero" else 3.]
    elif mutation == "range":
        hits[0]["reference_range"] = [8., 9.]
    elif mutation == "parameters":
        hits[0]["reference_parameters"][0] += 1.
    elif mutation in ("nan", "mask"):
        hits[0]["reference_parameters"][0] = np.nan if mutation == "nan" else None
    else:
        hits[0]["reference_range"] = [4.]
    with pytest.raises(ValueError, match=message):
        D.observed_pointer(ak.Array(hits), *args)


def state_fixture():
    hits = [dict(time=3.)]
    rows = [dict(ca_usable=True, ca_status=0, precision=D.NATIVE_CA_PRECISION,
                 particle_poca=[1., 2., 3., 3.], sensor_poca=[2., 3., 4., 4.],
                 particle_direction=[0., 0., 1.], sensor_direction=[1., 0., 0.])]
    return hits, rows


def test_observed_state_all_supported_statuses_keep_complete_state():
    for status in (0, 1, 2):
        hits, rows = state_fixture()
        rows[0]["ca_status"] = status
        result = D.observed_state(ak.Array(hits), ak.Array(rows))
        assert set(result) == set(D.P.STATE_FIELDS)
        np.testing.assert_array_equal(result["delta"], [[1., 1., 1.]])
        np.testing.assert_array_equal(result["particle_time"], [3.])


@pytest.mark.parametrize("field", ["particle_poca", "sensor_poca", "particle_direction", "sensor_direction"])
@pytest.mark.parametrize("value", [np.nan, np.inf, None])
def test_observed_state_nonfinite_and_masked_vectors_fail_without_exclusion(field, value):
    hits, rows = state_fixture()
    rows[0][field][0] = value
    with pytest.raises(ValueError, match="nonfinite|missing"):
        D.observed_state(ak.Array(hits), ak.Array(rows))


@pytest.mark.parametrize("field,value,message", [("ca_usable", False, "unusable"),
    ("ca_usable", 2, "expected bool"), ("precision", 1e-4, "precision"),
    ("ca_status", -1, "status"), ("ca_status", 3, "status"), ("ca_status", 0., "dtype")])
def test_observed_state_usable_precision_and_status_are_gates(field, value, message):
    hits, rows = state_fixture()
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        D.observed_state(ak.Array(hits), ak.Array(rows))


def test_observed_state_effect_time_is_exact():
    hits, rows = state_fixture()
    hits[0]["time"] = np.nextafter(3., np.inf)
    with pytest.raises(ValueError, match="CA/effect time differs"):
        D.observed_state(ak.Array(hits), ak.Array(rows))


@pytest.fixture
def real_population(monkeypatch):
    """One retained hit plus two new sources; only the measured-input join is stubbed.

    Real source-grouped zTime, constructor, reference update, candidate padding,
    pointer lookup, sidecar alignment, census and comparisons all run unchanged.
    Captured states here are synthetic oracle outputs, not native-fit evidence.
    """
    q = np.array([0., 0., 4., .25, 2., 2000.])
    par, ranges, counts = q.reshape(1, 1, 6), np.array([[[-10., 10.]]]), np.array([1])
    mass = np.full(3, 105.6583755)
    early = np.array([[100., 1., 0.], [100., 2., 20.], [-100., 3., -10.]])
    late = early.copy()
    late[:, 0] *= -1
    measured = dict(early=early, late=late, measurement_time=np.array([4., 5., 6.]),
                    corrected_time=np.array([2.1, 2.2, 2.3]), dtime=np.array([.25, .3, .35]),
                    wdist=np.array([3., 7., -11.]))
    geom = dict(direction=np.tile([1., 0., 0.], (3, 1)), hlen=np.full(3, 100.))
    line = D.A.sensor(early, late, measured["measurement_time"], np.array([180., 200., 220.]))
    origin_z = np.array([0., 20., -10.])
    measured_z = np.array([0., 25., -30.], dtype=np.float32)

    def measured_join(hits, events, geometry, speed_config):
        index = D.column(hits, "shindex")
        return ({k: v[index] for k, v in measured.items()}, {k: v[index] for k, v in geom.items()},
                {k: v[index] for k, v in line.items()}, origin_z[index])
    monkeypatch.setattr(D, "measured_for_hits", measured_join)
    target = dict(run=1, subrun=2, event=3, iseed=4)
    stages = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    iterations, pieces, effects, rows = [], [], [], []
    zt = D.H.ztime(par, ranges, counts, np.zeros(3, int), measured_z.astype(float),
                  measured["corrected_time"], mass)
    sensor_hint = line["measurement_time"] - (geom["hlen"] -
        D.A._dot3(line["direction"], geom["direction"])*measured["wdist"]) / line["speed"]
    built = D.H.L.piecewise(par, ranges, counts, np.zeros(3, int), line, zt["time"], sensor_hint, mass)
    _, constructed, policy = D.H.hit_update(par, ranges, counts, np.zeros(3, int), line, built,
                                          measured["dtime"], origin_z, mass)
    assert constructed["usable"].all() and not policy["retry"].any()
    current = deepcopy(constructed)
    for ordinal, (miter, iteration) in enumerate(stages):
        base = dict(target, iteration_ordinal=ordinal, miter=miter, iter=iteration)
        iterations.append(dict(base, swapped=True, nold=1, ncandidate=1, candidate_present=True))
        for owner in (0, 1):
            pieces.append(dict(base, owner=owner, piece_index=0, parameters=q.tolist(),
                               range=[-10., 10.], covariance=np.eye(6).ravel().tolist(),
                               bnom=[0., 0., 1.], mass=mass[0], charge=-1.))
        if ordinal:
            _, current, _ = D.H.hit_update(par, ranges, counts, np.zeros(3, int), line, current,
                                          measured["dtime"], origin_z, mass)
        if miter == 4:
            for field in current:
                current[field][1:] = constructed[field][1:]
        for source in ([0] if miter < 4 else [0, 1, 2]):
            hit = dict(base, identity_kind=1, effect_index=10 * ordinal + source,
                       input_index=7 + source, shindex=source, strawid=100 + source,
                       has_reference=True, reference_range=[-10., 10.], reference_parameters=q.tolist(),
                       time=float(current["particle_time"][source]))
            effects.append(hit)
            rows.append(dict(hit, ca_usable=bool(current["usable"][source]),
                ca_status=int(current["status"][source]), precision=D.NATIVE_CA_PRECISION,
                particle_poca=[*current["particle_poca"][source], current["particle_time"][source]],
                sensor_poca=[*current["sensor_poca"][source], current["sensor_time"][source]],
                particle_direction=current["direction"][source].tolist(),
                sensor_direction=current["sensor_direction"][source].tolist(), hit_state=miter % 2))
    tables = dict(effects=ak.Array(effects[::-1]), iterations=ak.Array(iterations),
                  pieces=ak.Array(pieces[::-1]), seed_contract=ak.Array([dict(target, mass=mass[0],
                      sh_index=[7], sh_shindex=[0], sh_strawid=[100])]))
    events = ak.Array([dict(run=1, subrun=2, event=3, s_sid=[100, 101, 102], s_z=measured_z.tolist())])
    events = ak.with_field(events, ak.values_astype(events.s_z, np.float32), "s_z")
    links = D.T.iteration_links(tables["iterations"])
    continuity = D.T.owner_continuity(tables["pieces"], tables["iterations"], links)
    args = dict(tables=tables, rows=ak.Array(rows[::-1]), events=events, geometry={}, speed_config={},
                links=links, continuity=continuity)
    return SimpleNamespace(args=args, config=dict(seed_stages=4, miter_origin=0, MaxStrawHitDt=50., MaxStrawHitDOCA=10.),
                           ztime=zt, sensor_hint=sensor_hint, built=built, constructed=constructed,
                           measured=measured, measured_z=measured_z, q=q, mass=mass)


def test_new_extension_hits_two_sources_real_construction_and_reference(real_population):
    f = real_population
    result = D.new_extension_hits(**f.args, config=f.config)
    assert result["n_hits"] == 2 and result["n_targets"] == 1 and result["n_excluded"] == 0
    assert result["membership_supplied"] is True
    assert result["candidate_discovery_replayed"] is False
    assert result["native_constructor_intermediates_compared"] is False
    assert result["final"]["agreement"] == dict(criterion=True, exact=True)
    source = np.array(result["source_keys"])[:, -1]
    assert sorted(source.tolist()) == [1, 2]
    np.testing.assert_array_equal(result["ztime"], f.ztime["time"][source])
    np.testing.assert_array_equal(result["sensor_hint"], f.sensor_hint[source])
    for field in D.P.STATE_FIELDS:
        np.testing.assert_array_equal(result["constructor_state"][field], f.built[field][source])
        np.testing.assert_array_equal(result["final_state"][field], f.constructed[field][source])
    # Independent explicit line z-time arithmetic: measured ComboHit z, not wire-origin z.
    q = np.tile(f.q, (2, 1))
    hint = f.measured["corrected_time"][source]
    position, direction, speed = D.H.C.particle(q, hint, f.mass[source])
    expected = hint + (f.measured_z[source] - position[:, 2]) / (direction[:, 2] * speed)
    np.testing.assert_array_equal(result["ztime"], expected)
    for name in ("n_constructor_moved", "n_constructor_exhausted", "n_constructor_ztime_exhausted",
                 "n_reference_moved", "n_retry", "n_give_up", "n_reference_exhausted",
                 "n_measured_kinematic_cuts_failed"):
        assert result[name] == 0
    json.dumps(result, allow_nan=False)


def test_replay_retained_real_comparison_and_transition_census(real_population):
    result = D.replay_transitions(**real_population.args)
    expected = {"seed_meta": 3, "extension_entry": 1, "extension_meta": 3}
    for label, n in expected.items():
        group = result[label]
        assert group["n_hits"] == n and group["n_targets"] == 1 and group["n_excluded"] == 0
        assert group["final"]["agreement"] == dict(criterion=True, exact=True)
        assert group["hit_state_decisions_replayed"] is False
        assert group["n_hit_state_changed"] == n
        assert group["n_retry"] == group["n_give_up"] == group["n_exhausted"] == 0
        assert len(group["previous_keys"]) == len(group["following_keys"]) == n
    assert result["census"] == dict(n_all_snapshots=11, n_existing_hit_adjacent=8, n_same_meta=1,
                                    n_new_boundary_comparisons=7, n_snapshots_without_next_same_hit=3)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("which", ["retained", "new"])
def test_replays_reject_owner_discontinuity_without_excluding_hits(real_population, which):
    f = real_population
    f.args["continuity"] = {**f.args["continuity"], "equality": np.zeros(len(f.args["links"]["previous"]), bool)}
    with pytest.raises(ValueError, match="owner continuity failed"):
        if which == "new":
            D.new_extension_hits(**f.args, config=f.config)
        else:
            D.replay_transitions(**f.args)


@pytest.mark.parametrize("which", ["retained", "new"])
def test_replays_numerical_disagreement_stays_in_denominator(real_population, which):
    f = real_population
    rows = ak.to_list(f.args["rows"])
    # Shift every observed sensor position, leaving particle/effect time and all identities intact.
    for row in rows:
        row["sensor_poca"][1] += .1
    f.args["rows"] = ak.Array(rows)
    result = (D.new_extension_hits(**f.args, config=f.config) if which == "new"
              else D.replay_transitions(**f.args)["extension_meta"])
    assert result["n_hits"] == (2 if which == "new" else 3)
    assert result["n_excluded"] == 0
    assert result["final"]["agreement"] == dict(criterion=False, exact=False)


@pytest.mark.parametrize("mutation,message", [("double", "not float32"), ("nan", "nonfinite measured"),
    ("length", "position lengths differ"), ("late", "outside extension entry"),
    ("not_first", "first extension iteration")])
def test_new_extension_hits_rejects_bad_measured_z_and_entry_contract(real_population, mutation, message):
    f = real_population
    if mutation == "double":
        events = f.args["events"]
        f.args["events"] = ak.with_field(events, ak.values_astype(events.s_z, np.float64), "s_z")
    elif mutation in ("nan", "length"):
        events = ak.to_list(f.args["events"])
        if mutation == "nan":
            events[0]["s_z"][1] = np.nan
        else:
            events[0]["s_z"].pop()
        events = ak.Array(events)
        f.args["events"] = ak.with_field(events, ak.values_astype(events.s_z, np.float32), "s_z")
    else:
        effects = ak.to_list(f.args["tables"]["effects"])
        for hit in effects:
            if hit["input_index"] > 7 and hit["miter"] == 4:
                hit["miter" if mutation == "late" else "iter"] += 1
        f.args["tables"]["effects"] = ak.Array(effects)
    with pytest.raises(ValueError, match=message):
        D.new_extension_hits(**f.args, config=f.config)


@pytest.fixture
def default_environment(monkeypatch):
    for key in list(D.os.environ):
        if key.startswith("F1_"):
            monkeypatch.delenv(key)


def run_args(tmp_path, path=None):
    return SimpleNamespace(later_report=path or tmp_path / "absent.json",
                           native_source=tmp_path / "native", out=tmp_path / "out.json")


def assert_failure_json(args, report):
    assert report["completed"] is False
    assert json.loads(args.out.read_text()) == report
    for field in ("full_history_closure", "full_state_policy_closure", "full_fit_closure",
                  "native_crossing_constructor_offsets_compared"):
        assert report[field] is False
    assert report["limits"] == D.LIMITS
    assert (report["rtol"], report["atol"]) == (1e-5, 1e-7)
    json.dumps(report, allow_nan=False)


def test_run_nondefault_environment_exclusive_failure_json(tmp_path, monkeypatch, default_environment):
    args = run_args(tmp_path)
    monkeypatch.setenv("F1_SYNTHETIC_NONDEFAULT", "1")
    def forbidden(*a, **k):
        pytest.fail("must reject environment before reading predecessor")
    monkeypatch.setattr(D.H, "receipt", forbidden)
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
    assert report["before_sha256"] == {str(args.later_report): D.LATER_SHA256}
    assert report["end_sha256"] == {str(args.later_report): None}
    assert report["hashes_unchanged"] is False
    assert "FileNotFoundError" in report["error"]


@pytest.mark.parametrize("mutation", ["unchanged", "changed", "deleted"])
def test_run_failure_rehashes_all_known_provenance(tmp_path, monkeypatch, default_environment, mutation):
    args = run_args(tmp_path)
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original")
    original = D.digest(evidence)
    def fail_receipt(path, watched, expected):
        assert path == args.later_report and expected == D.LATER_SHA256
        watched[str(evidence)] = original
        if mutation == "changed":
            evidence.write_text("changed")
        elif mutation == "deleted":
            evidence.unlink()
        raise ValueError("synthetic predecessor failure")
    monkeypatch.setattr(D.H, "receipt", fail_receipt)
    report = D.run(args)
    assert_failure_json(args, report)
    assert "synthetic predecessor failure" in report["error"]
    assert report["before_sha256"] == {str(evidence): original}
    assert report["end_sha256"] == {str(evidence): None if mutation == "deleted" else D.digest(evidence)}
    assert report["hashes_unchanged"] is (mutation == "unchanged")
    assert ("integrity_error" in report) is (mutation != "unchanged")


@pytest.mark.parametrize("bad", [None, [], "bad", float("nan")])
def test_run_malformed_manifest_still_emits_strict_failure_json(tmp_path, monkeypatch, default_environment, bad):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original")
    manifest = {str(evidence): bad}
    data = dict(completed=True, hashes_unchanged=True, before_sha256=manifest,
                end_sha256=manifest.copy(), rtol=D.R.RTOL, atol=D.R.ATOL)
    path = tmp_path / "later.json"
    path.write_text(json.dumps(data))
    pin, receipt = D.digest(path), D.H.receipt
    monkeypatch.setattr(D.H, "receipt", lambda p, watched, expected: receipt(p, watched, expected=pin))
    args = run_args(tmp_path, path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert "invalid manifest digest" in report["error"]
    assert report["hashes_unchanged"] is False


def test_observed_pointer_uses_range_not_parameters_or_ca_time_to_disambiguate():
    # Equal particle parameters in adjacent pieces do not identify a pointer.
    q = np.array([0., 0., 4., .25, 2., 2000.])
    par = np.tile(q, (1, 3, 1))
    ranges = np.array([[[0., 1.], [1., 1.], [1., 2.]]])
    hits = ak.Array([dict(reference_range=[1., 2.], reference_parameters=q.tolist(), time=.5),
                     dict(reference_range=[0., 1.], reference_parameters=q.tolist(), time=1.5)])
    np.testing.assert_array_equal(D.observed_pointer(hits, par, ranges, np.array([3]), np.array([0, 0])), [2, 0])


def test_compact_candidates_preserves_owner_remap_and_excludes_absent_candidates():
    iterations = ak.Array([dict(run=1, subrun=2, event=e, iseed=4, iteration_ordinal=0)
                           for e in (3, 4, 5)])
    pieces = []
    for owner, index in ((2, 1), (0, 0), (2, 0)):
        base = ak.to_list(iterations[owner])
        pieces.append(dict(base, owner=1, piece_index=index,
                           parameters=[0., 0., 4., .25, float(index), 2000.],
                           range=[float(index), float(index + 1)]))
    # Old-reference rows are irrelevant, even when there is no candidate owner.
    pieces.append(dict(ak.to_list(iterations[1]), owner=0, piece_index=0,
                       parameters=[float("nan")] * 6, range=[0., 1.]))
    par, ranges, counts, remap = D.compact_candidates(dict(iterations=iterations, pieces=ak.Array(pieces)))
    np.testing.assert_array_equal(counts, [1, 2])
    np.testing.assert_array_equal(remap, [0, -1, 1])
    np.testing.assert_array_equal(par[1, 1], pieces[0]["parameters"])
    np.testing.assert_array_equal(ranges[1], [[0., 1.], [1., 2.]])


@pytest.mark.parametrize("field", ["ca_usable", "ca_status", "precision"])
def test_observed_state_missing_scalar_fails(field):
    hits, rows = state_fixture()
    hits *= 2
    rows.append(deepcopy(rows[0]))
    rows[0][field] = None
    with pytest.raises(ValueError, match="missing"):
        D.observed_state(ak.Array(hits), ak.Array(rows))


def test_replay_retained_unswapped_attempt_is_failure_not_selection(real_population):
    f = real_population
    iterations = ak.to_list(f.args["tables"]["iterations"])
    iterations[4]["swapped"] = False
    f.args["tables"]["iterations"] = ak.Array(iterations)
    with pytest.raises(ValueError, match="unswapped attempt"):
        D.replay_transitions(**f.args)


def test_replay_retained_empty_declared_group_is_failure(real_population):
    f = real_population
    links = deepcopy(f.args["links"])
    links["labels"][links["labels"] == "seed_meta"] = "same_meta"
    f.args["links"] = links
    with pytest.raises(ValueError, match="empty declared transition group: seed_meta"):
        D.replay_transitions(**f.args)


def test_new_extension_hits_empty_population_is_failure(real_population):
    f = real_population
    effects = f.args["tables"]["effects"]
    f.args["tables"]["effects"] = effects[effects.input_index == 7]
    with pytest.raises(ValueError, match="empty extension-hit population"):
        D.new_extension_hits(**f.args, config=f.config)


def test_new_extension_measured_cuts_are_diagnostic_not_exclusions(real_population):
    f = real_population
    config = dict(f.config, MaxStrawHitDOCA=.001)
    result = D.new_extension_hits(**f.args, config=config)
    assert result["n_measured_kinematic_cuts_failed"] == 2
    assert result["n_hits"] == 2 and result["n_excluded"] == 0
    assert result["final"]["agreement"] == dict(criterion=True, exact=True)
    assert result["membership_supplied"] is True and result["candidate_discovery_replayed"] is False
    assert "conditional" in result["kinematic_cut_scope"]


def test_new_extension_unusable_constructor_aborts_without_exclusion(real_population, monkeypatch):
    f = real_population
    piecewise = D.H.L.piecewise
    def unusable(*args, **kwargs):
        result = piecewise(*args, **kwargs)
        result["usable"][0] = False
        return result
    monkeypatch.setattr(D.H.L, "piecewise", unusable)
    with pytest.raises(ValueError, match="unusable extension constructor unsupported; no exclusion"):
        D.new_extension_hits(**f.args, config=f.config)


def test_run_predecessor_pin_precedes_json_parsing(tmp_path, default_environment):
    path = tmp_path / "later.json"
    path.write_text("not JSON")
    args = run_args(tmp_path, path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert "predecessor receipt changed" in report["error"]
    assert report["before_sha256"] == {str(path): D.LATER_SHA256}
    assert report["end_sha256"] == {str(path): D.digest(path)}
    assert report["hashes_unchanged"] is False


@pytest.mark.parametrize("text", ["not JSON", "[]", "null"])
def test_run_malformed_pinned_predecessor_preserves_provenance(tmp_path, monkeypatch, default_environment, text):
    path = tmp_path / "later.json"
    path.write_text(text)
    pin, receipt = D.digest(path), D.H.receipt
    monkeypatch.setattr(D.H, "receipt", lambda p, watched, expected: receipt(p, watched, expected=pin))
    args = run_args(tmp_path, path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert report["before_sha256"] == report["end_sha256"] == {str(path): pin}
    assert report["hashes_unchanged"] is True
    assert "error" in report


def test_jsonable_recursively_converts_array_scalars_without_changing_values():
    source = dict(a=np.array([[1, 2]], dtype=np.int64), b=(np.bool_(True), np.float32(.5)),
                  c=[dict(n=np.int64(3))])
    result = D.jsonable(source)
    assert result == dict(a=[[1, 2]], b=[True, .5], c=[dict(n=3)])
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert isinstance(source["a"], np.ndarray)
