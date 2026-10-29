"""F1 boundary regressions: synthetic inputs only; no capture/native job IO."""
from copy import deepcopy
from contextlib import nullcontext
import json

import awkward as ak
import numpy as np
import pytest

import test_f1_transition_census as C
from test_f1_transition_audit import (
    D, SEED, EXTENSION, expanded_blocks, expanded_text, configure,
    real_population, default_environment, run_args, assert_failure_json, state_fixture,
)

T = C.T


def native_lifetime():
    it, effects, seeds = C.lifetime_fixture()
    for row in it + effects:
        row["miter"] += 1
    for row in it:
        row["history_index"] = row["iteration_ordinal"] + 1
    return it, effects, seeds


def native_census(it, effects, seeds):
    return T.object_lifetimes(ak.Array(effects), ak.Array(it), ak.Array(seeds), miter_origin=1)


def test_native_origin_stage_counts_and_constructor_history(tmp_path):
    config = configure(tmp_path, expanded_text(expanded_blocks()))
    assert (config["miter_origin"], config["seed_stages"], config["extension_stages"]) == (1, 4, 9)
    it = C.iterations(tuple((m, 0) for m in range(1, 14)))
    for row in it:
        row["history_index"] = row["iteration_ordinal"] + 1
    links = T.iteration_links(ak.Array(it[::-1]), 4, 9, 1)
    by_miter = {it[::-1][p]["miter"]: label for p, label in zip(links["previous"], links["labels"])}
    assert by_miter == {**{m: "seed_meta" for m in (1, 2, 3)}, 4: "extension_entry",
                        **{m: "extension_meta" for m in range(5, 13)}}
    assert links["census"]["miter_origin"] == 1
    assert links["census"]["by_label"] == dict(same_meta=0, seed_meta=3, extension_entry=1, extension_meta=8)
    assert links["census"]["history_index_semantics_checked"] is False
    with pytest.raises(ValueError, match="configured stages|first configured"):
        T.iteration_links(ak.Array(it))  # no inference of origin from data
    it.append(dict(it[-1], iteration_ordinal=13, history_index=14, miter=14))
    with pytest.raises(ValueError, match="configured stages"):
        T.iteration_links(ak.Array(it), miter_origin=1)


@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize("mutation", ["miter", "iter"])
def test_initial_iteration_must_be_origin_and_iter_zero(origin, mutation):
    it = C.iterations(((origin, 0),))
    it[0][mutation] += 1
    with pytest.raises(ValueError, match="initial iteration outside first configured stage"):
        T.iteration_links(ak.Array(it), miter_origin=origin)


@pytest.mark.parametrize("origin", [True, -1, 1., None])
def test_origin_is_explicit_nonnegative_integer(origin):
    with pytest.raises(ValueError, match="stage counts"):
        T.iteration_links(ak.Array(C.iterations()), miter_origin=origin)


def test_native_extension_counter_uses_origin_plus_stage_count():
    it, effects, seeds = native_lifetime()
    result = native_census(it, effects, seeds)
    for family, expected in (("hits", 1), ("crossings", 2)):
        obj, census = result[family], result["census"][family]
        assert obj["first_miter"][obj["extension"]].tolist() == [5] * expected
        assert census["n_first_extension_new"] == expected
        assert census["n_extension"] == expected


def test_seed_catalog_is_scoped_to_147_captured_of_2319_targets():
    iterations, effects, seeds = [], [], []
    for event in range(2319):
        target = dict(C.TARGET, event=event)
        seeds.append(C.seed(target))
        if event < 147:
            it = C.iterations(((1, 0), (1, 1)), target=target)
            iterations += it
            effects += [C.effect(row, code=code) for row in it for code in (1, 5)]
    result = native_census(iterations[::-1], effects[::-1], seeds[::-1])["census"]
    assert result["n_all_seed_targets"] == 2319
    assert result["n_captured_seed_targets"] == 147
    assert result["n_seed_targets_outside_iteration_capture"] == 2172
    for family in ("hits", "crossings"):
        assert result[family]["n_initial"] == 147
        assert result[family]["n_seed_objects_not_observed"] == 0


@pytest.mark.parametrize("family,prefix", [("hits", "sh"), ("crossings", "sx")])
def test_absent_initial_objects_inside_capture_are_not_out_of_scope(family, prefix):
    it, effects, seeds = native_lifetime()
    seeds.append(C.seed(dict(C.TARGET, event=999)))
    for field in tuple(seeds[0]):
        if field.startswith(prefix + "_"):
            seeds[0][field].append(9 if field.endswith("_index") else (-1 if field.endswith("partner_shindex") else 999))
    census = native_census(it, effects, seeds)["census"]
    assert census["n_seed_targets_outside_iteration_capture"] == 1
    assert census[family]["n_seed_objects_not_observed"] == 1
    assert census[family]["n_initial_first_seen_after_zero"] == 0
    assert census[family]["no_disappearance_or_reappearance"]


@pytest.mark.parametrize("family,prefix", [("hits", "sh"), ("crossings", "sx")])
def test_parent_gate_rejects_truly_missing_initial_objects(tmp_path, monkeypatch, default_environment, family, prefix):
    # Exercise the real gate with a real census; adapt only receipt/capture IO and
    # fixed production-size checks, rather than reproducing the gate predicate.
    it, effects, seeds = native_lifetime()
    for field in tuple(seeds[0]):
        if field.startswith(prefix + "_"):
            seeds[0][field].append(9 if field.endswith("_index") else (-1 if field.endswith("partner_shindex") else 999))
    lifetime = native_census(it, effects, seeds)
    expanded = tmp_path / "expanded.fcl"
    expanded.write_text(expanded_text(expanded_blocks()))
    gate = dict(inputs={name: dict(path=name) for name in ("capture", "baseline")},
                highest_cycles=dict(capture={}, baseline={}))
    prior = dict(speed_configuration=dict(path=str(expanded)), frozen_gate=gate, operand_gate=gate)
    chain = prior
    for name, key in (("prior", "ca_receipt"), ("residual", "residual_receipt"), ("cad", "cad_receipt")):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(chain))
        chain = {key: dict(path=str(path))}
    monkeypatch.setattr(D.H, "receipt", lambda *a, **kw: chain)
    monkeypatch.setattr(D, "NATIVE_PATHS", ())
    monkeypatch.setattr(D.R, "executing_sources", lambda watched: {})
    monkeypatch.setattr(D.R, "FROZEN_EXPANDED_SHA256", D.digest(expanded))
    monkeypatch.setattr(D.uproot, "open", lambda *a, **kw: nullcontext(None))
    padded_it = ak.Array([dict(it[0], iteration_ordinal=i, history_index=i+1) for i in range(3449)])
    tables = dict(iterations=padded_it, hit_operands=ak.Array([dict(dummy=0)] * 61622),
                  effects=ak.Array(effects), seed_contract=ak.Array(seeds), pieces=ak.Array(C.pieces(it)),
                  events=ak.Array([dict(dummy=0)]), conditions=ak.Array([dict(dummy=0)]))
    monkeypatch.setattr(D, "read_tree", lambda f, scope, name, *a: tables[name])
    monkeypatch.setattr(T, "iteration_links", lambda *a, **kw: {})
    monkeypatch.setattr(T, "owner_continuity", lambda *a, **kw: {})
    monkeypatch.setattr(T, "object_lifetimes", lambda *a, **kw: lifetime)
    def forbidden(*a, **kw):
        pytest.fail("incomplete initial population must fail before numerical replay")
    monkeypatch.setattr(D, "replay_transitions", forbidden)
    args = run_args(tmp_path)
    report = D.run(args)
    assert_failure_json(args, report)
    assert report["error"] == "ValueError: unsupported incomplete object history: " + family
    assert report["hashes_unchanged"] is True
    assert report["history_contract"]["construction_record_index"] == 0
    assert report["history_contract"]["extension_first_miter"] == 5


def test_late_initial_hit_is_not_misclassified_as_new_extension(real_population):
    f = real_population
    effects = f.args["tables"]["effects"]
    f.args["tables"]["effects"] = effects[(effects.input_index != 7) | (effects.miter >= 4)]
    with pytest.raises(ValueError, match="initial seed hit first observed after construction"):
        D.new_extension_hits(**f.args, config=f.config)


def test_extension_seed_membership_is_exact_not_index_cutoff(real_population):
    f = real_population
    seed = ak.to_list(f.args["tables"]["seed_contract"])
    seed[0]["sh_index"] = [999]
    f.args["tables"]["seed_contract"] = ak.Array(seed)
    rows = ak.to_list(f.args["tables"]["effects"])
    for row in rows:
        if row["input_index"] == 7:
            row["input_index"] = 999
    f.args["tables"]["effects"] = ak.Array(rows)
    result = D.new_extension_hits(**f.args, config=f.config)
    assert result["n_hits"] == 2 and result["n_excluded"] == 0
    assert result["final"]["agreement"]["exact"]


def test_native_origin_extension_construction_uses_miter_five(real_population):
    f = real_population
    for name in ("iterations", "effects", "pieces"):
        table = f.args["tables"][name]
        f.args["tables"][name] = ak.with_field(table, table.miter + 1, "miter")
    f.args["rows"] = ak.with_field(f.args["rows"], f.args["rows"].miter + 1, "miter")
    f.config["miter_origin"] = 1
    f.args["links"] = T.iteration_links(f.args["tables"]["iterations"], miter_origin=1)
    result = D.new_extension_hits(**f.args, config=f.config)
    assert result["n_hits"] == 2 and result["final"]["agreement"]["exact"]


def test_next_captured_ca_is_comparison_only_not_constructor_input(real_population):
    f = real_population
    expected = D.new_extension_hits(**f.args, config=f.config)
    rows = ak.to_list(f.args["rows"])
    for row in rows:
        if row["input_index"] != 7 and row["miter"] == 4:
            row["sensor_poca"][0] += 3.
            row["particle_direction"][0] += .2
    f.args["rows"] = ak.Array(rows)
    actual = D.new_extension_hits(**f.args, config=f.config)
    for field in ("constructor_state", "final_state", "ztime", "sensor_hint"):
        assert actual[field] == expected[field]
    assert not actual["final"]["agreement"]["criterion"]
    assert actual["n_hits"] == 2 and actual["n_excluded"] == 0


@pytest.mark.parametrize("axis,value,accepted", [
    ("time", -60., True), ("time", np.nextafter(-60., -np.inf), False),
    ("time", np.nextafter(60., -np.inf), True), ("time", 60., False),
    ("dt", np.nextafter(50., 0.), True), ("dt", 50., False),
    ("dt", np.nextafter(-50., 0.), True), ("dt", -50., False),
    ("doca", np.nextafter(10., 0.), True), ("doca", 10., False),
    ("doca", np.nextafter(-10., 0.), True), ("doca", -10., False),
    ("doca", 0., True),
])
def test_constructor_cut_boundaries_and_zero_doca(real_population, monkeypatch, axis, value, accepted):
    f = real_population
    measured_join, ztime, piecewise = D.measured_for_hits, D.H.ztime, D.H.L.piecewise
    def measured(*a, **kw):
        m, g, line, origin = measured_join(*a, **kw)
        m = dict(m, corrected_time=np.full(2, value if axis == "time" else 0.))
        return m, g, line, origin
    def zt(*a, **kw):
        result = ztime(*a, **kw)
        result["time"] = np.full(2, value if axis == "time" else (-value if axis == "dt" else 0.))
        return result
    def built(*a, **kw):
        result = piecewise(*a, **kw)
        result["delta"] = np.tile([value if axis == "doca" else 1., 0., 0.], (2, 1))
        return result
    monkeypatch.setattr(D, "measured_for_hits", measured)
    monkeypatch.setattr(D.H, "ztime", zt)
    monkeypatch.setattr(D.H.L, "piecewise", built)
    # Boundary controls isolate cut arithmetic; reference updating is separately
    # exercised by the real-population tests above. No derivative API is supplied.
    monkeypatch.setattr(D.H, "hit_update", lambda p, r, c, parent, line, state, *a:
                        (state, state, dict(retry=np.zeros(2, bool), give_up=np.zeros(2, bool))))
    result = D.new_extension_hits(**f.args, config=f.config)
    assert result["accepted_membership_kinematic_checks_passed"] is accepted
    assert result["n_measured_kinematic_cuts_failed"] == (0 if accepted else 2)
    assert result["n_hits"] == 2 and result["n_excluded"] == 0


@pytest.mark.parametrize("scope", ["FitSettings", "ExtensionSettings"])
def test_unused_empty_chi2_settings_allowed_but_not_schedule_changes(tmp_path, scope):
    blocks = expanded_blocks()
    blocks[scope] += "  Chi2SHUSettings: []\n"
    config = configure(tmp_path, expanded_text(blocks))
    assert config[scope]["schedule"] == (SEED if scope == "FitSettings" else EXTENSION)
    blocks[scope] = blocks[scope].replace('"CADSHU"', '"CADSHU:Chi2SHU"', 1)
    with pytest.raises(ValueError, match="schedule"):
        configure(tmp_path, expanded_text(blocks))


@pytest.mark.parametrize("scope", ["FitSettings", "ExtensionSettings"])
@pytest.mark.parametrize("extra", ["Chi2SHUSettings: [1]", "Chi2SHUSettings: []\n  Chi2SHUSettings: []"])
def test_nonempty_or_duplicate_inactive_chi2_rejected(tmp_path, scope, extra):
    blocks = expanded_blocks()
    blocks[scope] += "  " + extra + "\n"
    with pytest.raises(ValueError, match="updater"):
        configure(tmp_path, expanded_text(blocks))


@pytest.mark.parametrize("scope,index,value", [("FitSettings", 3, True), ("ExtensionSettings", 3, False)])
def test_schedule_booleans_are_not_numeric_temperatures(tmp_path, scope, index, value):
    blocks = expanded_blocks()
    original = SEED if scope == "FitSettings" else EXTENSION
    changed = deepcopy(original)
    changed[index][0] = value
    blocks[scope] = blocks[scope].replace(json.dumps(original), json.dumps(changed))
    with pytest.raises(ValueError, match="schedule"):
        configure(tmp_path, expanded_text(blocks))


@pytest.mark.parametrize("field,value", [("MaxStrawHitDt", "50garbage"), ("MaxStrawHitDOCA", "10oops"),
                                         ("TPOCAPrecision", "0.0001junk")])
def test_scalar_config_requires_entire_numeric_token(tmp_path, field, value):
    blocks = expanded_blocks()
    original = {"MaxStrawHitDt": "50", "MaxStrawHitDOCA": "10", "TPOCAPrecision": "0.0001"}[field]
    blocks["KKFitSettings"] = blocks["KKFitSettings"].replace(field + ": " + original, field + ": " + value)
    with pytest.raises(ValueError):
        configure(tmp_path, expanded_text(blocks))


@pytest.mark.parametrize("field", ["time", "precision", "ca_status", "ca_usable"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_scalar_observation_missing_and_nonfinite_fail_closed(field, value):
    hits, rows = state_fixture()
    (hits if field == "time" else rows)[0][field] = value
    with pytest.raises(ValueError):
        D.observed_state(ak.Array(hits), ak.Array(rows))


@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_measured_z_mask_and_nonfinite_are_not_exclusions(real_population, value):
    f = real_population
    events = ak.to_list(f.args["events"])
    events[0]["s_z"][1] = value
    table = ak.Array(events)
    f.args["events"] = ak.with_field(table, ak.values_astype(table.s_z, np.float32), "s_z")
    with pytest.raises(ValueError, match="missing|nonfinite"):
        D.new_extension_hits(**f.args, config=f.config)


def test_zero_constructor_doca_runs_real_reference_update_without_derivatives(real_population, monkeypatch):
    import f1_captured_ca_audit as captured
    f = real_population
    piecewise, updates = D.H.L.piecewise, []
    def zero_doca(*a, **kw):
        result = piecewise(*a, **kw)
        result["delta"][:] = 0.
        result["sensor_poca"] = result["particle_poca"].copy()
        updates.append(len(result["usable"]))
        return result
    def forbidden(*a, **kw):
        pytest.fail("geometry-only constructor audit must not require nonzero-DOCA derivatives")
    monkeypatch.setattr(D.H.L, "piecewise", zero_doca)
    monkeypatch.setattr(captured, "fixed_state_rows", forbidden)
    result = D.new_extension_hits(**f.args, config=f.config)
    assert len(updates) >= 2  # constructor and real retained-reference solve
    assert result["constructor_state"]["delta"] == [[0., 0., 0.]] * 2
    assert result["accepted_membership_kinematic_checks_passed"]
    assert result["n_hits"] == 2 and result["n_excluded"] == 0


@pytest.mark.parametrize("field", ["mass", "parameters", "range"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_constructor_source_operands_missing_or_nonfinite_fail_closed(real_population, field, value):
    f = real_population
    name = "seed_contract" if field == "mass" else "pieces"
    rows = ak.to_list(f.args["tables"][name])
    if field == "mass":
        rows[0][field] = value
    else:
        row = next(r for r in rows if r["owner"] == 1 and r["miter"] == 3)
        row[field][0] = value
    f.args["tables"][name] = ak.Array(rows)
    with pytest.raises(ValueError, match="missing|nonfinite|finite"):
        D.new_extension_hits(**f.args, config=f.config)


@pytest.mark.parametrize("field", ["corrected_time", "dtime", "wdist"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_measured_constructor_inputs_nonfinite_fail_closed(real_population, field, value):
    f = real_population
    f.measured[field][1] = value
    with pytest.raises(ValueError, match="nonfinite|finite"):
        D.new_extension_hits(**f.args, config=f.config)
