"""Synthetic crossing observability contracts; no native capture, replay or remote IO."""
from copy import deepcopy
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_crossing_observability as X


TARGETS = [(1, 2, 3, 4), (9, 2, 3, 4), (1, 9, 3, 4), (1, 2, 9, 4), (1, 2, 3, 9)]
CUTS = {"AddStrawMinDz": 0, "MaxDStraw": 2, "MaxStrawDOCA": 5,
        "MaxStrawDOCAConsistency": 1, "MaxStrawUposBuffer": 0, "StrawNBuffer": 0}


def material_text():
    return ("KKFitSettings: {\n" + "".join(f"  {k}: {v}\n" for k, v in CUTS.items())
            + "  SkipStrawCheck: false\n}\n")


def isolated_material(tmp_path, monkeypatch, text):
    path = tmp_path / "expanded.fcl"
    path.write_text(text)
    monkeypatch.setattr(X.D, "configuration", lambda supplied: None)
    return X.material_configuration(path)


@pytest.mark.parametrize("layout", ["single", "identical_copies", "nested", "mixed_indentation"])
def test_material_normal_branch_and_identical_expanded_copies(tmp_path, monkeypatch, layout):
    text = material_text()
    nested = "outer: {\n" + "\n".join("  " + line for line in text.splitlines()) + "\n}\n"
    text = {"single": text, "identical_copies": text + text, "nested": nested,
            "mixed_indentation": text + nested}[layout]
    if layout == "mixed_indentation":
        with pytest.raises(ValueError, match="ambiguous material configuration"):
            isolated_material(tmp_path, monkeypatch, text)
        return
    result = isolated_material(tmp_path, monkeypatch, text)
    assert {k: result[k] for k in CUTS} == CUTS
    assert result["SkipStrawCheck"] is False
    assert result["vertical_branch_reachable"] is False
    assert result["discovery_replayed"] is False
    assert "AddStrawMinDz=0" in result["branch_basis"]
    assert "does not establish finite/nonzero" in result["branch_basis"]


def test_material_invokes_upstream_configuration_first(tmp_path, monkeypatch):
    path = tmp_path / "expanded.fcl"
    path.write_text(material_text())
    calls = []

    def gate(supplied):
        calls.append(supplied)
        raise ValueError("upstream configuration sentinel")

    monkeypatch.setattr(X.D, "configuration", gate)
    with pytest.raises(ValueError, match="upstream configuration sentinel"):
        X.material_configuration(path)
    assert calls == [path]


@pytest.mark.parametrize("name", list(CUTS))
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong", "quoted", "nan", "inf", "suffix", "other_scope"])
def test_material_strict_numeric_fields(tmp_path, monkeypatch, name, mutation):
    text = material_text()
    line = f"  {name}: {CUTS[name]}\n"
    replacement = {"missing": "", "duplicate": line + line, "wrong": f"  {name}: {CUTS[name]+1}\n",
                   "quoted": f'  {name}: "{CUTS[name]}"\n', "nan": f"  {name}: nan\n",
                   "inf": f"  {name}: inf\n", "suffix": f"  {name}: {CUTS[name]}junk\n", "other_scope": ""}[mutation]
    text = text.replace(line, replacement)
    if mutation == "other_scope":
        text += "Other: {\n" + line + "}\n"
    with pytest.raises(ValueError, match="unsupported material " + name):
        isolated_material(tmp_path, monkeypatch, text)


@pytest.mark.parametrize("replacement", ["", "  SkipStrawCheck: true\n", "  SkipStrawCheck: 0\n",
    '  SkipStrawCheck: "false"\n', "  SkipStrawCheck: False\n", "  SkipStrawCheck: false\n" * 2])
def test_material_skip_is_single_literal_false(tmp_path, monkeypatch, replacement):
    with pytest.raises(ValueError, match="SkipStrawCheck"):
        isolated_material(tmp_path, monkeypatch, material_text().replace("  SkipStrawCheck: false\n", replacement))


@pytest.mark.parametrize("text", ["", "Other: {}\n", material_text() + material_text().replace("MaxDStraw: 2", "MaxDStraw: 3")])
def test_material_ambiguous_or_missing_block(tmp_path, monkeypatch, text):
    with pytest.raises(ValueError, match="ambiguous material"):
        isolated_material(tmp_path, monkeypatch, text)


def lifetime_fixture():
    """Same local IDs on five targets, each with initial, two associated and material."""
    hk, ck, partner = [], [], []
    for target in TARGETS:
        start = len(hk)
        hk.extend([list(target) + [i] for i in (4, 9, 12)])
        ck.extend([list(target) + [i] for i in (0, 10, 20, 30)])
        partner.extend((start, start + 1, start + 2, -1))
    return dict(
        hits=dict(source_keys=np.array(hk), initial=np.tile([True, False, False], len(TARGETS)),
                  first_ordinal=np.tile([0, 4, 4], len(TARGETS)),
                  shindex=np.tile([100, 200, 300], len(TARGETS))),
        crossings=dict(source_keys=np.array(ck), initial=np.tile([True, False, False, False], len(TARGETS)),
                       first_ordinal=np.tile([0, 4, 4, 4], len(TARGETS)),
                       partner_hit_rows=np.array(partner), strawid=np.tile([11, 12, 13, 14], len(TARGETS))))


def test_source_partition_full_target_keys_and_source_order_not_storage_order():
    data = lifetime_fixture()
    order = np.arange(len(data["crossings"]["initial"]))[::-1]
    data["crossings"] = {k: v[order] for k, v in data["crossings"].items()}
    result = X.source_partition(data)
    np.testing.assert_array_equal(result["labels"], np.tile(
        ["initial", "associated_extension", "associated_extension", "material_extension"], len(TARGETS))[order])
    assert result["census"] == {
        "initial": {"n_objects": 5, "n_targets": 5},
        "associated_extension": {"n_objects": 10, "n_targets": 5},
        "material_extension": {"n_objects": 5, "n_targets": 5}}
    assert result["associated_before_material"] is True
    assert result["associated_in_combohit_order"] is True
    assert result["bad_order_targets"] == result["bad_hit_order_source_keys"] == []
    assert result["native_constructor_order_observed"] is False


@pytest.mark.parametrize("target", range(len(TARGETS)))
def test_source_append_order_disagreement_retained_with_exact_target(target):
    data = lifetime_fixture()
    data["crossings"]["source_keys"][4 * target + 3, 4] = 15
    result = X.source_partition(data)
    assert result["associated_before_material"] is False
    assert result["bad_order_targets"] == [list(TARGETS[target])]
    assert sum(v["n_objects"] for v in result["census"].values()) == 20


def test_source_hit_order_uses_shindex_not_hit_input_index():
    data = lifetime_fixture()
    data["hits"]["shindex"][2] = 199
    result = X.source_partition(data)
    assert result["associated_in_combohit_order"] is False
    assert result["bad_hit_order_source_keys"] == [list(TARGETS[0]) + [20]]
    data["hits"]["shindex"][2] = 200
    assert X.source_partition(data)["associated_in_combohit_order"] is False


@pytest.mark.parametrize("family,field", [("hits", "source_keys"), ("hits", "initial"),
    ("hits", "first_ordinal"), ("hits", "shindex"), ("crossings", "source_keys"),
    ("crossings", "initial"), ("crossings", "first_ordinal"),
    ("crossings", "partner_hit_rows"), ("crossings", "strawid")])
def test_source_masks_fail_closed(family, field):
    data = lifetime_fixture()
    value = data[family][field]
    mask = np.zeros(value.shape, dtype=bool)
    mask.flat[0] = True
    data[family][field] = np.ma.array(value, mask=mask)
    with pytest.raises(ValueError, match="missing"):
        X.source_partition(data)


@pytest.mark.parametrize("mutation", ["unpaired_initial", "two_partners", "birth", "initial_class", "bounds",
    "cross_target", "duplicate_source", "duplicate_straw", "integer_flag", "float_partner", "negative_shindex"])
def test_source_malformed_association_is_not_partitioned(mutation):
    data = lifetime_fixture()
    c, h = data["crossings"], data["hits"]
    if mutation == "unpaired_initial":
        c["partner_hit_rows"][0] = -1
    elif mutation == "two_partners":
        c["partner_hit_rows"][3] = 1
    elif mutation == "birth":
        c["first_ordinal"][1] += 1
    elif mutation == "initial_class":
        h["initial"][1] = True
    elif mutation == "bounds":
        c["partner_hit_rows"][1] = len(h["initial"])
    elif mutation == "cross_target":
        c["partner_hit_rows"][1] = 4
    elif mutation == "duplicate_source":
        c["source_keys"][2] = c["source_keys"][1]
    elif mutation == "duplicate_straw":
        c["strawid"][3] = c["strawid"][1]
    elif mutation == "integer_flag":
        c["initial"] = c["initial"].astype(int)
    elif mutation == "float_partner":
        c["partner_hit_rows"] = c["partner_hit_rows"].astype(float)
    else:
        h["shindex"][1] = -1
    with pytest.raises(ValueError):
        X.source_partition(data)


def reference_fixture():
    effects, pieces, iterations = [], [], []
    for t, target in enumerate(TARGETS):
        for ordinal in (0, 1):
            base = dict(zip(X.KEYS, target), iteration_ordinal=ordinal, miter=ordinal, iter=2)
            iterations.append(base)
            for index in (0, 1):
                value = 100 * t + 30 * ordinal + 10 * index
                state = dict(range=[value, value + 10.], parameters=(np.arange(6) + value).tolist(),
                             covariance=((np.eye(6) * (value + 1)).ravel()).tolist(), bnom=[0., 0., value + 1.])
                pieces.append(dict(base, owner=0, piece_index=index, **state))
                other = deepcopy(state)
                other["parameters"][0] += 999
                pieces.append(dict(base, owner=1, piece_index=index, **other))
                effects.append(dict(base, identity_kind=5, effect_index=30 + index, input_index=7 + index,
                                    has_reference=True, time=-999., **{"reference_" + k: v for k, v in state.items()}))
    return effects, pieces, iterations


def reference_call(effects, pieces, iterations):
    return X.reference_membership(ak.Array(effects), ak.Array(pieces), ak.Array(iterations))


def test_reference_every_iteration_old_owner_and_full_identity():
    e, p, i = reference_fixture()
    # Deliberately unrelated effect times and differently ordered owners/tables.
    result = reference_call(e[::-1], p[1::2] + p[::2][::-1], i[::-1])
    assert result["exact"] is True
    assert result["n_snapshots"] == 20 and result["n_targets"] == 5
    assert result["n_excluded"] == 0
    assert result["piece_index"] == [1, 0] * 10
    np.testing.assert_array_equal(result["snapshot_keys"], X.keys(ak.Array(e[::-1]), X.HIT_KEYS))
    assert set(result["fields"]) == {"range", "parameters", "covariance", "bnom"}
    assert all(v == {"exact": True, "n_unequal": 0, "mismatch_keys": []} for v in result["fields"].values())


@pytest.mark.parametrize("field", ["range", "parameters", "covariance", "bnom"])
def test_reference_mismatch_is_reported_with_no_exclusions_or_tolerance(field):
    e, p, i = reference_fixture()
    e[7]["reference_" + field] = deepcopy(e[7]["reference_" + field])
    old = e[7]["reference_" + field][0]
    e[7]["reference_" + field][0] = np.nextafter(float(old), np.inf)
    result = reference_call(e, p, i)
    assert result["exact"] is False
    assert result["n_snapshots"] == len(e) and result["n_excluded"] == 0
    assert result["snapshot_keys"] == X.keys(ak.Array(e), X.HIT_KEYS).tolist()
    assert result["fields"][field] == dict(exact=False, n_unequal=1,
        mismatch_keys=[[e[7][k] for k in X.HIT_KEYS]])
    assert all(v["exact"] for k, v in result["fields"].items() if k != field)


@pytest.mark.parametrize("mutation", ["duplicate_effect", "duplicate_input", "duplicate_piece", "duplicate_iteration",
    "missing_iteration", "missing_owner", "missing_piece", "wrong_iteration", "miter", "iter", "no_reference",
    "empty_crossings", "negative_key"])
def test_reference_identity_and_membership_guards(mutation):
    e, p, i = reference_fixture()
    if mutation == "duplicate_effect":
        e.append(deepcopy(e[0]))
    elif mutation == "duplicate_input":
        e[1]["input_index"] = e[0]["input_index"]
    elif mutation == "duplicate_piece":
        p.append(deepcopy(p[0]))
    elif mutation == "duplicate_iteration":
        i.append(deepcopy(i[0]))
    elif mutation == "missing_iteration":
        i.pop(0)
    elif mutation == "missing_owner":
        p = p[1::2]
    elif mutation == "missing_piece":
        p.pop(0)  # Leaves a noncontiguous old owner.
    elif mutation == "wrong_iteration":
        e[0]["iteration_ordinal"] = 99
    elif mutation in ("miter", "iter"):
        e[0][mutation] += 1
    elif mutation == "no_reference":
        e[0]["has_reference"] = False
    elif mutation == "empty_crossings":
        for row in e:
            row["identity_kind"] = 1
    else:
        e[0]["run"] = -1
    with pytest.raises(ValueError):
        reference_call(e, p, i)


@pytest.mark.parametrize("table,field", [("effects", "run"), ("effects", "iteration_ordinal"),
    ("effects", "input_index"), ("effects", "has_reference"), ("effects", "time"),
    ("effects", "reference_range"), ("effects", "reference_parameters"),
    ("effects", "reference_covariance"), ("effects", "reference_bnom"),
    ("pieces", "range"), ("pieces", "parameters"), ("pieces", "covariance"), ("pieces", "bnom"),
    ("iterations", "event")])
def test_reference_missing_values_rejected(table, field):
    e, p, i = reference_fixture()
    rows = {"effects": e, "pieces": p, "iterations": i}[table]
    if isinstance(rows[0][field], list):
        rows[0][field] = deepcopy(rows[0][field])
        rows[0][field][0] = None
    else:
        rows[0][field] = None
    with pytest.raises(ValueError):
        reference_call(e, p, i)


@pytest.mark.parametrize("table,field", [("effects", "time"), ("effects", "reference_range"),
    ("effects", "reference_parameters"), ("effects", "reference_covariance"), ("effects", "reference_bnom"),
    ("pieces", "range"), ("pieces", "parameters"), ("pieces", "covariance"), ("pieces", "bnom")])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_reference_nonfinite_guards(table, field, bad):
    e, p, i = reference_fixture()
    rows = e if table == "effects" else p
    if isinstance(rows[0][field], list):
        rows[0][field] = deepcopy(rows[0][field])
        rows[0][field][0] = bad
    else:
        rows[0][field] = bad
    with pytest.raises(ValueError):
        reference_call(e, p, i)


@pytest.mark.parametrize("value", [[1., 1.], [2., 1.], [0., 1., 2.]])
def test_reference_range_positive_and_width(value):
    e, p, i = reference_fixture()
    e[0]["reference_range"] = value
    with pytest.raises(ValueError):
        reference_call(e, p, i)


def hints_fixture():
    lifetime = lifetime_fixture()
    h, c = lifetime["hits"], lifetime["crossings"]
    rows = []
    h["first_rows"] = np.arange(len(h["initial"]))
    for j, key in enumerate(h["source_keys"]):
        base = dict(zip(X.KEYS, key[:4]), iteration_ordinal=int(h["first_ordinal"][j]),
                    effect_index=40 + j % 3, input_index=int(key[4]), identity_kind=1,
                    strawid=11 + j % 3, shindex=int(h["shindex"][j]), partner_shindex=-1,
                    time=50. + j)
        rows.append(base)
    c["first_rows"] = np.arange(len(c["initial"])) + len(rows)
    for j, key in enumerate(c["source_keys"]):
        partner = c["partner_hit_rows"][j]
        rows.append(dict(zip(X.KEYS, key[:4]), iteration_ordinal=int(c["first_ordinal"][j]),
                         effect_index=60 + j % 4, input_index=int(key[4]), identity_kind=5,
                         strawid=int(c["strawid"][j]), shindex=-1,
                         partner_shindex=int(h["shindex"][partner]) if partner >= 0 else -1,
                         time=(rows[partner]["time"] - 20. if partner >= 0 else -999.)))
    selected = h["first_rows"][~h["initial"]]
    hit_keys = X.keys(ak.Array(rows)[selected], X.HIT_KEYS)
    # Reverse predecessor storage; constructor and final-state values intentionally disagree.
    values = np.arange(len(selected), dtype=float) + 1000.
    new = dict(hit_keys=hit_keys[::-1], accepted_membership_kinematic_checks_passed=True,
               constructor_state=dict(particle_time=values[::-1], sensor_time=(values + .125)[::-1],
                                      usable=np.ones(len(selected), dtype=bool)),
               final=dict(agreement=dict(exact=True), particle_time=np.full(len(selected), -123.),
                          sensor_time=np.full(len(selected), -456.)))
    return lifetime, rows, {"new_extension_hits": new}


def hints_call(lifetime, rows, predecessor):
    return X.associated_hints(lifetime, ak.Array(rows), predecessor)


def test_hints_join_constructor_not_final_ca_and_do_not_claim_observations():
    data, rows, predecessor = hints_fixture()
    result = hints_call(data, rows, predecessor)
    expected = np.arange(10, dtype=float) + 1000.
    assert result["n_crossings"] == 10
    np.testing.assert_array_equal(result["reconstructed_hit_constructor_particle_hint"], expected)
    np.testing.assert_array_equal(result["reconstructed_hit_constructor_sensor_hint"], expected + .125)
    assert result["straw_line_measurement_time"] == expected.tolist()
    assert result["partner_hit_keys"] == predecessor["new_extension_hits"]["hit_keys"][::-1].tolist()
    assert result["source_keys"] == data["crossings"]["source_keys"][np.tile([False, True, True, False], 5)].tolist()
    assert result["first_crossing_minus_hit_effect_time"] == [-20.] * 10
    assert result["n_nonpositive_effect_time_gaps"] == 10
    assert result["effect_time_gap_min"] == result["effect_time_gap_max"] == -20.
    for flag in ("effect_time_gap_is_private_offset", "constructor_hints_natively_observed",
                 "crossing_constructor_ca_replayed", "private_offset_compared", "retained_crossing_ca_compared"):
        assert result[flag] is False
    assert X.source_partition(data)["associated_before_material"] is True


def test_hints_effect_time_changes_only_reported_gap_not_constructor_inputs():
    data, rows, predecessor = hints_fixture()
    before = hints_call(data, rows, predecessor)
    chosen = data["crossings"]["first_rows"][1]
    rows[chosen]["time"] += 100.
    after = hints_call(data, rows, predecessor)
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"first_crossing_minus_hit_effect_time", "effect_time_gap_max", "n_nonpositive_effect_time_gaps"}
    assert after["first_crossing_minus_hit_effect_time"] == [80.] + [-20.] * 9
    assert after["effect_time_gap_max"] == 80. and after["n_nonpositive_effect_time_gaps"] == 9
    assert X.source_partition(data)["associated_before_material"] is True


def test_hints_first_row_offsets_survive_effect_storage_reordering():
    data, rows, predecessor = hints_fixture()
    expected = hints_call(data, rows, predecessor)
    order = np.arange(len(rows))[::-1]
    inverse = np.argsort(order)
    data["hits"]["first_rows"] = inverse[data["hits"]["first_rows"]]
    data["crossings"]["first_rows"] = inverse[data["crossings"]["first_rows"]]
    assert hints_call(data, ak.to_list(ak.Array(rows)[order]), predecessor) == expected


@pytest.mark.parametrize("family", ["hits", "crossings"])
@pytest.mark.parametrize("field", list(X.KEYS) + ["input_index", "identity_kind", "strawid", "iteration_ordinal"])
def test_hints_firstrow_fullkey_kind_straw_and_iteration_guards(family, field):
    data, rows, predecessor = hints_fixture()
    index = data[family]["first_rows"][1]
    rows[index][field] += 100
    with pytest.raises(ValueError):
        hints_call(data, rows, predecessor)


@pytest.mark.parametrize("column", range(6))
def test_hints_predecessor_join_requires_every_full_key_component(column):
    data, rows, predecessor = hints_fixture()
    predecessor["new_extension_hits"]["hit_keys"][0, column] += 100
    with pytest.raises(ValueError, match="identity join"):
        hints_call(data, rows, predecessor)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "float", "negative", "masked"])
def test_hints_predecessor_coverage_and_identity_guards(mutation):
    data, rows, predecessor = hints_fixture()
    new = predecessor["new_extension_hits"]
    k = new["hit_keys"]
    if mutation == "duplicate":
        k[1] = k[0]
    elif mutation == "missing":
        new["hit_keys"] = k[:-1]
    elif mutation == "extra":
        extra = k[:1].copy()
        extra[0, -1] += 100
        new["hit_keys"] = np.concatenate((k, extra))
    elif mutation == "float":
        new["hit_keys"] = k.astype(float)
    elif mutation == "negative":
        k[0, 0] = -1
    else:
        mask = np.zeros(k.shape, dtype=bool)
        mask[0, 0] = True
        new["hit_keys"] = np.ma.array(k, mask=mask)
    with pytest.raises(ValueError):
        hints_call(data, rows, predecessor)


@pytest.mark.parametrize("flag", ["exact", "accepted_membership_kinematic_checks_passed"])
@pytest.mark.parametrize("value", [False, 1, "true", np.bool_(True), None])
def test_hints_receipt_flags_must_be_literal_true(flag, value):
    data, rows, predecessor = hints_fixture()
    new = predecessor["new_extension_hits"]
    (new["final"]["agreement"] if flag == "exact" else new)[flag] = value
    with pytest.raises(ValueError, match="predecessor"):
        hints_call(data, rows, predecessor)


@pytest.mark.parametrize("field", ["particle_time", "sensor_time", "usable"])
@pytest.mark.parametrize("mutation", ["shape", "masked", "invalid"])
def test_hints_constructor_state_guards(field, mutation):
    data, rows, predecessor = hints_fixture()
    state = predecessor["new_extension_hits"]["constructor_state"]
    if mutation == "shape":
        state[field] = state[field][:, None]
    elif mutation == "masked":
        mask = np.zeros(state[field].shape, dtype=bool)
        mask[0] = True
        state[field] = np.ma.array(state[field], mask=mask)
    elif field == "usable":
        state[field] = state[field].astype(int)
    else:
        state[field][0] = np.nan
    with pytest.raises(ValueError):
        hints_call(data, rows, predecessor)


@pytest.mark.parametrize("mutation", ["unusable", "partner_shindex", "cross_time", "hit_time", "masked_firstrow"])
def test_hints_partner_and_time_guards(mutation):
    data, rows, predecessor = hints_fixture()
    if mutation == "unusable":
        predecessor["new_extension_hits"]["constructor_state"]["usable"][0] = False
    elif mutation == "partner_shindex":
        rows[data["crossings"]["first_rows"][1]]["partner_shindex"] += 1
    elif mutation in ("cross_time", "hit_time"):
        family = "crossings" if mutation == "cross_time" else "hits"
        rows[data[family]["first_rows"][1]]["time"] = np.inf
    else:
        a = data["crossings"]["first_rows"]
        mask = np.zeros(a.shape, dtype=bool)
        mask[1] = True
        data["crossings"]["first_rows"] = np.ma.array(a, mask=mask)
    with pytest.raises(ValueError):
        hints_call(data, rows, predecessor)


def test_material_normal_branch_with_real_upstream_configuration(tmp_path):
    # Same schedules and KK contract used by transition audit fixtures.
    seed = '[[5,"CADSHU"],[5,"CADSHU"],[2,"CADSHU"],[1,"CADSHU"]]'
    extension = ('[[2,"CADSHU"],[1,"CADSHU"],[0.3,"CADSHU"],[0,"CADSHU:BkgANNSHU"],'
                 '[1,"CADSHU:DriftANNSHU"],[0.5,"CADSHU:DriftANNSHU"],'
                 '[0.2,"CADSHU:DriftANNSHU"],[0,"CADSHU:DriftANNSHU:BkgANNSHU"],'
                 '[0,"CADSHU:DriftANNSHU"]]')
    text = "".join(name + ': {\n  ProcessEnds: false\n  BFieldCorrection: false\n'
                   + '  MetaIterationSettings: ' + schedule + '\n}\n'
                   for name, schedule in (("FitSettings", seed), ("ExtensionSettings", extension)))
    kk = ('  StrawHitClusterLevel: "none"\n  UseCaloCluster: false\n  AddHits: true\n'
          '  AddMaterial: true\n  MaterialCorrection: true\n  MaxStrawHitDt: 50\n'
          '  MaxStrawHitDOCA: 10\n  TPOCAPrecision: 0.0001\n')
    text += material_text().replace("KKFitSettings: {\n", "KKFitSettings: {\n" + kk)
    path = tmp_path / "expanded.fcl"
    path.write_text(text)
    result = X.material_configuration(path)
    assert result["AddStrawMinDz"] == 0
    assert result["vertical_branch_reachable"] is False


@pytest.mark.parametrize("name,value", [("AddStrawMinDz", "+0.0e0"), ("MaxDStraw", "2E0"),
                                        ("MaxStrawDOCAConsistency", "1.")])
def test_material_equivalent_numeric_spellings(tmp_path, monkeypatch, name, value):
    text = material_text().replace(f"{name}: {CUTS[name]}\n", f"{name}: {value}\n")
    assert isolated_material(tmp_path, monkeypatch, text)[name] == CUTS[name]


@pytest.mark.parametrize("family", ["hits", "crossings"])
def test_source_all_false_masks_are_not_missing(family):
    data = lifetime_fixture()
    expected = X.source_partition(data)
    data[family] = {key: np.ma.array(value, mask=np.zeros(value.shape, dtype=bool))
                    for key, value in data[family].items()}
    result = X.source_partition(data)
    np.testing.assert_array_equal(result.pop("labels"), expected.pop("labels"))
    assert result == expected


def test_source_no_material_partition_fails_closed():
    data = lifetime_fixture()
    keep = data["crossings"]["partner_hit_rows"] >= 0
    data["crossings"] = {k: v[keep] for k, v in data["crossings"].items()}
    with pytest.raises(ValueError, match="empty extension partition"):
        X.source_partition(data)


def test_hints_final_ca_and_unrelated_following_snapshot_times_are_not_constructor_inputs():
    data, rows, predecessor = hints_fixture()
    before = hints_call(data, rows, predecessor)
    new = predecessor["new_extension_hits"]
    new["final"]["particle_time"][:] = 999999.
    new["final"]["sensor_time"][:] = -999999.
    later = deepcopy(rows[data["crossings"]["first_rows"][1]])
    later.update(iteration_ordinal=5, time=1e20)
    rows.append(later)
    assert hints_call(data, rows, predecessor) == before


@pytest.mark.parametrize("field", ["particle_time", "sensor_time"])
def test_hints_supplied_constructor_inputs_remain_distinct(field):
    data, rows, predecessor = hints_fixture()
    before = hints_call(data, rows, predecessor)
    predecessor["new_extension_hits"]["constructor_state"][field][0] += .25
    after = hints_call(data, rows, predecessor)
    expected = ({"reconstructed_hit_constructor_particle_hint", "straw_line_measurement_time"}
                if field == "particle_time" else {"reconstructed_hit_constructor_sensor_hint"})
    assert {key for key in before if before[key] != after[key]} == expected
    output = "reconstructed_hit_constructor_" + ("particle_hint" if field == "particle_time" else "sensor_hint")
    assert after[output][-1] == before[output][-1] + .25
