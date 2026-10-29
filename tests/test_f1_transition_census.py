"""Synthetic full-key transition/owner/lifetime contracts; CPU only, no capture IO."""
from copy import deepcopy
from pathlib import Path
import json
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_transition_census as T


TARGET = dict(run=1, subrun=2, event=3, iseed=4)


def iterations(stages=((0, 0), (0, 1)), target=None, count=2):
    return [dict(TARGET if target is None else target, iteration_ordinal=o, miter=m, iter=i,
                 swapped=o % 2 == 0, nold=count, ncandidate=count, candidate_present=count > 0)
            for o, (m, i) in enumerate(stages)]


def pieces(it):
    result = []
    for row in it:
        for owner, count in ((0, row["nold"]), (1, row["ncandidate"])):
            for i in range(count):
                result.append(dict({k: row[k] for k in (*T.ITER_KEYS, "miter", "iter")}, owner=owner,
                                   piece_index=i, parameters=[float(i)]*6, covariance=[float(i+1)]*36,
                                   range=[float(i), float(i+1)], bnom=[0., 0., .001], mass=105.658, charge=-1.))
    return result


def seed(target=None, initial=True):
    row = dict(TARGET if target is None else target)
    row.update(sh_index=[0] if initial else [], sh_shindex=[100] if initial else [],
               sh_strawid=[11] if initial else [], sx_index=[0] if initial else [],
               sx_strawid=[11] if initial else [], sx_partner_shindex=[100] if initial else [])
    return row


def effect(it, code=1, index=0, shindex=100, strawid=11, partner=100):
    return dict({k: it[k] for k in (*T.ITER_KEYS, "miter", "iter")}, identity_kind=code,
                input_index=index, shindex=shindex, strawid=strawid, partner_shindex=partner,
                identity_from_final_collections=True)


def lifetime_fixture():
    it = iterations(((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (5, 0)))
    effects = [effect(row, code) for row in it for code in (1, 5)]
    # input index zero in another family is allowed and must not enter joins.
    effects += [effect(row, code=2, index=0, shindex=999, strawid=999) for row in it]
    effects += [effect(row, code=1, index=1, shindex=101, strawid=12) for row in it[4:]]
    effects += [effect(row, code=5, index=1, strawid=12, partner=101) for row in it[4:]]
    effects += [effect(row, code=5, index=2, strawid=13, partner=-1) for row in it[4:]]
    return it, effects, [seed()]


def owners(it, ps=None):
    table = ak.Array(it)
    return T.owner_continuity(ak.Array(pieces(it) if ps is None else ps), table, T.iteration_links(table))


def lifetimes(it, eff, seeds):
    return T.object_lifetimes(ak.Array(eff), ak.Array(it), ak.Array(seeds))


def test_all_link_labels_unswapped_terminal_and_row_order():
    it = iterations(((0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (5, 0)))
    permutation = np.array([7, 2, 5, 0, 6, 1, 4, 3])
    table = ak.Array(it)[permutation]
    result = T.iteration_links(table)
    before = np.asarray(table.iteration_ordinal)[result["previous"]]
    after = np.asarray(table.iteration_ordinal)[result["following"]]
    np.testing.assert_array_equal(after, before+1)
    expected = np.array(["same_meta", "seed_meta", "seed_meta", "seed_meta", "extension_entry", "same_meta", "extension_meta"])
    np.testing.assert_array_equal(result["labels"], expected[before])
    assert result["census"]["n_links"] == 7
    assert result["census"]["n_unswapped"] == 3
    assert result["census"]["terminal_keys"] == [[1, 2, 3, 4, 7]]
    assert not result["census"]["complete_native_history_claim"]
    json.dumps(result["census"], allow_nan=False)


@pytest.mark.parametrize("field", T.KEYS)
def test_target_keys_never_cross_join(field):
    other = dict(TARGET)
    other[field] += 1
    it = iterations() + iterations(target=other)
    result = T.iteration_links(ak.Array(it))
    np.testing.assert_array_equal(result["previous"], [0, 2])
    np.testing.assert_array_equal(result["following"], [1, 3])
    assert result["census"]["n_terminal"] == 2
    ps = pieces(it)
    for row in ps:
        if row[field] == other[field]:
            row["parameters"] = [40.+row["piece_index"]]*6
    assert owners(it, ps)["equality"].all()


@pytest.mark.parametrize("change,match", [
    (lambda it: it.append(deepcopy(it[0])), "duplicate"),
    (lambda it: it[0].update(iteration_ordinal=2), "ordinal gaps"),
    (lambda it: it[1].update(iteration_ordinal=3), "ordinal gaps"),
    (lambda it: it[1].update(miter=-1), "invalid integer"),
    (lambda it: it[0].update(iter=-1), "invalid integer"),
    (lambda it: it[1].update(miter=2, iter=0), "unknown adjacent"),
    (lambda it: it[1].update(miter=1, iter=1), "unknown adjacent"),
    (lambda it: it[1].update(iter=2), "unknown adjacent"),
    (lambda it: it[1].update(miter=13), "configured stages"),
    (lambda it: [row.update(swapped=1) for row in it], "invalid dtype"),
])
def test_bad_iteration_identity_and_transition(change, match):
    it = iterations()
    change(it)
    with pytest.raises(ValueError, match=match):
        T.iteration_links(ak.Array(it))


def test_negative_terminal_stage_and_mask_rejected():
    it = iterations(((0, 0),))
    it[0]["iter"] = -1
    with pytest.raises(ValueError, match="invalid integer"):
        T.iteration_links(ak.Array(it))
    it = iterations()
    it[1]["swapped"] = None
    with pytest.raises(ValueError, match="missing values"):
        T.iteration_links(ak.Array(it))
    it = iterations()
    for row in it:
        row["history_index"] = -1
    with pytest.raises(ValueError, match="history_index"):
        T.iteration_links(ak.Array(it))


def test_stage_configuration():
    table = ak.Array(iterations(((0, 0), (1, 0), (2, 0))))
    result = T.iteration_links(table, seed_stages=1, extension_stages=2)
    assert result["labels"].tolist() == ["extension_entry", "extension_meta"]
    with pytest.raises(ValueError, match="stage counts"):
        T.iteration_links(table, seed_stages=True)


def test_exact_multi_piece_owner_comparison_shuffled():
    it = iterations()
    ps = pieces(it)[::-1]
    result = owners(it, ps)
    assert result["equality"].tolist() == [True]
    assert result["counts_previous"].tolist() == result["counts_following"].tolist() == [2]
    assert all(v.tolist() == [True] for v in result["field_equal"].values())
    json.dumps(result["census"], allow_nan=False)


@pytest.mark.parametrize("field", ["parameters", "covariance", "range", "bnom", "mass", "charge"])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_all_fields_exact_finite_and_inequality_not_exception(field, nonfinite):
    it = iterations()
    ps = pieces(it)
    row = next(row for row in ps if row["iteration_ordinal"] == 1 and row["owner"] == 0 and row["piece_index"] == 1)
    if isinstance(row[field], list):
        row[field][-1] = np.nan if nonfinite else row[field][-1]+1e-12
    else:
        row[field] = np.inf if nonfinite else row[field]+1e-12
    result = owners(it, ps)
    assert result["equality"].tolist() == [False]
    assert not result["field_equal"][field][0]
    assert result["census"]["unequal_previous_keys"] == [[1, 2, 3, 4, 0]]
    assert result["census"]["unequal_following_keys"] == [[1, 2, 3, 4, 1]]
    json.dumps(result["census"], allow_nan=False)


def test_unequal_counts_missing_unswapped_and_zero_zero_are_not_matches():
    it = iterations()
    it[0].update(swapped=False, ncandidate=0, candidate_present=False)
    result = owners(it)
    assert result["equality"].tolist() == [False]
    assert result["census"]["n_missing_candidate_unswapped"] == 1
    assert result["counts_previous"].tolist() == [0]
    assert result["counts_following"].tolist() == [2]
    it[1]["nold"] = 0
    result = owners(it)
    assert result["count_equal"].tolist() == [True]
    assert result["equality"].tolist() == [False]
    it = iterations()
    it[1]["nold"] = 1
    assert owners(it)["equality"].tolist() == [False]


@pytest.mark.parametrize("mutation,match", [
    (lambda ps: ps.append(deepcopy(ps[0])), "duplicate"),
    (lambda ps: ps.pop(), "count mismatch"),
    (lambda ps: ps[-1].update(piece_index=2), "outside owner"),
    (lambda ps: ps[-1].update(owner=2), "unknown piece owner"),
    (lambda ps: ps[-1].update(miter=1), "parent identity"),
    (lambda ps: ps[-1].update(parameters=[0.]*5), "shape"),
    (lambda ps: ps[-1].update(event=7), "missing full identity"),
])
def test_malformed_owners_even_terminal_unselected(mutation, match):
    it = iterations()
    ps = pieces(it)
    mutation(ps)
    with pytest.raises(ValueError, match=match):
        owners(it, ps)


def test_link_shape_and_full_keys_validated():
    it = iterations()
    table = ak.Array(it)
    links = T.iteration_links(table)
    links["previous"] = np.array([[0]])
    with pytest.raises(ValueError, match="shape"):
        T.owner_continuity(ak.Array(pieces(it)), table, links)


def test_lifetime_census_initial_extension_final_metadata_and_collisions():
    result = lifetimes(*lifetime_fixture())
    assert result["hits"]["initial"].tolist() == [True, False]
    assert result["hits"]["first_miter"].tolist() == [0, 4]
    assert result["crossings"]["partner_hit_rows"].tolist() == [0, 1, -1]
    c = result["census"]
    assert c["hits"]["n_extension"] == 1
    assert c["crossings"]["n_extension_associated"] == 1
    assert c["crossings"]["n_extension_unassociated"] == 1
    assert c["hits"]["no_disappearance_or_reappearance"]
    assert c["crossing_partner_basis"] == "final_collections_not_snapshot"
    assert not c["snapshot_partner_association_claim"]
    assert not c["extension_constructor_offsets_captured"]
    json.dumps(c, allow_nan=False)


@pytest.mark.parametrize("field", T.KEYS)
def test_lifetime_full_target_separation(field):
    it, eff, seeds = lifetime_fixture()
    it2, eff2, seeds2 = deepcopy((it, eff, seeds))
    for row in it2 + eff2 + seeds2:
        row[field] += 1
    for row in eff2:
        row["strawid"] += 50
    seeds2[0]["sh_strawid"][0] += 50
    seeds2[0]["sx_strawid"][0] += 50
    result = lifetimes(it+it2, eff+eff2, seeds+seeds2)
    assert result["census"]["hits"]["n_objects"] == 4
    assert result["census"]["crossings"]["n_objects"] == 6


def test_final_partner_need_not_exist_at_crossing_snapshot():
    it, eff, seeds = lifetime_fixture()
    # Final association can name a hit first seen at extension entry.
    seeds[0]["sx_partner_shindex"] = [101]
    seeds[0]["sx_strawid"] = [12]
    for row in eff:
        if row["identity_kind"] == 5 and row["input_index"] == 0:
            row.update(partner_shindex=101, strawid=12)
    result = lifetimes(it, eff, seeds)
    assert result["crossings"]["first_ordinal"][0] == 0
    assert result["hits"]["first_ordinal"][result["crossings"]["partner_hit_rows"][0]] == 4
    assert not result["census"]["snapshot_partner_association_claim"]


@pytest.mark.parametrize("mutation,match", [
    (lambda e, s: e[0].update(identity_from_final_collections=False), "final collections"),
    (lambda e, s: e[0].update(shindex=900), "identity changed"),
    (lambda e, s: e[1].update(partner_shindex=900), "identity changed"),
    (lambda e, s: e.append(deepcopy(e[0])), "duplicate identities"),
    (lambda e, s: s[0]["sx_partner_shindex"].__setitem__(0, 900), "initial seed identity"),
    (lambda e, s: e[-1].update(input_index=-1), "invalid integer identity"),
])
def test_lifetime_malformed_identities(mutation, match):
    it, eff, seeds = lifetime_fixture()
    mutation(eff, seeds)
    with pytest.raises(ValueError, match=match):
        lifetimes(it, eff, seeds)


def test_partner_join_rejects_missing_hit_and_wrong_straw():
    it, eff, seeds = lifetime_fixture()
    for row in eff:
        if row["identity_kind"] == 5 and row["input_index"] == 1:
            row["partner_shindex"] = 999
    with pytest.raises(ValueError, match="missing full identity"):
        lifetimes(it, eff, seeds)
    for row in eff:
        if row["identity_kind"] == 5 and row["input_index"] == 1:
            row["partner_shindex"] = 100
    with pytest.raises(ValueError, match="partner straw"):
        lifetimes(it, eff, seeds)


def test_final_unique_hit_identity_rejects_aliases():
    it, eff, seeds = lifetime_fixture()
    for row in eff:
        if row["identity_kind"] == 1 and row["input_index"] == 1:
            row["shindex"] = 100
    with pytest.raises(ValueError, match="duplicate final hit"):
        lifetimes(it, eff, seeds)


def test_disappearance_reappearance_reported_not_failure():
    it, eff, seeds = lifetime_fixture()
    eff = [row for row in eff if not (row["identity_kind"] == 1 and row["input_index"] == 0 and row["iteration_ordinal"] == 2)]
    result = lifetimes(it, eff, seeds)
    c = result["census"]["hits"]
    assert c["n_disappearances"] == c["n_reappearances"] == 1
    assert c["disappearance_keys"] == [[1, 2, 3, 4, 1, 0]]
    assert c["reappearance_keys"] == [[1, 2, 3, 4, 3, 0]]
    assert not c["no_disappearance_or_reappearance"]


def test_new_object_outside_extension_entry_rejected():
    it, eff, seeds = lifetime_fixture()
    eff.append(effect(it[3], index=1, shindex=101, strawid=12))
    with pytest.raises(ValueError, match="outside extension entry"):
        lifetimes(it, eff, seeds)


def test_initial_membership_uses_seed_index_not_count_cutoff():
    it, eff, seeds = lifetime_fixture()
    seeds[0]["sh_index"] = [19]
    seeds[0]["sx_index"] = [27]
    for row in eff:
        if row["identity_kind"] == 1 and row["input_index"] == 0:
            row["input_index"] = 19
        if row["identity_kind"] == 5 and row["input_index"] == 0:
            row["input_index"] = 27
    result = lifetimes(it, eff, seeds)
    assert result["census"]["hits"]["n_initial"] == 1
    assert result["census"]["crossings"]["n_initial"] == 1


def test_empty_links_and_empty_families():
    it = iterations(((0, 0),))
    result = owners(it)
    assert result["equality"].shape == (0,)
    eff = [effect(it[0], code=2)]
    result = lifetimes(it, eff, [seed(initial=False)])
    assert result["census"]["hits"]["n_objects"] == 0
    assert result["crossings"]["partner_hit_rows"].shape == (0,)


@pytest.mark.parametrize("stages", [((3, 1), (4, 1)), ((4, 0), (6, 0)), ((4, 0), (3, 0)), ((4, 0), (4, 0))])
def test_unknown_extension_transitions_rejected(stages):
    with pytest.raises(ValueError, match="unknown adjacent|initial iteration outside"):
        T.iteration_links(ak.Array(iterations(stages)))


def test_ragged_seed_mask_and_shape_fail_closed():
    it, eff, seeds = lifetime_fixture()
    seeds[0]["sh_shindex"] = [None]
    with pytest.raises(ValueError, match="missing values"):
        lifetimes(it, eff, seeds)
    seeds[0]["sh_shindex"] = [100, 101]
    with pytest.raises(ValueError, match="lengths differ"):
        lifetimes(it, eff, seeds)


def test_owner_matrix_shape_is_not_broadcast():
    it = iterations()
    ps = pieces(it)
    for row in ps:
        row["bnom"] = [[0., 0., .001]]
    with pytest.raises(ValueError, match="shape"):
        owners(it, ps)


def test_permanent_disappearance_is_not_terminal_object_ending():
    it, eff, seeds = lifetime_fixture()
    eff = [row for row in eff if not (row["identity_kind"] == 1 and row["input_index"] == 0 and row["iteration_ordinal"] >= 5)]
    result = lifetimes(it, eff, seeds)
    assert result["census"]["hits"]["n_disappearances"] == 1
    assert result["census"]["hits"]["n_reappearances"] == 0
    assert result["census"]["hits"]["disappearance_keys"] == [[1, 2, 3, 4, 4, 0]]
