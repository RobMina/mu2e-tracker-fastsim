"""Paired geometric envelopes are neither resolved crossings nor transport admission."""
from copy import deepcopy
from fractions import Fraction
from itertools import product

import numpy as np
import pandas as pd
import pytest

from trkfastsim import omittedpath, supportenvelope, supportnavigation


SOURCE = "p[0]/ring[0]"
HIT_COLS = ["lane", "s0_mm", "s1_mm", "material", "sign"]


def hits(*rows):
    return pd.DataFrame(rows, columns=HIT_COLS).astype(
        dict(lane="int64", s0_mm="float64", s1_mm="float64", material="object", sign="int64"))


def cell(lane=0, start=2., stop=4., **changes):
    row = dict(lane=lane, s0_mm=start, s1_mm=stop, material="Al", state="inside",
               ancestors_verified=True, leaf_verified=True, replaces_material="Honeycomb",
               source=SOURCE, scope="support") | changes
    row.setdefault("reason", "certificate" if row["state"] == "inside" else "min_width")
    return row


def cells(*rows):
    return pd.DataFrame(rows, columns=supportnavigation.CELL_COLS + ["reason"]).astype(
        dict(lane="int64", s0_mm="float64", s1_mm="float64"))


def record(source=SOURCE, **changes):
    return dict(placement_path=source, classification="unsupported", material="Al", solid_tag="torus",
                scope="support", is_leaf=True, placement_assumptions=[]) | changes


def reports_for(spans, records, lengths, parent="Honeycomb"):
    """Synthetic trusted evidence includes the torus probe's strict reason/cap metadata."""
    reports = []
    for rec in records:
        if rec["classification"] != "unsupported":
            continue
        subset = spans[spans.source == rec["placement_path"]]
        inside = subset.state == "inside"
        unknown = subset[~inside]
        known = inside & subset.ancestors_verified.astype(bool) & subset.leaf_verified.astype(bool)
        method = dict(inside_cells=int(inside.sum()), unresolved_cells=len(unknown))
        if rec["solid_tag"] == "torus":
            method.update(cap_reached=False, lanes=len(lengths),
                          positive_length_lanes=int(np.count_nonzero(lengths)),
                          unresolved_lanes=int(unknown.lane.nunique()),
                          unresolved_reasons={str(k): int(v) for k, v in unknown.reason.value_counts().items()})
        else:
            method["capped"] = False
        reports.append(dict(source=rec["placement_path"], material=rec["material"],
                            solid_tag=rec["solid_tag"], scope=rec["scope"], leaf=rec["is_leaf"],
                            resolved=True, parent_material=(parent[rec["placement_path"]]
                                                           if isinstance(parent, dict) else parent),
                            solid_lower_mm=omittedpath.length_bound(subset[inside], upper=False),
                            solid_upper_mm=omittedpath.length_bound(subset, upper=True),
                            hierarchy_verified_lower_mm=omittedpath.length_bound(subset[known], upper=False),
                            hit_lanes=int(subset.loc[inside, "lane"].nunique()),
                            possible_lanes=int(subset.lane.nunique()), method=method))
    return reports


def compose(raw, lengths, spans=None, *, records=None, reports=None, parent="Honeycomb", **kwargs):
    spans = cells() if spans is None else spans
    records = [record()] if records is None else records
    reports = reports_for(spans, records, lengths, parent) if reports is None else reports
    options = dict(endpoint_budget_mm=0., min_cell_mm=10., max_component_mm=100.,
                   max_ray_unknown_mm=100., max_pairs=2000000) | kwargs
    return supportenvelope.compose(raw, lengths, spans, reports, records, **options)


def reasons(result, lane):
    return set(result["refusals"].loc[result["refusals"].lane == lane, "reason"])


def occupied(result, arm, lane, material):
    rows = result[arm + "_intervals"]
    rows = rows[(rows.lane == lane) & (rows.material == material)]
    return float((rows.s1_mm - rows.s0_mm).sum())


def bound(result, lane, material):
    rows = result["material_bounds"]
    row = rows[(rows.lane == lane) & (rows.material == material)].iloc[0]
    return row.lower_mm, row.upper_mm


def assert_bracket(pair, lower, upper=None):
    upper = lower if upper is None else upper
    assert pair[0] <= lower <= upper <= pair[1], (pair, lower, upper)
    assert pair[0] == pytest.approx(lower, rel=2e-12, abs=1e-12)
    assert pair[1] == pytest.approx(upper, rel=2e-12, abs=1e-12)


def assert_contract(result, lengths):
    decisions, summary = result["decisions"], result["summary"]
    assert decisions.lane.tolist() == list(range(len(lengths)))
    np.testing.assert_array_equal(decisions.requested_length_mm, lengths)
    assert not decisions.admitted_for_transport.any()
    assert summary["admitted_for_transport"] is False
    assert summary["mode"] == "support-navigation-torus-envelope-v1"
    assert summary["unresolved_policy"] == "paired-inner-outer-min-width-torus-bands"
    assert summary["rays"] == len(lengths)
    assert summary["bounded_pair_rays"] == int(decisions.bounded_pair_admitted.sum())
    assert summary["refused_rays"] == int((~decisions.bounded_pair_admitted).sum())
    assert result["control"]["summary"]["admitted_for_transport"] is False
    np.testing.assert_array_equal(decisions.strict_candidate_resolved,
                                  result["control"]["decisions"].candidate_resolved)
    for lane, length in enumerate(lengths):
        admitted = decisions.loc[lane, "bounded_pair_admitted"]
        assert bool(reasons(result, lane)) != bool(admitted)
        for arm in ("inner", "outer"):
            rows = result[arm + "_intervals"]
            rows = rows[rows.lane == lane]
            if not admitted or length == 0:
                assert rows.empty
            else:
                assert rows.s0_mm.iloc[0] == 0.
                assert rows.s1_mm.iloc[-1] == length
                assert (rows.s1_mm > rows.s0_mm).all()
                np.testing.assert_array_equal(rows.s1_mm.to_numpy()[:-1], rows.s0_mm.to_numpy()[1:])
        if not admitted:
            for key in ("atoms", "material_bounds", "nonvacuum_bounds", "replacement_terms"):
                assert result[key][result[key].lane == lane].empty
    atoms = result["atoms"]
    assert not ((atoms.inner_material != atoms.outer_material) & ~atoms.disputed_geometry).any()


@pytest.mark.parametrize("parent", ["Honeycomb", "vacuum", "Al"])
def test_pair_replaces_parent_retains_unknown_state_and_strict_control(parent):
    raw = hits() if parent == "vacuum" else hits((0, 0., 10., parent, 1))
    spans = cells(cell(replaces_material=parent),
                  cell(start=4., stop=5., state="unresolved", replaces_material=parent))
    result = compose(raw, [10.], spans, parent=parent)
    assert_contract(result, [10.])
    assert result["decisions"].bounded_pair_admitted.tolist() == [True]
    assert result["decisions"].strict_candidate_resolved.tolist() == [False]
    assert result["control"]["intervals"].empty
    assert reasons(result["control"], 0) == {"unresolved-solid-boundary"}
    assert result["unknown_cells"].state.tolist() == ["unresolved"]
    assert result["unknown_cells"].reason.tolist() == ["min_width"]
    atoms = result["atoms"]
    disputed = atoms[atoms.disputed_geometry]
    np.testing.assert_array_equal(disputed[["s0_mm", "s1_mm"]], [[4., 5.]])
    terms = result["replacement_terms"]
    assert set(terms.source) == {SOURCE}
    assert set(terms.replaces_material) == {parent}
    assert set(terms[terms.envelope == "inner"].state) == {"inside"}
    assert set(terms[terms.envelope == "outer"].state) == {"inside", "unresolved"}
    assert set(terms.sign) == ({1} if parent == "vacuum" else {-1, 1})
    assert set(terms.operation) == ({"add-leaf"} if parent == "vacuum" else {"add-leaf", "subtract-parent"})
    assert result["parent_comparisons"].expected_parent.all()
    if parent == "Al":
        assert occupied(result, "inner", 0, "Al") == occupied(result, "outer", 0, "Al") == 10.
        assert (disputed.inner_material == disputed.outer_material).all()
        assert_bracket(bound(result, 0, "Al"), 10.)
        assert sorted(terms[terms.envelope == "inner"].sign) == [-1, 1]
        assert sorted(terms[terms.envelope == "outer"].sign) == [-1, -1, 1, 1]
    else:
        assert occupied(result, "inner", 0, "Al") == 2.
        assert occupied(result, "outer", 0, "Al") == 3.
        assert_bracket(bound(result, 0, "Al"), 2., 3.)
        assert_bracket(bound(result, 0, parent), 7., 8.)
    aggregate = result["nonvacuum_bounds"].iloc[0]
    assert aggregate.quantity == "all-nonvacuum-path"
    assert_bracket((aggregate.lower_mm, aggregate.upper_mm), 2. if parent == "vacuum" else 10.,
                   3. if parent == "vacuum" else 10.)


def test_opposite_replacements_have_equal_arm_columns_but_wide_marginals_and_correlated_sum():
    records = [record(material="B"), record("p[0]/ring[1]", material="A")]
    parents = {SOURCE: "A", "p[0]/ring[1]": "B"}
    spans = cells(cell(start=1., stop=3., material="B", replaces_material="A", state="unresolved"),
                  cell(start=7., stop=9., material="A", replaces_material="B", state="unresolved",
                       source="p[0]/ring[1]"))
    result = compose(hits((0, 0., 5., "A", 1), (0, 5., 10., "B", 1)), [10.], spans,
                     records=records, parent=parents)
    assert_contract(result, [10.])
    for material in ("A", "B"):
        assert occupied(result, "inner", 0, material) == occupied(result, "outer", 0, material) == 5.
        assert_bracket(bound(result, 0, material), 3., 7.)
    # Each band can independently choose parent or leaf: whole-arm min/max (=5) is unsound.
    for first, second in product((False, True), repeat=2):
        path_a = 5. - 2. * first + 2. * second
        for material, truth in (("A", path_a), ("B", 10. - path_a)):
            lower, upper = bound(result, 0, material)
            assert lower <= truth <= upper
    aggregate = result["nonvacuum_bounds"].iloc[0]
    assert_bracket((aggregate.lower_mm, aggregate.upper_mm), 10.)
    assert result["material_bounds"].lower_mm.sum() < aggregate.lower_mm
    assert result["material_bounds"].upper_mm.sum() > aggregate.upper_mm


def test_connected_band_not_individual_rows_controls_component_threshold():
    spans = cells(cell(start=1., stop=1.5, state="unresolved"),
                  cell(start=1.5, stop=2., state="unresolved", source="p[0]/ring[1]"))
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10.], spans,
                     records=[record(), record("p[0]/ring[1]")], min_cell_mm=.5,
                     max_component_mm=.75)
    assert_contract(result, [10.])
    assert reasons(result, 0) == {"unknown-component-budget-exceeded"}
    component = result["unknown_components"].iloc[0]
    assert (component.s0_mm, component.s1_mm, component.cells, component.sources) == (1., 2., 2, 2)
    assert component.upper_mm > 1.
    assert not component.within_component_budget
    assert result["unknown_cells"].component.nunique() == 1


def test_per_ray_budget_sums_disjoint_union_components_without_pooling_lanes():
    spans = cells(cell(start=1., stop=1.5, state="unresolved"),
                  cell(start=3., stop=3.5, state="unresolved"),
                  cell(lane=1, start=1., stop=1.5, state="unresolved"))
    raw = hits((0, 0., 5., "Honeycomb", 1), (1, 0., 5., "Honeycomb", 1))
    result = compose(raw, [5., 5.], spans, min_cell_mm=.5, max_component_mm=.75, max_ray_unknown_mm=.75)
    assert_contract(result, [5., 5.])
    assert result["decisions"].bounded_pair_admitted.tolist() == [False, True]
    assert reasons(result, 0) == {"unknown-ray-budget-exceeded"}
    assert result["unknown_components"].within_component_budget.all()
    np.testing.assert_allclose(result["decisions"].unknown_upper_mm, [1., .5], rtol=1e-13)


def test_unknown_union_does_not_double_count_nested_overlapping_or_touching_rows():
    spans = cells(cell(start=1., stop=3., state="unresolved"),
                  cell(start=1.5, stop=2., state="unresolved"),
                  cell(start=2., stop=4., state="unresolved", source="p[0]/ring[1]"),
                  cell(start=4., stop=5., state="unresolved"),
                  cell(start=6., stop=7., state="unresolved"))
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10.], spans,
                     records=[record(), record("p[0]/ring[1]")])
    assert_contract(result, [10.])
    assert reasons(result, 0) == {"overlapping-possible-leaf-cells"}
    np.testing.assert_array_equal(result["unknown_components"][["s0_mm", "s1_mm"]], [[1., 5.], [6., 7.]])
    assert result["decisions"].unknown_lower_mm.iloc[0] <= 5. <= result["decisions"].unknown_upper_mm.iloc[0]
    assert result["decisions"].unknown_upper_mm.iloc[0] < (spans.s1_mm - spans.s0_mm).sum()
    assert len(result["unknown_cells"]) == len(spans)


@pytest.mark.parametrize("budget,field,reason", [
    ("max_component_mm", "max_component_upper_mm", "unknown-component-budget-exceeded"),
    ("max_ray_unknown_mm", "max_ray_unknown_upper_mm", "unknown-ray-budget-exceeded"),
])
def test_budget_equality_and_adjacent_floats_use_outward_upper_not_nominal_width(budget, field, reason):
    raw = hits((0, 0., 10., "Honeycomb", 1))
    spans = cells(cell(start=2., stop=3., state="unresolved"))
    baseline = compose(raw, [10.], spans)
    threshold = baseline["summary"][field]
    assert threshold > 1.
    for value, admitted in ((1., False), (np.nextafter(threshold, -np.inf), False),
                            (threshold, True), (np.nextafter(threshold, np.inf), True)):
        result = compose(raw, [10.], spans, **{budget: value})
        assert_contract(result, [10.])
        assert bool(result["decisions"].bounded_pair_admitted.iloc[0]) is admitted
        assert (reason in reasons(result, 0)) is not admitted


def test_zero_unknown_budget_allows_certified_cells_but_not_a_one_ulp_unknown():
    raw = hits((0, 0., 10., "Honeycomb", 1))
    known = compose(raw, [10.], cells(cell()), max_component_mm=0., max_ray_unknown_mm=0.)
    assert_contract(known, [10.])
    pd.testing.assert_frame_equal(known["inner_intervals"], known["outer_intervals"])
    assert known["decisions"].bounded_pair_admitted.all()
    start, stop = 4., np.nextafter(4., np.inf)
    spans = cells(cell(start=start, stop=stop, state="unresolved", reason="roundoff"))
    unknown = compose(raw, [10.], spans, endpoint_budget_mm=1e-6,
                      max_component_mm=0., max_ray_unknown_mm=0.)
    assert_contract(unknown, [10.])
    assert reasons(unknown, 0) == {"unknown-component-budget-exceeded", "unknown-ray-budget-exceeded", "ineligible-torus-boundary-reason"}
    np.testing.assert_array_equal(unknown["unknown_cells"][["s0_mm", "s1_mm"]], [[start, stop]])


@pytest.mark.parametrize("reason", ["roundoff", "max_nodes", "max_active", "max_rounds", "certificate"])
def test_non_min_width_unknown_reasons_are_not_budget_eligible(reason):
    spans = cells(cell(state="unresolved", reason=reason))
    reports = reports_for(spans, [record()], [10., 0.])
    capped = reason.startswith("max_")
    reports[0]["method"]["cap_reached"] = capped
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10., 0.], spans, reports=reports)
    assert_contract(result, [10., 0.])
    assert "ineligible-torus-boundary-reason" in reasons(result, 0)
    assert result["decisions"].bounded_pair_admitted.tolist() == [False, not capped]


@pytest.mark.parametrize("case", ["missing-reason", "unknown-reason", "null-reason", "inside-min-width",
                                   "oversize-min-width"])
def test_malformed_or_contradictory_cell_reasons_raise(case):
    spans = cells(cell(state="unresolved"))
    if case == "unknown-reason":
        spans["reason"] = "guessed-boundary"
    elif case == "null-reason":
        spans["reason"] = None
    elif case == "inside-min-width":
        spans["state"] = "inside"
    reports = reports_for(spans, [record()], [10.])
    if case == "missing-reason":
        spans = spans.drop(columns="reason")
    with pytest.raises(ValueError, match="reason|declared probe width"):
        compose(hits(), [10.], spans, reports=reports, min_cell_mm=1. if case == "oversize-min-width" else 10.)


@pytest.mark.parametrize("key", ["lanes", "positive_length_lanes", "inside_cells", "unresolved_cells",
                                 "unresolved_lanes", "unresolved_reasons"])
@pytest.mark.parametrize("change", ["missing", "wrong"])
def test_torus_method_counts_and_reason_counts_are_required_and_exact(key, change):
    spans = cells(cell(), cell(lane=1, state="unresolved"))
    reports = reports_for(spans, [record()], [10., 10., 0.])
    if change == "missing":
        del reports[0]["method"][key]
    else:
        reports[0]["method"][key] = {"roundoff": 1} if key == "unresolved_reasons" else 99
    with pytest.raises(ValueError, match="counts disagree|reason/count evidence"):
        compose(hits(), [10., 10., 0.], spans, reports=reports)


@pytest.mark.parametrize("tag,key", [("torus", "cap_reached"), ("xtru", "capped")])
@pytest.mark.parametrize("value", [None, 0, 1, "false", np.bool_(False)])
def test_cap_metadata_requires_explicit_python_boolean(tag, key, value):
    records = [record(solid_tag=tag)]
    reports = reports_for(cells(), records, [10.])
    reports[0]["method"][key] = value
    with pytest.raises(ValueError, match="explicit boolean"):
        compose(hits(), [10.], records=records, reports=reports)


@pytest.mark.parametrize("tag,key", [("torus", "cap_reached"), ("xtru", "capped")])
def test_missing_cap_metadata_raises_even_for_excluded_placement(tag, key):
    records = [record(solid_tag=tag)]
    reports = reports_for(cells(), records, [10.])
    del reports[0]["method"][key]
    with pytest.raises(ValueError, match="explicit boolean"):
        compose(hits(), [10.], records=records, reports=reports)


@pytest.mark.parametrize("with_unknown", [False, True])
@pytest.mark.parametrize("tag,key", [("torus", "cap_reached"), ("xtru", "capped")])
def test_capped_placement_refuses_every_lane_including_unaffected_and_zero_length(tag, key, with_unknown):
    records = [record(solid_tag=tag)]
    spans = cells(cell(state="unresolved")) if with_unknown else cells()
    reports = reports_for(spans, records, [10., 4., 0.])
    reports[0]["method"][key] = True
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10., 4., 0.], spans,
                     records=records, reports=reports)
    assert_contract(result, [10., 4., 0.])
    assert not result["decisions"].bounded_pair_admitted.any()
    assert result["summary"]["all_pairs_admitted"] is False
    for lane in range(3):
        assert "capped-placement-probe" in reasons(result, lane)
    if with_unknown:
        assert result["control"]["decisions"].candidate_resolved.tolist() == [False, True, True]


def test_unknown_xtru_is_refused_even_with_min_width_reason_and_generous_budget():
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10., 4.],
                     cells(cell(state="unresolved")), records=[record(solid_tag="xtru")])
    assert_contract(result, [10., 4.])
    assert reasons(result, 0) == {"non-torus-unresolved-boundary"}
    assert result["decisions"].bounded_pair_admitted.tolist() == [False, True]


@pytest.mark.parametrize("flag", ["ancestors_verified", "leaf_verified"])
def test_unknown_band_requires_verified_ancestry_and_leaf(flag):
    spans = cells(cell(state="unresolved", **{flag: False}))
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10.], spans,
                     records=[record(is_leaf=flag != "leaf_verified")])
    assert_contract(result, [10.])
    assert ("unverified-ancestor" if flag == "ancestors_verified" else "unverified-leaf") in reasons(result, 0)


def test_parent_mismatch_only_in_unknown_band_is_not_hidden_by_certified_prefix():
    raw = hits((0, 0., 4.75, "Honeycomb", 1), (0, 4.75, 5., "Steel", 1),
               (0, 5., 10., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1))
    spans = cells(cell(start=1., stop=2.), cell(start=4., stop=5., state="unresolved"), cell(lane=1))
    result = compose(raw, [10., 10.], spans)
    assert_contract(result, [10., 10.])
    assert reasons(result["control"], 0) == {"unresolved-solid-boundary"}
    assert "possible-cell-parent-mismatch" in reasons(result, 0)
    assert "possible-cell-ambiguous-ownership" in reasons(result, 0)
    assert result["decisions"].bounded_pair_admitted.tolist() == [False, True]
    pairs = result["parent_comparisons"]
    assert pairs.loc[~pairs.expected_parent, "state"].tolist() == ["unresolved"]
    assert result["control"]["replacement_terms"].lane.eq(0).any()
    assert not result["replacement_terms"].lane.eq(0).any()


@pytest.mark.parametrize("extra_material", ["Honeycomb", "Steel"])
@pytest.mark.parametrize("start,stop,refused", [(4., 5., True), (7., 8., False), (4.5, 4.5, False)])
def test_raw_cancellation_ownership_is_checked_in_possible_band_not_only_certified_cells(extra_material, start, stop, refused):
    raw = hits((0, 0., 10., "Honeycomb", 1), (0, start, stop, extra_material, 1),
               (0, start, stop, extra_material, -1))
    spans = cells(cell(start=1., stop=2.), cell(start=4., stop=5., state="unresolved"))
    result = compose(raw, [10.], spans)
    assert_contract(result, [10.])
    assert reasons(result["control"], 0) == {"unresolved-solid-boundary"}
    assert bool(result["decisions"].bounded_pair_admitted.iloc[0]) is not refused
    assert ("possible-cell-ambiguous-ownership" in reasons(result, 0)) is refused
    assert result["parent_comparisons"].expected_parent.all()
    if refused:
        assert set(result["ownership_comparisons"].loc[result["ownership_comparisons"].sign == -1, "state"]) == {"unresolved"}


@pytest.mark.parametrize("same_material", [False, True])
@pytest.mark.parametrize("duplicate_source", [False, True])
@pytest.mark.parametrize("second_state", ["inside", "unresolved"])
def test_possible_overlap_refused_independently_of_material_or_source(same_material, duplicate_source, second_state):
    parent = "Al" if same_material else "Honeycomb"
    source = SOURCE if duplicate_source else "p[0]/ring[1]"
    records = [record()] if duplicate_source else [record(), record(source)]
    spans = cells(cell(start=2., stop=5., replaces_material=parent, state="unresolved"),
                  cell(start=4., stop=6., replaces_material=parent, source=source, state=second_state))
    result = compose(hits((0, 0., 10., parent, 1)), [10.], spans, records=records, parent=parent)
    assert_contract(result, [10.])
    assert reasons(result, 0) == {"overlapping-possible-leaf-cells"}


@pytest.mark.parametrize("lengths", [[], [0.], [0., 3., 0.]])
def test_empty_and_zero_length_requests_have_explicit_decisions(lengths):
    result = compose(hits(), lengths)
    assert_contract(result, lengths)
    assert result["summary"]["all_pairs_admitted"] is True
    assert result["unknown_cells"].empty and result["unknown_components"].empty
    assert result["replacement_terms"].empty
    assert result["decisions"].unknown_upper_mm.eq(0.).all()
    if len(lengths) == 3:
        assert occupied(result, "inner", 1, "vacuum") == 3.
        assert occupied(result, "outer", 1, "vacuum") == 3.
        assert_bracket(bound(result, 1, "vacuum"), 3.)


@pytest.mark.parametrize("lengths", [[], [0., 5.]])
def test_unevaluated_global_refusal_is_not_empty_success(lengths):
    reports = reports_for(cells(), [record()], lengths)
    reports[0]["resolved"] = False
    result = compose(hits(), lengths, reports=reports)
    assert_contract(result, lengths)
    assert result["summary"]["all_pairs_admitted"] is False
    assert not result["decisions"].bounded_pair_admitted.any()
    for key in ("inner_intervals", "outer_intervals", "atoms", "material_bounds", "nonvacuum_bounds"):
        assert result[key].empty


@pytest.mark.parametrize("key", ["min_cell_mm", "max_component_mm", "max_ray_unknown_mm"])
@pytest.mark.parametrize("value", [-1., np.nan, np.inf, True, [1.]])
def test_resolution_budgets_must_be_finite_real_nonnegative_scalars(key, value):
    with pytest.raises(ValueError):
        compose(hits(), [1.], **{key: value})


def test_min_cell_width_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        compose(hits(), [1.], min_cell_mm=0.)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, np.inf])
def test_pair_limit_is_positive_integer(value):
    with pytest.raises(ValueError, match="max_pairs must be a positive integer"):
        compose(hits(), [10.], max_pairs=value)


def test_comparison_bound_includes_unknown_cells_and_never_truncates():
    raw = hits((0, 0., 5., "Honeycomb", 1), (0, 5., 10., "Honeycomb", 1))
    spans = cells(cell(start=1., stop=2., state="unresolved"), cell(start=6., stop=7., state="unresolved"))
    with pytest.raises(ValueError, match="possible-cell comparison exceeds explicit pair bound"):
        compose(raw, [10.], spans, max_pairs=3)
    result = compose(raw, [10.], spans, max_pairs=4)
    assert_contract(result, [10.])
    assert result["decisions"].bounded_pair_admitted.all()


def canonical(frame):
    return frame.sort_values(list(frame.columns), kind="stable").reset_index(drop=True)


def test_no_input_mutation_and_source_cell_report_permutations_preserve_results():
    lengths = np.array([10., 10., 0.])
    raw = hits((0, 0., 5., "Honeycomb", 1), (0, 5. + 1e-10, 10., "Honeycomb", 1),
               (1, 0., 10., "Honeycomb", 1))
    raw["particle_id"] = [91, 91, 91]
    raw["source"] = ["left", "right", "other-ray"]
    raw.index = [8, 8, 2]
    spans = cells(cell(start=1., stop=2.), cell(start=4., stop=6., state="unresolved"),
                  cell(lane=1, start=6., stop=7., state="unresolved", source="p[0]/ring[1]"))
    records = [record(), record("p[0]/ring[1]")]
    reports = reports_for(spans, records, lengths)
    before = raw.copy(deep=True), spans.copy(deep=True), lengths.copy(), deepcopy(records), deepcopy(reports)
    strict = compose(raw, lengths, spans, records=records, reports=reports)
    assert "possible-cell-parent-mismatch" in reasons(strict, 0)
    baseline = compose(raw, lengths, spans, records=records, reports=reports, endpoint_budget_mm=1e-9)
    assert_contract(baseline, lengths)
    assert baseline["decisions"].bounded_pair_admitted.all()
    assert baseline["control"]["summary"]["endpoint_policy"]["changed_endpoints"] == 2
    for seed in (3, 11, 29):
        result = compose(raw.sample(frac=1, random_state=seed), lengths,
                         spans.sample(frac=1, random_state=seed), records=records[::-1], reports=reports[::-1],
                         endpoint_budget_mm=1e-9)
        for key in ("inner_intervals", "outer_intervals", "atoms", "material_bounds", "nonvacuum_bounds",
                    "decisions", "refusals", "unknown_components"):
            pd.testing.assert_frame_equal(result[key], baseline[key])
        for key in ("replacement_terms", "unknown_cells"):
            pd.testing.assert_frame_equal(canonical(result[key]), canonical(baseline[key]))
        assert result["summary"] == baseline["summary"]
    pd.testing.assert_frame_equal(raw, before[0])
    pd.testing.assert_frame_equal(spans, before[1])
    np.testing.assert_array_equal(lengths, before[2])
    assert records == before[3] and reports == before[4]
    baseline["unknown_cells"].loc[:, "s0_mm"] = 0.
    baseline["control"]["adjusted_hits"].loc[:, "s0_mm"] = 0.
    pd.testing.assert_frame_equal(raw, before[0])
    pd.testing.assert_frame_equal(spans, before[1])


def test_control_is_exactly_the_unchanged_strict_compositor_result():
    raw = hits((0, 0., 10., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1))
    lengths = np.array([10., 10., 0.])
    spans = cells(cell(), cell(start=7., stop=8., state="unresolved"), cell(lane=1))
    records = [record()]
    reports = reports_for(spans, records, lengths)
    strict = supportnavigation.compose(raw, lengths, spans, reports, records, endpoint_budget_mm=0.)
    paired = compose(raw, lengths, spans, records=records, reports=reports)
    assert set(strict) == set(paired["control"])
    for key, value in strict.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(paired["control"][key], value)
        else:
            assert paired["control"][key] == value
    assert paired["decisions"].bounded_pair_admitted.tolist() == [True, True, True]
    assert paired["decisions"].strict_candidate_resolved.tolist() == [False, True, True]


def test_synthetic_lane_permutation_is_equivariant_including_refused_and_zero_length_lanes():
    lengths = np.array([10., 8., 0., 7.])
    raw = hits((0, 0., 10., "Honeycomb", 1), (1, 0., 8., "Honeycomb", 1),
               (3, 0., 7., "Steel", 1))
    spans = cells(cell(), cell(start=7., stop=8., state="unresolved"),
                  cell(lane=1, state="unresolved"), cell(lane=3, state="unresolved"))
    baseline = compose(raw, lengths, spans)
    permutation = np.array([3, 0, 2, 1])
    inverse = np.argsort(permutation)
    permuted_raw = raw.assign(lane=inverse[raw.lane.to_numpy()])
    permuted_cells = spans.assign(lane=inverse[spans.lane.to_numpy()])
    result = compose(permuted_raw, lengths[permutation], permuted_cells)
    assert_contract(result, lengths[permutation])
    assert baseline["decisions"].bounded_pair_admitted.tolist() == [True, True, True, False]
    for key in ("inner_intervals", "outer_intervals", "atoms", "material_bounds", "nonvacuum_bounds",
                "decisions", "refusals", "replacement_terms"):
        restored = result[key].assign(lane=permutation[result[key].lane.to_numpy(dtype=int)])
        pd.testing.assert_frame_equal(canonical(restored), canonical(baseline[key]))
    assert result["summary"] == baseline["summary"]


@pytest.mark.parametrize("bad_rows", [
    [(0, 1., 3., "Honeycomb", -1)],
    [(0, 0., 10., "Honeycomb", 1), (0, 1., 3., "Honeycomb", 1)],
    [(0, 0., 10., "Honeycomb", 1), (0, 1., 3., "Steel", 1)],
])
def test_invalid_stored_occupancy_removes_both_arms_without_destroying_other_lanes(bad_rows):
    raw = hits(*bad_rows, (1, 0., 10., "Honeycomb", 1))
    result = compose(raw, [10., 10.], cells(cell(state="unresolved"), cell(lane=1, state="unresolved")))
    assert_contract(result, [10., 10.])
    assert "invalid-stored-occupancy" in reasons(result, 0)
    assert result["decisions"].bounded_pair_admitted.tolist() == [False, True]
    assert occupied(result, "inner", 1, "Al") == 0.
    assert occupied(result, "outer", 1, "Al") == 2.


def test_declared_min_cell_width_equality_is_not_a_geometric_budget_waiver():
    raw = hits((0, 0., 10., "Honeycomb", 1))
    spans = cells(cell(start=2., stop=3., state="unresolved"))
    result = compose(raw, [10.], spans, min_cell_mm=1.)
    assert_contract(result, [10.])
    assert result["decisions"].bounded_pair_admitted.all()
    with pytest.raises(ValueError, match="min_width cell exceeds the declared probe width"):
        compose(raw, [10.], spans, min_cell_mm=np.nextafter(1., 0.))


def test_grouped_bounds_enclose_exact_binary64_endpoint_difference_sums():
    # Fraction is an independent exact-arithmetic oracle, including cancellation and subnormals.
    spans = pd.DataFrame([
        (0, 0., .1), (0, .1, .3), (0, 1., np.nextafter(1., np.inf)),
        (0, 1e16, np.nextafter(1e16, np.inf)), (0, 0., np.nextafter(0., np.inf)),
        (1, 2., 2.), (1, 3., 3.), (2, -.1, .3),
    ], columns=["lane", "s0_mm", "s1_mm"])
    result = supportenvelope.grouped_bounds(spans, ["lane"])
    for row in result.itertuples(index=False):
        subset = spans[spans.lane == row.lane]
        exact = sum((Fraction.from_float(stop) - Fraction.from_float(start)
                     for start, stop in subset[["s0_mm", "s1_mm"]].itertuples(index=False, name=None)), Fraction())
        assert Fraction.from_float(row.lower_mm) <= exact <= Fraction.from_float(row.upper_mm)
        assert row.lower_mm >= 0.
    zero = result[result.lane == 1].iloc[0]
    assert zero.lower_mm == zero.upper_mm == 0.
    assert supportenvelope.grouped_bounds(spans.iloc[:0], ["lane"]).empty


@pytest.fixture
def torus_geometry(tmp_path):
    def make(*, parent="Honeycomb", start=0., delta=360., parent_size=60.):
        path = tmp_path / "envelope_torus.gdml"
        path.write_text(f'''<gdml><solids><box name="world" x="100" y="100" z="100"/>
        <box name="parent" x="{parent_size}" y="{parent_size}" z="{parent_size}"/>
        <torus name="ring" rtor="10" rmin="0" rmax="2" startphi="{start}"
               deltaphi="{delta}" aunit="deg" lunit="mm"/></solids>
        <structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="world"/>
        <physvol name="p"><volumeref ref="Parent"/></physvol></volume>
        <volume name="Parent"><materialref ref="{parent}"/><solidref ref="parent"/>
        <physvol name="ring"><volumeref ref="Leaf"/></physvol></volume>
        <volume name="Leaf"><materialref ref="Al"/><solidref ref="ring"/></volume></structure>
        <setup><world ref="World"/></setup></gdml>''')
        return path, [record()]
    return make


def probe_and_compose(gdml, records, origins, lengths, *, directions=None, parent="Honeycomb",
                      min_cell_mm=1e-4, max_cells=100000, **kwargs):
    origins = np.asarray(origins, float).reshape(-1, 3)
    lengths = np.asarray(lengths, float)
    directions = np.tile([0., 0., 1.], (len(origins), 1)) if directions is None else np.asarray(directions, float)
    spans, reports = omittedpath.probe(gdml, records, origins, directions, lengths,
                                       min_cell_mm=min_cell_mm, max_cells=max_cells)
    raw = hits() if parent == "vacuum" else hits(*[(lane, 0., length, parent, 1)
                                                  for lane, length in enumerate(lengths) if length > 0])
    result = compose(raw, lengths, spans, records=records, reports=reports, min_cell_mm=min_cell_mm, **kwargs)
    assert_contract(result, lengths)
    return result, spans, reports


def assert_analytic(result, spans, lane, truth):
    subset = spans[spans.lane == lane]
    lower = omittedpath.length_bound(subset[subset.state == "inside"], upper=False)
    upper = omittedpath.length_bound(subset, upper=True)
    assert lower <= truth <= upper, (lane, lower, truth, upper)
    if result["decisions"].loc[lane, "bounded_pair_admitted"]:
        rows = result["material_bounds"]
        material = rows[(rows.lane == lane) & (rows.material == "Al")]
        if material.empty:
            assert truth == 0.
        else:
            assert material.lower_mm.iloc[0] <= truth <= material.upper_mm.iloc[0]
        assert occupied(result, "inner", lane, "Al") <= truth + 1e-12
        assert occupied(result, "outer", lane, "Al") >= truth - 1e-12
    if (subset.state == "unresolved").any():
        assert "unresolved-solid-boundary" in reasons(result["control"], lane)
        assert not result["decisions"].loc[lane, "strict_candidate_resolved"]


@pytest.mark.parametrize("parent", ["Honeycomb", "vacuum", "Al"])
def test_actual_torus_certified_inside_hole_and_analytic_crossing_chord(torus_geometry, parent):
    gdml, records = torus_geometry(parent="Vacuum" if parent == "vacuum" else parent)
    result, spans, reports = probe_and_compose(gdml, records,
        [[10., 0., -.5], [0., 0., -5.], [10., 0., -5.]], [1., 10., 10.], parent=parent,
        min_cell_mm=1e-5, max_component_mm=1e-3, max_ray_unknown_mm=1e-3)
    assert result["decisions"].bounded_pair_admitted.tolist() == [True, True, True]
    assert result["decisions"].strict_candidate_resolved.tolist() == [True, True, False]
    assert reports[0]["resolved"] is True and reports[0]["method"]["cap_reached"] is False
    assert set(spans.reason) == {"certificate", "min_width"}
    for lane, truth in enumerate((1., 0., 4.)):
        if parent != "Al":
            assert_analytic(result, spans, lane, truth)
        else:
            assert_bracket(bound(result, lane, "Al"), [1., 10., 10.][lane])
    assert result["atoms"].disputed_geometry.any()
    assert result["summary"]["all_pairs_admitted"] is True


def test_actual_torus_radial_two_chords_and_tangent_plus_minus_ulp_brackets(torus_geometry):
    gdml, records = torus_geometry()
    x = np.array([12. - 1e-5, np.nextafter(12., -np.inf), 12.,
                  np.nextafter(12., np.inf), 12. + 1e-5])
    origins = np.vstack(([-15., 0., 0.], np.column_stack((x, np.zeros(5), np.full(5, -5.)))))
    directions = np.vstack(([1., 0., 0.], np.tile([0., 0., 1.], (5, 1))))
    result, spans, reports = probe_and_compose(gdml, records, origins, [30.] + [10.] * 5,
                                              directions=directions, min_cell_mm=1e-5,
                                              max_component_mm=.01, max_ray_unknown_mm=.1)
    assert reports[0]["method"]["cap_reached"] is False
    assert result["decisions"].bounded_pair_admitted.all()
    truth = np.r_[8., 2. * np.sqrt(np.maximum(0., (12. - x) * (x - 8.)))]
    for lane, chord in enumerate(truth):
        assert_analytic(result, spans, lane, chord)
    assert truth[1] > truth[2] > 0.
    assert not result["decisions"].strict_candidate_resolved.iloc[3]
    assert result["decisions"].strict_candidate_resolved.iloc[-1]


@pytest.mark.parametrize("edge", ["start", "end"])
@pytest.mark.parametrize("nonaxis", [False, True])
def test_actual_torus_sector_edge_perturbations_remain_brackets_or_global_cap_refusal(torus_geometry, edge, nonaxis):
    gdml, records = torus_geometry(start=45. if nonaxis else 0., delta=90.)
    boundary = (7. if edge == "start" else -7.) if nonaxis else 0.
    varied = np.array([boundary - 1e-5, np.nextafter(boundary, -np.inf), boundary,
                       np.nextafter(boundary, np.inf), boundary + 1e-5])
    fixed = np.full(5, 7. if nonaxis else 10.)
    x, y = (fixed, varied) if edge == "start" else (varied, fixed)
    origins = np.column_stack((x, y, np.full(5, -.5)))
    result, spans, reports = probe_and_compose(gdml, records, origins, [1.] * 5,
                                              min_cell_mm=1e-3, max_cells=100000,
                                              max_component_mm=2., max_ray_unknown_mm=2.)
    for lane, value in enumerate(varied):
        assert_analytic(result, spans, lane, float(value >= boundary))
    assert reports[0]["method"]["cap_reached"] is False
    assert result["decisions"].bounded_pair_admitted.all()
    assert result["decisions"].strict_candidate_resolved.tolist() == [True, False, False, False, True]
    assert occupied(result, "outer", 0, "Al") == 0.
    assert occupied(result, "inner", 4, "Al") == 1.
    # The uncertain edge occupies an entire ray, not just one tiny min-width cell.
    central = result["unknown_components"][result["unknown_components"].lane == 2]
    assert central.s0_mm.iloc[0] == 0. and central.s1_mm.iloc[0] == 1.
    assert central.cells.iloc[0] > 1


def test_actual_torus_node_cap_preserves_analytic_evidence_but_refuses_all_pairs(torus_geometry):
    gdml, records = torus_geometry()
    result, spans, reports = probe_and_compose(gdml, records,
        [[-15., 0., 0.], [0., 0., -5.], [0., 0., 0.]], [30., 10., 0.],
        directions=[[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]], max_cells=1)
    assert reports[0]["method"]["cap_reached"] is True
    assert reports[0]["method"]["evaluated_nodes"] <= 1
    assert not result["decisions"].bounded_pair_admitted.any()
    for lane, truth in enumerate((8., 0., 0.)):
        assert_analytic(result, spans, lane, truth)
        assert "capped-placement-probe" in reasons(result, lane)
    assert not result["unknown_cells"].empty


@pytest.mark.parametrize("lengths,origins", [([], []), ([0.], [[10., 0., 0.]])])
def test_actual_torus_empty_and_zero_requests(torus_geometry, lengths, origins):
    gdml, records = torus_geometry()
    result, spans, reports = probe_and_compose(gdml, records, origins, lengths)
    assert result["summary"]["all_pairs_admitted"] is True
    assert spans.empty and result["atoms"].empty
    assert reports[0]["solid_upper_mm"] == 0.


def test_actual_sector_whole_connected_band_exceeds_budget_despite_small_cells(torus_geometry):
    gdml, records = torus_geometry(delta=90.)
    result, spans, reports = probe_and_compose(gdml, records,
        [[10., 0., -.5], [10., 1e-5, -.5]], [1., 1.], min_cell_mm=1e-3,
        max_component_mm=.01, max_ray_unknown_mm=2.)
    assert reports[0]["method"]["cap_reached"] is False
    unknown = spans[spans.state == "unresolved"]
    assert ((unknown.s1_mm - unknown.s0_mm) <= 1e-3).all()
    assert reasons(result, 0) == {"unknown-component-budget-exceeded"}
    assert result["decisions"].bounded_pair_admitted.tolist() == [False, True]
    assert_analytic(result, spans, 0, 1.)
    assert_analytic(result, spans, 1, 1.)


def test_actual_torus_finer_probe_narrows_crossing_bounds_without_resolving_control(torus_geometry):
    gdml, records = torus_geometry()
    widths = []
    for resolution in (1e-3, 1e-5):
        result, spans, reports = probe_and_compose(gdml, records, [[10., 0., -5.]], [10.],
                                                  min_cell_mm=resolution)
        assert reports[0]["method"]["cap_reached"] is False
        assert result["decisions"].bounded_pair_admitted.tolist() == [True]
        assert result["decisions"].strict_candidate_resolved.tolist() == [False]
        assert_analytic(result, spans, 0, 4.)
        low, high = bound(result, 0, "Al")
        widths.append(high - low)
    assert 0. < widths[1] < widths[0]


def test_actual_torus_unverified_ancestor_cannot_be_waived_by_boundary_budget(torus_geometry):
    gdml, records = torus_geometry(parent_size=10.)
    result, spans, reports = probe_and_compose(gdml, records, [[10., 0., -5.]], [10.])
    assert reports[0]["unresolved_ancestors"]
    assert not spans.ancestors_verified.any()
    assert "unverified-ancestor" in reasons(result, 0)
    assert not result["decisions"].bounded_pair_admitted.any()
    assert_analytic(result, spans, 0, 4.)


@pytest.mark.parametrize("reason", ["max_nodes", "max_active", "max_rounds"])
def test_cap_reasons_cannot_admit_other_lane_under_false_cap_flag(reason):
    spans = cells(cell(state="unresolved", reason=reason), cell(lane=1, state="unresolved"))
    with pytest.raises(ValueError, match="cap reason contradicts"):
        compose(hits((0, 0., 10., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1)), [10., 10.], spans)


@pytest.mark.parametrize("tag,alias", [("torus", "capped"), ("xtru", "cap_reached")])
@pytest.mark.parametrize("value", [True, "false", 0, None])
def test_conflicting_cap_aliases_are_not_ignored(tag, alias, value):
    records = [record(solid_tag=tag)]
    reports = reports_for(cells(), records, [10.])
    reports[0]["method"][alias] = value
    with pytest.raises(ValueError, match="conflicting cap metadata"):
        compose(hits(), [10.], records=records, reports=reports)


@pytest.mark.parametrize("start", [4., np.nextafter(4., np.inf)])
def test_stalled_bracket_mislabelled_min_width_is_not_admitted(start):
    stop = np.nextafter(start, np.inf)
    spans = cells(cell(start=start, stop=stop, state="unresolved"))
    with pytest.raises(ValueError, match="stalled midpoint"):
        compose(hits((0, 0., 10., "Honeycomb", 1)), [10.], spans,
                min_cell_mm=1e-7, max_component_mm=1e-6, max_ray_unknown_mm=1e-5)
