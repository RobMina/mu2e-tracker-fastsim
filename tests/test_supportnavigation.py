"""Separate support-navigation candidate tests; no transport or GDML-truth admission."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from trkfastsim import omittedpath, supportnavigation


HIT_COLS = ["lane", "s0_mm", "s1_mm", "material", "sign"]


def hits(*rows):
    return pd.DataFrame(rows, columns=HIT_COLS).astype(
        dict(lane="int64", s0_mm="float64", s1_mm="float64", material="object", sign="int64"))


def cell(lane=0, start=2., stop=4., **changes):
    return dict(lane=lane, s0_mm=start, s1_mm=stop, material="Al", state="inside",
                ancestors_verified=True, leaf_verified=True, replaces_material="Honeycomb",
                source="p[0]/ring[0]", scope="support") | changes


def cells(*rows):
    return pd.DataFrame(rows, columns=supportnavigation.CELL_COLS).astype(
        dict(lane="int64", s0_mm="float64", s1_mm="float64"))


def record(source="p[0]/ring[0]", **changes):
    return dict(placement_path=source, classification="unsupported", material="Al", solid_tag="torus",
                scope="support", is_leaf=True, placement_assumptions=[]) | changes


def reports_for(spans, records, parent="Honeycomb"):
    reports = []
    for rec in records:
        if rec["classification"] != "unsupported":
            continue
        subset = spans[spans.source == rec["placement_path"]]
        inside = subset.state == "inside"
        known = inside & subset.ancestors_verified.astype(bool) & subset.leaf_verified.astype(bool)
        reports.append(dict(source=rec["placement_path"], material=rec["material"],
                            solid_tag=rec["solid_tag"], scope=rec["scope"], leaf=rec["is_leaf"],
                            resolved=True, parent_material=parent,
                            solid_lower_mm=omittedpath.length_bound(subset[inside], upper=False),
                            solid_upper_mm=omittedpath.length_bound(subset, upper=True),
                            hierarchy_verified_lower_mm=omittedpath.length_bound(subset[known], upper=False),
                            hit_lanes=int(subset.loc[inside, "lane"].nunique()),
                            possible_lanes=int(subset.lane.nunique()),
                            method=dict(inside_cells=int(inside.sum()), unresolved_cells=int((~inside).sum()))))
    return reports


def compose(raw, lengths, spans=None, *, records=None, reports=None, parent="Honeycomb", **kwargs):
    spans = cells() if spans is None else spans
    records = [record()] if records is None else records
    reports = reports_for(spans, records, parent) if reports is None else reports
    return supportnavigation.compose(raw, lengths, spans, reports, records,
                                     endpoint_budget_mm=kwargs.pop("endpoint_budget_mm", 0.), **kwargs)


def reasons(result, lane):
    return set(result["refusals"].loc[result["refusals"].lane == lane, "reason"])


def occupied(result, lane, material):
    rows = result["intervals"]
    subset = rows[(rows.lane == lane) & (rows.material == material)]
    return float((subset.s1_mm - subset.s0_mm).sum())


def assert_candidate_contract(result, lengths):
    decisions = result["decisions"]
    assert decisions.lane.tolist() == list(range(len(lengths)))
    np.testing.assert_array_equal(decisions.requested_length_mm, lengths)
    assert not decisions.admitted_for_transport.any()
    assert result["summary"]["admitted_for_transport"] is False
    assert result["summary"]["mode"] == "support-navigation-torus-replace-refuse-v1"
    assert result["summary"]["unresolved_policy"] == "refuse-whole-lane-on-any-unresolved-cell"
    assert result["summary"]["rays"] == len(lengths)
    assert result["summary"]["resolved_rays"] == int(decisions.candidate_resolved.sum())
    assert result["summary"]["refused_rays"] == int((~decisions.candidate_resolved).sum())
    assert result["summary"]["interval_rows"] == len(result["intervals"])
    for lane, length in enumerate(lengths):
        rows = result["intervals"][result["intervals"].lane == lane]
        if not decisions.loc[lane, "candidate_resolved"]:
            assert rows.empty
            assert reasons(result, lane)
        elif length == 0:
            assert rows.empty
        else:
            assert not reasons(result, lane)
            assert rows.s0_mm.iloc[0] == 0.
            assert rows.s1_mm.iloc[-1] == length
            assert (rows.s1_mm > rows.s0_mm).all()
            np.testing.assert_array_equal(rows.s1_mm.to_numpy()[:-1], rows.s0_mm.to_numpy()[1:])


@pytest.mark.parametrize("parent", ["Honeycomb", "vacuum"])
def test_signed_leaf_replaces_parent_without_double_counting(parent):
    raw = hits((0, 0., 10., parent, 1)) if parent != "vacuum" else hits()
    spans = cells(cell(replaces_material=parent))
    result = compose(raw, [10.], spans, parent=parent)
    assert_candidate_contract(result, [10.])
    assert result["decisions"].candidate_resolved.tolist() == [True]
    assert occupied(result, 0, "Al") == 2.
    assert occupied(result, 0, parent) == 8.
    terms = result["replacement_terms"]
    expected = {("Al", 1, "add-leaf")}
    if parent != "vacuum":
        expected.add((parent, -1, "subtract-parent"))
    assert set(terms[["material", "sign", "operation"]].itertuples(index=False, name=None)) == expected
    assert set(terms.source) == {"p[0]/ring[0]"}
    assert set(terms.replaces_material) == {parent}
    assert result["parent_comparisons"].expected_parent.all()
    assert result["summary"]["unexpected_parent_comparisons"] == 0
    assert result["summary"]["all_rays_resolved"] is True


@pytest.mark.parametrize("parent", ["Vacuum", "G4_Galactic"])
def test_vacuum_parent_report_is_normalized(parent):
    result = compose(hits(), [10.], cells(cell(replaces_material="vacuum")), parent=parent)
    assert_candidate_contract(result, [10.])
    assert result["replacement_terms"].sign.tolist() == [1]
    assert occupied(result, 0, "Al") == 2.


def test_same_material_parent_cancels_but_retains_both_signed_terms():
    result = compose(hits((0, 0., 10., "Al", 1)), [10.],
                     cells(cell(replaces_material="Al")), parent="Al")
    assert_candidate_contract(result, [10.])
    assert occupied(result, 0, "Al") == 10.
    assert sorted(result["replacement_terms"].sign) == [-1, 1]
    assert set(result["replacement_terms"].material) == {"Al"}


@pytest.mark.parametrize("state,ancestor,leaf,reason", [
    ("unresolved", True, True, "unresolved-solid-boundary"),
    ("inside", False, True, "unverified-ancestor"),
    ("inside", True, False, "unverified-leaf"),
])
def test_any_unresolved_evidence_refuses_whole_lane_not_only_affected_span(state, ancestor, leaf, reason):
    spans = cells(cell(start=1., stop=2.), cell(start=7., stop=8., state=state,
                  ancestors_verified=ancestor, leaf_verified=leaf, source="p[0]/ring[1]"))
    records = [record(), record("p[0]/ring[1]", is_leaf=leaf)]
    result = compose(hits((0, 0., 10., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1)),
                     [10., 10.], spans, records=records)
    assert reason in reasons(result, 0)
    assert not result["decisions"].candidate_resolved.iloc[0]
    assert_candidate_contract(result, [10., 10.])
    if leaf:
        assert result["decisions"].candidate_resolved.tolist() == [False, True]
        assert occupied(result, 1, "Honeycomb") == 10.
    else:
        assert "non-leaf-placement:p[0]/ring[1]" in reasons(result, 1)
    assert not result["replacement_terms"].empty
    assert set(result["replacement_terms"].source) == {"p[0]/ring[0]"}


def test_tiny_unresolved_bracket_is_never_coalesced_filled_or_assigned_parent():
    start = 4.
    stop = np.nextafter(start, np.inf)
    spans = cells(cell(start=start, stop=stop, state="unresolved"))
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10.], spans, endpoint_budget_mm=1e-6)
    assert_candidate_contract(result, [10.])
    assert result["intervals"].empty and result["replacement_terms"].empty
    assert reasons(result, 0) == {"unresolved-solid-boundary"}
    assert result["summary"]["unresolved_cells"] == 1
    assert result["summary"]["unresolved_cell_width_upper_mm"] == omittedpath.length_bound(spans, upper=True)
    assert result["summary"]["unresolved_cell_width_upper_mm"] > 0.


@pytest.mark.parametrize("same_material", [False, True])
def test_overlapping_leaf_cells_refused_even_when_parent_terms_cancel(same_material):
    parent = "Al" if same_material else "Honeycomb"
    spans = cells(cell(start=2., stop=5., replaces_material=parent),
                  cell(start=4., stop=6., replaces_material=parent, source="p[0]/ring[1]"))
    result = compose(hits((0, 0., 10., parent, 1), (1, 0., 10., parent, 1)), [10., 10.], spans,
                     records=[record(), record("p[0]/ring[1]")], parent=parent)
    assert_candidate_contract(result, [10., 10.])
    assert reasons(result, 0) == {"overlapping-possible-leaf-cells"}
    assert result["replacement_terms"].empty
    assert result["decisions"].candidate_resolved.tolist() == [False, True]


def test_overlap_includes_unresolved_and_nested_cells_but_touching_cells_are_allowed():
    raw = hits((0, 0., 10., "Honeycomb", 1))
    nested = cells(cell(start=1., stop=9., state="unresolved"), cell(start=2., stop=3.),
                   cell(start=5., stop=6.))
    refused = compose(raw, [10.], nested)
    assert {"unresolved-solid-boundary", "overlapping-possible-leaf-cells"} <= reasons(refused, 0)
    assert refused["replacement_terms"].empty
    touching = compose(raw, [10.], cells(cell(start=1., stop=3.), cell(start=3., stop=5.)))
    assert_candidate_contract(touching, [10.])
    assert occupied(touching, 0, "Al") == 4.


@pytest.mark.parametrize("wrong", ["Al", "vacuum", "Steel"])
def test_wrong_parent_refused_even_if_leaf_already_matches_stored_material(wrong):
    raw = hits((0, 0., 10., wrong, 1), (1, 0., 10., "Honeycomb", 1))
    result = compose(raw, [10., 10.], cells(cell(), cell(lane=1)))
    assert_candidate_contract(result, [10., 10.])
    assert "stored-parent-mismatch" in reasons(result, 0)
    assert result["decisions"].candidate_resolved.tolist() == [False, True]
    assert set(result["replacement_terms"].lane) == {1}
    assert result["summary"]["unexpected_parent_comparisons"] == 1


def test_parent_check_covers_full_cell_not_only_midpoint():
    raw = hits((0, 0., 9., "Honeycomb", 1), (0, 9., 10., "Steel", 1))
    result = compose(raw, [10.], cells(cell(start=2., stop=9.5)))
    assert_candidate_contract(result, [10.])
    assert "stored-parent-mismatch" in reasons(result, 0)
    pairs = result["parent_comparisons"].sort_values("s0_mm")
    np.testing.assert_array_equal(pairs[["s0_mm", "s1_mm"]], [[2., 9.], [9., 9.5]])
    assert pairs.expected_parent.tolist() == [True, False]
    assert result["replacement_terms"].empty


@pytest.mark.parametrize("bad_rows", [
    [(0, 1., 3., "Honeycomb", -1)],
    [(0, 0., 10., "Honeycomb", 1), (0, 1., 3., "Honeycomb", 1)],
    [(0, 0., 10., "Honeycomb", 1), (0, 1., 3., "Steel", 1)],
    [(0, 0., 10., "Honeycomb", 1), (0, 1., 3., "Steel", -1)],
])
def test_invalid_stored_lane_does_not_destroy_good_lane(bad_rows):
    raw = hits(*bad_rows, (1, 0., 10., "Honeycomb", 1))
    result = compose(raw, [10., 10.], cells(cell(), cell(lane=1)))
    assert_candidate_contract(result, [10., 10.])
    assert "invalid-stored-occupancy" in reasons(result, 0)
    assert result["decisions"].candidate_resolved.tolist() == [False, True]
    assert occupied(result, 1, "Al") == 2.
    assert set(result["replacement_terms"].lane) == {1}


def test_signed_stored_cancellation_is_ambiguous_but_zero_width_is_not_occupancy():
    raw = hits((0, 0., 10., "Honeycomb", 1), (0, 1., 5., "Steel", 1),
               (0, 1., 5., "Steel", -1), (0, 6., 6., "Other", -1))
    result = compose(raw, [10.], cells(cell()))
    assert_candidate_contract(result, [10.])
    assert result["intervals"].empty and result["replacement_terms"].empty
    assert reasons(result, 0) == {"ambiguous-stored-parent-ownership"}
    clear = compose(raw.iloc[[0, 3]], [10.], cells(cell()))
    assert occupied(clear, 0, "Al") == 2.
    assert occupied(clear, 0, "Honeycomb") == 8.


@pytest.mark.parametrize("lengths", [[], [0.], [0., 3., 0.]])
def test_empty_and_zero_length_lanes_remain_explicit_decisions(lengths):
    result = compose(hits(), lengths)
    assert_candidate_contract(result, lengths)
    assert result["decisions"].candidate_resolved.all()
    assert result["replacement_terms"].empty
    assert result["summary"]["global_refusals"] == []
    assert result["summary"]["all_rays_resolved"] is True
    if len(lengths) == 3:
        assert occupied(result, 1, "vacuum") == 3.


def test_unevaluated_placement_is_global_refusal_even_with_no_cells():
    records = [record()]
    reports = reports_for(cells(), records)
    reports[0]["resolved"] = False
    result = compose(hits(), [0., 10.], records=records, reports=reports)
    assert_candidate_contract(result, [0., 10.])
    assert result["decisions"].candidate_resolved.tolist() == [False, False]
    assert result["summary"]["global_refusals"] == ["unevaluated-placement:p[0]/ring[0]"]
    assert result["summary"]["all_rays_resolved"] is False
    empty = compose(hits(), [], records=records, reports=reports)
    assert empty["summary"]["all_rays_resolved"] is False


@pytest.mark.parametrize("key", ["cap_reached", "capped"])
def test_cap_without_unresolved_evidence_blocks_every_lane(key):
    spans = cells(cell())
    reports = reports_for(spans, [record()])
    reports[0]["method"][key] = True
    result = compose(hits((0, 0., 10., "Honeycomb", 1)), [10., 4.], spans, reports=reports)
    assert_candidate_contract(result, [10., 4.])
    assert result["decisions"].candidate_resolved.tolist() == [False, False]
    assert result["summary"]["global_refusals"] == ["cap-without-unresolved-evidence:p[0]/ring[0]"]


def test_cap_with_retained_unresolved_evidence_refuses_only_affected_lane():
    spans = cells(cell(state="unresolved"))
    reports = reports_for(spans, [record()])
    reports[0]["method"]["cap_reached"] = True
    result = compose(hits(), [10., 4.], spans, reports=reports)
    assert_candidate_contract(result, [10., 4.])
    assert result["decisions"].candidate_resolved.tolist() == [False, True]
    assert result["summary"]["global_refusals"] == []


@pytest.mark.parametrize("case", ["missing", "duplicate-report", "duplicate-record", "orphan", "extra-report"])
def test_exact_unsupported_placement_report_coverage_required(case):
    spans = cells(cell())
    records = [record()]
    reports = reports_for(spans, records)
    if case == "missing":
        reports = []
    elif case == "duplicate-report":
        reports *= 2
    elif case == "duplicate-record":
        records *= 2
    elif case == "orphan":
        spans.loc[0, "source"] = "unaccounted[0]"
    else:
        reports.append(dict(reports[0], source="unaccounted[0]"))
    with pytest.raises(ValueError, match="report per unsupported|orphan"):
        compose(hits(), [10.], spans, records=records, reports=reports)


def test_nonunsupported_census_records_do_not_require_probe_reports():
    records = [record(), record("retained[0]", classification="retained"),
               record("approximate[0]", classification="approximate")]
    result = compose(hits(), [10.], records=records)
    assert_candidate_contract(result, [10.])
    assert result["summary"]["all_rays_resolved"] is True


@pytest.mark.parametrize("key,value", [("material", "Steel"), ("scope", "different"),
                                       ("solid_tag", "xtru"), ("leaf", False)])
def test_report_must_agree_with_census(key, value):
    spans = cells(cell())
    reports = reports_for(spans, [record()])
    reports[0][key] = value
    with pytest.raises(ValueError, match="disagrees with census"):
        compose(hits(), [10.], spans, reports=reports)


@pytest.mark.parametrize("key", ["resolved", "leaf"])
@pytest.mark.parametrize("value", [1, "true", np.bool_(True), None])
def test_report_flags_require_actual_python_booleans(key, value):
    reports = reports_for(cells(), [record()])
    reports[0][key] = value
    with pytest.raises(ValueError, match="actual booleans"):
        compose(hits(), [10.], reports=reports)


@pytest.mark.parametrize("key,value", [("state", "outside"), ("state", None),
    ("ancestors_verified", 1), ("ancestors_verified", "True"), ("ancestors_verified", None),
    ("leaf_verified", 0), ("leaf_verified", "False"), ("leaf_verified", None),
    ("source", ""), ("scope", ""), ("replaces_material", ""), ("material", ""),
    ("scope", 4), ("replaces_material", None), ("s0_mm", -1.), ("s1_mm", 11.),
    ("s1_mm", np.nan), ("s1_mm", np.inf), ("s1_mm", 2.), ("s1_mm", 1.),
    ("lane", -1), ("lane", 1), ("lane", .5)])
def test_invalid_cell_evidence_is_rejected_before_composition(key, value):
    spans = cells(cell())
    reports = reports_for(spans, [record()])
    spans[key] = [value]
    with pytest.raises(ValueError):
        compose(hits(), [10.], spans, reports=reports)


@pytest.mark.parametrize("case", ["missing-column", "duplicate-column", "not-dataframe"])
def test_cell_table_schema_is_validated(case):
    spans = cells(cell())
    reports = reports_for(spans, [record()])
    if case == "missing-column":
        spans = spans.drop(columns="state")
    elif case == "duplicate-column":
        spans = pd.concat([spans, spans[["source"]]], axis=1)
    else:
        spans = spans.to_dict("records")
    with pytest.raises(ValueError, match="omission cells require unique columns"):
        supportnavigation.compose(hits(), [10.], spans, reports, [record()], endpoint_budget_mm=0.)


@pytest.mark.parametrize("key,value", [("material", "Steel"), ("scope", "different"),
                                      ("replaces_material", "vacuum"), ("leaf_verified", False)])
def test_cell_metadata_must_agree_with_placement_report(key, value):
    spans = cells(cell())
    reports = reports_for(spans, [record()])
    spans[key] = value
    with pytest.raises(ValueError, match="metadata disagrees"):
        compose(hits(), [10.], spans, reports=reports)


@pytest.mark.parametrize("key", ["solid_lower_mm", "solid_upper_mm", "hierarchy_verified_lower_mm",
                                "hit_lanes", "possible_lanes"])
def test_report_bounds_and_lane_counts_must_match_retained_cells_exactly(key):
    spans = cells(cell(), cell(lane=1, state="unresolved"))
    reports = reports_for(spans, [record()])
    reports[0][key] = np.nextafter(reports[0][key], np.inf)
    with pytest.raises(ValueError, match="bounds/counts disagree"):
        compose(hits(), [10., 10.], spans, reports=reports)


@pytest.mark.parametrize("key", ["inside_cells", "unresolved_cells"])
def test_method_cell_counts_must_match_evidence(key):
    spans = cells(cell())
    reports = reports_for(spans, [record()])
    reports[0]["method"][key] += 1
    with pytest.raises(ValueError, match="method counts disagree"):
        compose(hits(), [10.], spans, reports=reports)


def test_unevaluated_report_cannot_retain_cells():
    spans = cells(cell())
    reports = reports_for(spans, [record()])
    reports[0]["resolved"] = False
    with pytest.raises(ValueError, match="unevaluated placement has cells"):
        compose(hits(), [10.], spans, reports=reports)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, np.inf])
def test_compose_pair_cap_must_be_positive_integer(value):
    with pytest.raises(ValueError, match="max_pairs must be a positive integer"):
        compose(hits(), [10.], max_pairs=value)


def test_parent_pair_budget_is_explicit_and_never_truncates():
    raw = hits((0, 0., 5., "Honeycomb", 1), (0, 5., 10., "Honeycomb", 1))
    spans = cells(cell(start=1., stop=2.), cell(start=6., stop=7.))
    with pytest.raises(ValueError, match="parent comparison exceeds explicit pair bound"):
        compose(raw, [10.], spans, max_pairs=3)
    result = compose(raw, [10.], spans, max_pairs=4)
    assert_candidate_contract(result, [10.])
    assert occupied(result, 0, "Al") == 2.


def test_endpoint_coalescence_is_explicit_and_does_not_mutate_source_evidence():
    raw = hits((0, 0., 5., "Honeycomb", 1), (0, 5. + 1e-10, 10., "Honeycomb", 1))
    raw["particle_id"] = [91, 91]
    raw["saved_source"] = ["a", "b"]
    raw.index = [8, 8]
    lengths = np.array([10.])
    spans = cells(cell(start=4., stop=6.))
    records = [record()]
    reports = reports_for(spans, records)
    before = raw.copy(deep=True), spans.copy(deep=True), lengths.copy(), deepcopy(records), deepcopy(reports)
    strict = compose(raw, lengths, spans, records=records, reports=reports)
    assert "stored-parent-mismatch" in reasons(strict, 0)
    result = compose(raw, lengths, spans, records=records, reports=reports, endpoint_budget_mm=1e-9)
    assert_candidate_contract(result, lengths)
    assert occupied(result, 0, "Al") == 2.
    assert result["summary"]["endpoint_policy"]["changed_endpoints"] == 2
    assert result["summary"]["endpoint_policy"]["actual_max_shift_mm"] <= 1e-9
    pd.testing.assert_frame_equal(raw, before[0])
    pd.testing.assert_frame_equal(spans, before[1])
    np.testing.assert_array_equal(lengths, before[2])
    assert records == before[3] and reports == before[4]
    pd.testing.assert_frame_equal(result["adjusted_hits"][["particle_id", "saved_source"]],
                                  raw[["particle_id", "saved_source"]])
    assert result["endpoints"].row_position.tolist() == [0, 1, 0, 1]
    assert result["endpoints"].row_index.tolist() == [8, 8, 8, 8]


def test_coalescence_failure_is_not_downgraded_to_an_empty_success():
    raw = hits((0, 2., 2. + 1e-10, "Honeycomb", 1))
    with pytest.raises(ValueError, match="collapse a positive-width"):
        compose(raw, [10.], endpoint_budget_mm=1e-9)


def test_lane_is_positional_and_particle_identity_binding_belongs_to_runner():
    raw = hits((0, 0., 10., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1))
    raw["particle_id"] = [800, 800]
    raw["interval_id"] = [4, 7]
    result = compose(raw, [10., 10.], cells(cell(lane=1)))
    assert_candidate_contract(result, [10., 10.])
    assert occupied(result, 0, "Al") == 0.
    assert occupied(result, 1, "Al") == 2.
    assert "particle_id" not in result["decisions"]
    pd.testing.assert_frame_equal(result["adjusted_hits"], raw)


def canonical(frame, columns):
    return frame[columns].sort_values(columns, kind="stable").reset_index(drop=True)


def test_source_cell_and_report_permutations_preserve_physical_result():
    raw = hits((0, 0., 10., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1),
               (2, 0., 10., "Honeycomb", 1))
    spans = cells(cell(start=1., stop=3.), cell(lane=1, state="unresolved"),
                  cell(start=6., stop=8., source="p[0]/ring[1]"), cell(lane=2, source="p[0]/ring[1]"))
    records = [record(), record("p[0]/ring[1]")]
    reports = reports_for(spans, records)
    baseline = compose(raw, [10.] * 3, spans, records=records, reports=reports)
    for seed in (3, 11, 29):
        result = compose(raw.sample(frac=1, random_state=seed), [10.] * 3,
                         spans.sample(frac=1, random_state=seed), records=records[::-1], reports=reports[::-1])
        for key in ("intervals", "decisions", "refusals"):
            pd.testing.assert_frame_equal(result[key], baseline[key])
        columns = supportnavigation.COLS + ["sign", "source", "operation", "replaces_material"]
        pd.testing.assert_frame_equal(canonical(result["replacement_terms"], columns),
                                      canonical(baseline["replacement_terms"], columns))
        assert result["summary"] == baseline["summary"]


@pytest.fixture
def torus_geometry(tmp_path):
    def make(*, parent="Honeycomb", delta=360., start=0., parent_size=60., solid=None):
        path = tmp_path / "simple_torus.gdml"
        leaf = solid or (f'<torus name="ring" rtor="10" rmin="0" rmax="2" '
                         f'startphi="{start}" deltaphi="{delta}" aunit="deg" lunit="mm"/>')
        path.write_text(f'''<gdml><solids><box name="world" x="100" y="100" z="100"/>
        <box name="parent" x="{parent_size}" y="{parent_size}" z="{parent_size}"/>{leaf}</solids>
        <structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="world"/>
        <physvol name="p"><volumeref ref="Parent"/></physvol></volume>
        <volume name="Parent"><materialref ref="{parent}"/><solidref ref="parent"/>
        <physvol name="ring"><volumeref ref="Leaf"/></physvol></volume>
        <volume name="Leaf"><materialref ref="Al"/><solidref ref="ring"/></volume></structure>
        <setup><world ref="World"/></setup></gdml>''')
        census = dict(schema="tracker-navigation-census-v1", bbox_mm=[[-20.] * 3, [20.] * 3],
                      records=[record()], counts={"unsupported": 1}, placement_warnings=[])
        return path, census
    return make


def navigate(gdml, census, origins, lengths, *, raw=None, directions=None, **kwargs):
    origins = np.asarray(origins, float).reshape(-1, 3)
    directions = np.tile([0., 0., 1.], (len(origins), 1)) if directions is None else directions
    lengths = np.asarray(lengths, float)
    if raw is None:
        raw = hits(*[(lane, 0., length, "Honeycomb", 1) for lane, length in enumerate(lengths)])
    return supportnavigation.navigate(gdml, census, origins, directions, lengths, raw,
        endpoint_budget_mm=kwargs.pop("endpoint_budget_mm", 0.), **kwargs)


def assert_analytic_bracket(result, lane, truth):
    spans = result["omitted_cells"]
    subset = spans[spans.lane == lane]
    lower = omittedpath.length_bound(subset[subset.state == "inside"], upper=False)
    upper = omittedpath.length_bound(subset, upper=True)
    assert lower <= truth <= upper, (lane, lower, truth, upper)
    if (subset.state == "unresolved").any():
        assert "unresolved-solid-boundary" in reasons(result, lane)
        assert not result["decisions"].loc[lane, "candidate_resolved"]
        assert result["intervals"][result["intervals"].lane == lane].empty
    else:
        assert result["decisions"].loc[lane, "candidate_resolved"]
        assert occupied(result, lane, "Al") == pytest.approx(truth, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("parent", ["Honeycomb", "Vacuum"])
def test_navigate_real_torus_inside_hole_and_boundary_crossing(torus_geometry, parent):
    gdml, census = torus_geometry(parent=parent)
    lengths = [1., 10., 10.]
    origins = [[10., 0., -.5], [0., 0., -5.], [10., 0., -5.]]
    raw = hits() if parent == "Vacuum" else None
    result = navigate(gdml, census, origins, lengths, raw=raw, min_cell_mm=1e-5)
    assert_candidate_contract(result, lengths)
    assert result["decisions"].candidate_resolved.tolist() == [True, True, False]
    for lane, truth in enumerate([1., 0., 4.]):
        assert_analytic_bracket(result, lane, truth)
    report = result["placement_reports"][0]
    assert report["resolved"] is True and report["leaf"] is True
    assert report["parent_material"] == parent
    assert report["hit_lanes"] == report["possible_lanes"] == 2
    assert result["summary"]["placements"] == 1
    assert result["summary"]["census_bbox_roundoff_guard_mm"] > 0.
    spans = result["omitted_cells"]
    assert spans.ancestors_verified.all() and spans.leaf_verified.all()
    assert report["solid_lower_mm"] == omittedpath.length_bound(spans[spans.state == "inside"], upper=False)
    assert report["solid_upper_mm"] == omittedpath.length_bound(spans, upper=True)
    assert set(result["replacement_terms"].sign) == ({1} if parent == "Vacuum" else {-1, 1})


def test_navigate_wrong_parent_and_unverified_ancestor_refuse(torus_geometry):
    gdml, census = torus_geometry()
    wrong = navigate(gdml, census, [[10., 0., 0.]], [1.], raw=hits())
    assert reasons(wrong, 0) == {"stored-parent-mismatch"}
    assert_candidate_contract(wrong, [1.])
    gdml, census = torus_geometry(parent_size=10.)
    unverified = navigate(gdml, census, [[10., 0., 0.]], [1.])
    assert "unverified-ancestor" in reasons(unverified, 0)
    assert not unverified["omitted_cells"].ancestors_verified.any()
    assert unverified["placement_reports"][0]["unresolved_ancestors"]
    assert unverified["replacement_terms"].empty
    assert_candidate_contract(unverified, [1.])


def test_navigate_tangent_plus_minus_and_ulp_perturbations_bracket_or_refuse(torus_geometry):
    gdml, census = torus_geometry()
    x = np.array([12. - 1e-5, np.nextafter(12., -np.inf), 12.,
                  np.nextafter(12., np.inf), 12. + 1e-5])
    origins = np.column_stack((x, np.zeros(len(x)), np.full(len(x), -5.)))
    result = navigate(gdml, census, origins, [10.] * len(x), min_cell_mm=1e-5, max_cells=100000)
    assert_candidate_contract(result, [10.] * len(x))
    truth = 2. * np.sqrt(np.maximum(0., (12. - x) * (x - 8.)))
    for lane, chord in enumerate(truth):
        assert_analytic_bracket(result, lane, chord)
    assert truth[0] > truth[1] > 0.
    assert result["decisions"].candidate_resolved.iloc[-1]
    assert not result["decisions"].candidate_resolved.iloc[2]


@pytest.mark.parametrize("sector_edge", ["start", "end"])
def test_navigate_sector_plus_minus_and_ulp_perturbations_bracket_or_refuse(torus_geometry, sector_edge):
    gdml, census = torus_geometry(delta=90.)
    offsets = np.array([-1e-5, np.nextafter(0., -np.inf), 0., np.nextafter(0., np.inf), 1e-5])
    x, y = np.full(len(offsets), 10.), offsets
    if sector_edge == "end":
        x, y = offsets, np.full(len(offsets), 10.)
    origins = np.column_stack((x, y, np.full(len(offsets), -.5)))
    result = navigate(gdml, census, origins, [1.] * len(offsets), min_cell_mm=1e-4, max_cells=500)
    assert_candidate_contract(result, [1.] * len(offsets))
    for lane, offset in enumerate(offsets):
        assert_analytic_bracket(result, lane, float(offset >= 0.))
    assert result["decisions"].candidate_resolved.iloc[0]
    assert result["decisions"].candidate_resolved.iloc[-1]
    assert occupied(result, 0, "Al") == 0.
    assert occupied(result, len(offsets) - 1, "Al") == 1.
    assert not result["decisions"].candidate_resolved.iloc[2]


def test_navigate_node_cap_preserves_unknown_cells_and_analytic_upper_bound(torus_geometry):
    gdml, census = torus_geometry()
    result = navigate(gdml, census, [[-15., 0., 0.], [0., 0., -5.]], [30., 10.],
                      directions=np.array([[1., 0., 0.], [0., 0., 1.]]), max_cells=1)
    assert_candidate_contract(result, [30., 10.])
    report = result["placement_reports"][0]
    assert report["method"]["cap_reached"]
    assert report["method"]["evaluated_nodes"] <= 1
    assert (result["omitted_cells"].state == "unresolved").any()
    assert_analytic_bracket(result, 0, 8.)
    assert_analytic_bracket(result, 1, 0.)
    assert result["summary"]["unresolved_cell_width_upper_mm"] > 0.


@pytest.mark.parametrize("lengths,origins", [([], []), ([0.], [[10., 0., 0.]])])
def test_navigate_empty_and_zero_length(torus_geometry, lengths, origins):
    gdml, census = torus_geometry()
    result = navigate(gdml, census, origins, lengths)
    assert_candidate_contract(result, lengths)
    assert result["intervals"].empty and result["omitted_cells"].empty
    assert result["summary"]["all_rays_resolved"] is True
    assert result["placement_reports"][0]["solid_upper_mm"] == 0.


def test_navigate_unsupported_actual_solid_cannot_be_silently_treated_as_miss(torus_geometry):
    gdml, census = torus_geometry(solid='<sphere name="ring" rmin="0" rmax="2"/>')
    census["records"][0]["solid_tag"] = "sphere"
    result = navigate(gdml, census, [[0., 0., 0.], [15., 0., 0.]], [1., 1.])
    assert_candidate_contract(result, [1., 1.])
    assert result["decisions"].candidate_resolved.tolist() == [False, False]
    assert result["placement_reports"][0]["resolved"] is False
    assert "not implemented" in result["placement_reports"][0]["reason"]
    assert result["summary"]["global_refusals"] == ["unevaluated-placement:p[0]/ring[0]"]


@pytest.mark.parametrize("case,match", [
    ("schema", "recognized navigation census"), ("bbox-shape", "finite ordered census bbox"),
    ("bbox-infinite", "finite ordered census bbox"), ("bbox-inverted", "finite ordered census bbox"),
    ("origin-outside", "rays outside"), ("endpoint-outside", "rays outside"),
    ("duplicate", "duplicate census placement"), ("warnings", "placement assumptions"),
    ("assumptions", "placement assumptions"), ("counts", "classification counts disagree"),
    ("classification", "unrecognized census classification"),
])
def test_navigate_validates_census_and_selection_domain(torus_geometry, case, match):
    gdml, census = torus_geometry()
    origin, length = [[10., 0., 0.]], [1.]
    if case == "schema":
        census["schema"] = "unknown"
    elif case == "bbox-shape":
        census["bbox_mm"] = [0., 1.]
    elif case == "bbox-infinite":
        census["bbox_mm"][0][0] = -np.inf
    elif case == "bbox-inverted":
        census["bbox_mm"][0][0] = 21.
    elif case == "origin-outside":
        origin = [[21., 0., 0.]]
    elif case == "endpoint-outside":
        length = [21.]
    elif case == "duplicate":
        census["records"] *= 2
    elif case == "warnings":
        census["placement_warnings"] = ["unresolved"]
    elif case == "assumptions":
        census["records"][0]["placement_assumptions"] = ["unresolved"]
    elif case == "counts":
        census["counts"] = {"unsupported": 2}
    else:
        census["records"][0]["classification"] = "unknown"
        census["counts"] = {"unknown": 1}
    with pytest.raises(ValueError, match=match):
        navigate(gdml, census, origin, length)


@pytest.mark.parametrize("kwargs", [dict(min_cell_mm=0.), dict(min_cell_mm=-1.),
    dict(min_cell_mm=np.nan), dict(min_cell_mm=np.inf), dict(max_cells=0),
    dict(max_cells=True), dict(max_cells=1.5), dict(max_pairs=0), dict(max_pairs=False)])
def test_navigate_requires_explicit_positive_work_and_resolution_bounds(torus_geometry, kwargs):
    gdml, census = torus_geometry()
    with pytest.raises(ValueError, match="positive"):
        navigate(gdml, census, [[10., 0., 0.]], [1.], **kwargs)


def test_navigate_does_not_mutate_gdml_census_rays_or_hits(torus_geometry):
    gdml, census = torus_geometry()
    origin = np.array([[10., 0., -.5], [0., 0., -5.]])
    direction = np.tile([0., 0., 1.], (2, 1))
    lengths = np.array([1., 10.])
    raw = hits((0, 0., 1., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1))
    raw["particle_id"] = [12, 19]
    before = (gdml.read_bytes(), deepcopy(census), origin.copy(), direction.copy(), lengths.copy(), raw.copy(deep=True))
    result = supportnavigation.navigate(gdml, census, origin, direction, lengths, raw, endpoint_budget_mm=0.)
    assert_candidate_contract(result, lengths)
    assert gdml.read_bytes() == before[0] and census == before[1]
    for actual, saved in zip((origin, direction, lengths), before[2:5]):
        np.testing.assert_array_equal(actual, saved)
    pd.testing.assert_frame_equal(raw, before[5])
    result["adjusted_hits"].loc[0, "s0_mm"] = .25
    pd.testing.assert_frame_equal(raw, before[5])


def test_navigate_lane_permutation_is_equivariant(torus_geometry):
    gdml, census = torus_geometry()
    origins = np.array([[10., 0., -.5], [0., 0., -5.], [10., 0., -5.]])
    lengths = np.array([1., 10., 10.])
    baseline = navigate(gdml, census, origins, lengths, min_cell_mm=1e-4)
    permutation = np.array([2, 0, 1])
    replay = navigate(gdml, census, origins[permutation], lengths[permutation], min_cell_mm=1e-4)
    for key in ("intervals", "decisions", "refusals", "omitted_cells"):
        restored = replay[key].copy()
        restored["lane"] = permutation[restored.lane.to_numpy(dtype=int)]
        columns = list(baseline[key].columns)
        pd.testing.assert_frame_equal(canonical(restored, columns), canonical(baseline[key], columns))
    for key in ("solid_lower_mm", "solid_upper_mm", "hierarchy_verified_lower_mm", "hit_lanes", "possible_lanes"):
        assert replay["placement_reports"][0][key] == baseline["placement_reports"][0][key]


@pytest.mark.parametrize("edge", ["start", "end"])
def test_nonaxis_sector_boundary_coordinate_ulps_are_not_fabricated_crossings(torus_geometry, edge):
    gdml, census = torus_geometry(start=45., delta=90.)
    boundary = 7. if edge == "start" else -7.
    varied = np.array([boundary - 1e-5, np.nextafter(boundary, -np.inf), boundary,
                       np.nextafter(boundary, np.inf), boundary + 1e-5])
    x, y = (np.full(5, 7.), varied) if edge == "start" else (varied, np.full(5, 7.))
    origins = np.column_stack((x, y, np.full(5, -.5)))
    result = navigate(gdml, census, origins, [1.] * 5, min_cell_mm=1e-4, max_cells=500)
    assert_candidate_contract(result, [1.] * 5)
    for lane, value in enumerate(varied):
        assert_analytic_bracket(result, lane, float(value >= boundary))
    assert result["decisions"].candidate_resolved.tolist() == [True, False, False, False, True]
    assert occupied(result, 0, "Al") == 0.
    assert occupied(result, 4, "Al") == 1.


def test_independent_lane_replay_preserves_decisions_and_partitions():
    lengths = np.array([10., 10., 0., 7.])
    raw = hits((0, 0., 10., "Honeycomb", 1), (1, 0., 10., "Honeycomb", 1),
               (3, 0., 7., "Steel", 1))
    spans = cells(cell(), cell(lane=1, state="unresolved"), cell(lane=3))
    result = compose(raw, lengths, spans)
    assert_candidate_contract(result, lengths)
    assert result["decisions"].candidate_resolved.tolist() == [True, False, True, False]
    for lane, length in enumerate(lengths):
        lane_hits = raw[raw.lane == lane].assign(lane=0)
        lane_cells = spans[spans.lane == lane].assign(lane=0)
        replay = compose(lane_hits, [length], lane_cells)
        assert_candidate_contract(replay, [length])
        for key in ("intervals", "decisions", "refusals"):
            expected = result[key][result[key].lane == lane].reset_index(drop=True)
            restored = replay[key].assign(lane=lane)
            pd.testing.assert_frame_equal(restored, expected)


@pytest.mark.parametrize("leaf_material", ["Al", "Honeycomb"])
def test_retained_same_material_sibling_cancellation_cannot_admit_omitted_leaf(leaf_material):
    raw = hits((0, 0., 10., "Honeycomb", 1), (0, 1., 5., "Honeycomb", 1),
               (0, 1., 5., "Honeycomb", -1), (1, 0., 10., "Honeycomb", 1))
    spans = cells(cell(material=leaf_material), cell(lane=1, material=leaf_material))
    result = compose(raw, [10., 10.], spans, records=[record(material=leaf_material)])
    assert_candidate_contract(result, [10., 10.])
    assert result["decisions"].candidate_resolved.tolist() == [False, True]
    assert reasons(result, 0) == {"ambiguous-stored-parent-ownership"}
    assert result["summary"]["ambiguous_ownership_rays"] == 1
    assert set(result["replacement_terms"].lane) == {1}
    assert result["parent_comparisons"].expected_parent.all()


@pytest.mark.parametrize("dtype", ["int64", "uint8", "uint64"])
def test_unsigned_signs_do_not_wrap_negative_exit_events(dtype):
    raw = hits((0, 0., 1., "Al", 1), (0, 2., 3., "Al", 1)).astype({"sign": dtype})
    result = compose(raw, [4.])
    assert_candidate_contract(result, [4.])
    assert result["decisions"].candidate_resolved.tolist() == [True]
    assert occupied(result, 0, "Al") == 2.
