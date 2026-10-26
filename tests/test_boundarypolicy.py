"""Bounded stored-interval diagnostics, not tests of GDML truth or transport admission."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim import boundarypolicy as policy
from trkfastsim.analyses.navigationaudit import occupancy


BUDGET = 1e-9
COLUMNS = ["lane", "s0_mm", "s1_mm", "material", "sign"]


def hits(rows=()):
    return pd.DataFrame(rows, columns=COLUMNS)


def check_partition(frame, lengths):
    assert frame.columns.tolist() == COLUMNS[:-1]
    assert frame.dtypes.astype(str).to_dict() == dict(lane="int64", s0_mm="float64", s1_mm="float64", material="object")
    assert (frame.s1_mm > frame.s0_mm).all()
    pd.testing.assert_frame_equal(frame, frame.sort_values(["lane", "s0_mm"]).reset_index(drop=True))
    lane = frame.lane.to_numpy()
    np.testing.assert_array_equal(np.unique(lane), np.flatnonzero(np.asarray(lengths) > 0))
    if not len(frame):
        return
    first = np.r_[True, lane[1:] != lane[:-1]]
    last = np.r_[lane[:-1] != lane[1:], True]
    np.testing.assert_array_equal(frame.s0_mm.to_numpy()[first], np.zeros(first.sum()))
    np.testing.assert_array_equal(frame.s1_mm.to_numpy()[last], np.asarray(lengths)[lane[last]])
    np.testing.assert_array_equal(frame.s1_mm.to_numpy()[:-1][~first[1:]], frame.s0_mm.to_numpy()[1:][~first[1:]])


def check_rows(frame, rows):
    expected = pd.DataFrame(rows, columns=COLUMNS[:-1]).astype(policy._DTYPES)
    pd.testing.assert_frame_equal(frame, expected)


@pytest.mark.parametrize("offset", [-.5 * BUDGET, .5 * BUDGET])
@pytest.mark.parametrize("material", ["Al", "Cu"])
def test_tiny_adjacent_mismatch_is_only_coalesced_when_explicit(offset, material):
    raw = hits([(0, 0., 1., "Al", 1), (0, 1. + offset, 2., material, 1)])
    if offset < 0:
        with pytest.raises(ValueError, match="occupancy|cross-material"):
            policy.partition(raw, [2.])
    else:
        strict = policy.partition(raw, [2.])
        assert strict.material.tolist() == ["Al", "vacuum", material]
    adjusted, evidence = policy.coalesce(raw, [2.], max_shift_mm=BUDGET)
    canonical = min(1., 1. + offset) + .5 * abs(offset)
    assert adjusted.s1_mm.iloc[0] == canonical
    assert adjusted.s0_mm.iloc[1] == canonical
    result = policy.partition(adjusted, [2.])
    check_rows(result, [(0, 0., canonical, "Al"), (0, canonical, 2., material)])
    check_partition(result, [2.])
    table, summary = evidence["endpoints"], evidence["summary"]
    assert summary["actual_max_shift_mm"] == table.shift_mm.abs().max()
    assert summary["actual_max_shift_mm"] <= BUDGET
    assert summary["total_length_change_mm"] == pytest.approx(offset, rel=0, abs=3e-16)
    assert summary["total_absolute_length_change_mm"] == pytest.approx(abs(offset), rel=0, abs=3e-16)
    assert summary["changed_endpoints"] == 2
    assert table.group_max_mm.sub(table.group_min_mm).max() <= BUDGET


def test_evidence_is_replayable_preserves_sign_and_duplicate_row_identity():
    raw = hits([(0, 0., 3., "Al", 1), (0, 1., 2., "Al", -1),
                (0, 1. + .25 * BUDGET, 2. - .25 * BUDGET, "Cu", 1)])
    raw.index = pd.Index([7, 7, 4], name="original")
    raw["source"] = pd.Categorical(["mother", "hole", "daughter"])
    raw["component"] = "calo"
    raw["piece"] = [3., np.nan, 8.]
    raw["raw_endpoint_mm"] = ["user", "columns", "are retained"]
    original = raw.copy(deep=True)
    adjusted, evidence = policy.coalesce(raw, [3.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(adjusted.drop(columns=["s0_mm", "s1_mm"]), original.drop(columns=["s0_mm", "s1_mm"]))
    assert adjusted.index.equals(original.index)
    table = evidence["endpoints"]
    assert table.row_position.tolist() == [0, 1, 2, 0, 1, 2]
    assert table.row_index.tolist() == [7, 7, 4, 7, 7, 4]
    assert table.sign.tolist() == [1, -1, 1, 1, -1, 1]
    assert table.endpoint.tolist() == ["s0_mm"] * 3 + ["s1_mm"] * 3
    np.testing.assert_array_equal(table.raw_endpoint_mm, np.r_[raw.s0_mm, raw.s1_mm])
    np.testing.assert_array_equal(table.canonical_endpoint_mm, np.r_[adjusted.s0_mm, adjusted.s1_mm])
    np.testing.assert_array_equal(table.shift_mm, table.canonical_endpoint_mm - table.raw_endpoint_mm)
    assert (table.group_size == table.group_anchor_size + table.group_source_size).all()
    delta = (adjusted.s1_mm - adjusted.s0_mm) - (raw.s1_mm - raw.s0_mm)
    assert evidence["summary"]["total_length_change_mm"] == pytest.approx(delta.sum(), rel=0, abs=1e-15)
    assert evidence["summary"]["signed_total_length_change_mm"] == pytest.approx((delta * raw.sign).sum(), rel=0, abs=1e-15)
    check_partition(policy.partition(adjusted, [3.]), [3.])


def test_anchor_canonicalization_is_exact_and_fills_exact_domain():
    raw = hits([(0, .5 * BUDGET, 3. - .5 * BUDGET, "Al", 1)])
    adjusted, evidence = policy.coalesce(raw, [3.], max_shift_mm=BUDGET)
    assert adjusted.s0_mm.iloc[0] == 0.
    assert adjusted.s1_mm.iloc[0] == 3.
    assert evidence["endpoints"].group_anchor_size.tolist() == [1, 1]
    check_rows(policy.partition(adjusted, [3.]), [(0, 0., 3., "Al")])
    check_partition(policy.partition(raw, [3.]), [3.])


@pytest.mark.parametrize("material", ["Al", "Cu"])
@pytest.mark.parametrize("width", [2 * BUDGET, .1])
def test_above_budget_overlap_remains_a_strict_refusal(material, width):
    raw = hits([(0, 0., 1., "Al", 1), (0, 1. - width, 2., material, 1)])
    adjusted, evidence = policy.coalesce(raw, [2.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(adjusted, raw)
    assert evidence["summary"]["actual_max_shift_mm"] == 0.
    with pytest.raises(ValueError, match="occupancy|cross-material"):
        policy.partition(adjusted, [2.])


def test_long_adjacent_gap_chain_is_rejected_not_split_or_swallowed():
    raw = hits([(0, 0., 1., "Al", 1), (0, 1. + .75 * BUDGET, 3., "Cu", 1),
                (0, 1. + 1.5 * BUDGET, 4., "Fe", 1)])
    with pytest.raises(ValueError, match="diameter.*transitive chain"):
        policy.coalesce(raw, [5.], max_shift_mm=BUDGET)


@pytest.mark.parametrize("sign", [1, -1])
def test_positive_width_thin_source_cannot_collapse_even_for_subtraction(sign):
    raw = hits([(0, 1., 1. + .5 * BUDGET, "Al", sign)])
    with pytest.raises(ValueError, match="collapse.*positive-width"):
        policy.coalesce(raw, [2.], max_shift_mm=BUDGET)
    unchanged, _ = policy.coalesce(raw, [2.], max_shift_mm=0.)
    pd.testing.assert_frame_equal(unchanged, raw)


def test_distinct_domain_anchors_never_coalesce():
    with pytest.raises(ValueError, match="distinct 0/length anchors"):
        policy.coalesce(hits([(0, 0., .5 * BUDGET, "Al", 1)]), [.5 * BUDGET], max_shift_mm=BUDGET)


def test_budget_is_fixed_absolute_even_at_large_coordinates():
    x = 1e12
    edge = np.nextafter(x, -np.inf)
    assert x - edge > BUDGET
    raw = hits([(0, x - 2., x, "Al", 1), (0, edge, x + 2., "Al", 1)])
    adjusted, evidence = policy.coalesce(raw, [x + 4.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(adjusted, raw)
    assert evidence["summary"]["actual_max_shift_mm"] == 0.
    with pytest.raises(ValueError, match="occupancy"):
        policy.partition(adjusted, [x + 4.])


def test_inclusive_budget_and_midpoint_not_mean_of_duplicates():
    budget = 2. ** -20
    raw = hits([(0, 0., 1., "Al", 1), (0, 0., 1., "Al", -1),
                (0, 0., 1., "Al", 1), (0, 1. + budget, 2., "Cu", 1)])
    adjusted, evidence = policy.coalesce(raw, [2.], max_shift_mm=budget)
    midpoint = 1. + budget / 2
    assert adjusted.s1_mm.iloc[:3].eq(midpoint).all()
    assert adjusted.s0_mm.iloc[3] == midpoint
    table = evidence["endpoints"]
    assert table.loc[table.canonical_endpoint_mm == midpoint, "group_size"].eq(4).all()
    check_partition(policy.partition(adjusted, [2.]), [2.])


def test_exact_boundaries_gaps_and_zero_budget():
    raw = hits([(0, 1., 2., "Al", 1), (0, 2., 3., "Al", 1), (0, 4., 5., "Cu", 1)])
    adjusted, evidence = policy.coalesce(raw, [6.], max_shift_mm=0.)
    pd.testing.assert_frame_equal(adjusted, raw)
    assert evidence["summary"]["actual_max_shift_mm"] == 0.
    result = policy.partition(adjusted, [6.])
    check_rows(result, [(0, 0., 1., "vacuum"), (0, 1., 2., "Al"), (0, 2., 3., "Al"),
                        (0, 3., 4., "vacuum"), (0, 4., 5., "Cu"), (0, 5., 6., "vacuum")])
    check_partition(result, [6.])


def test_signed_subtraction_and_material_replacement_match_audit():
    raw = hits([(0, 0., 5., "Al", 1), (0, 1., 3., "Al", -1), (0, 1., 2., "Cu", 1)])
    result = policy.partition(raw, [6.])
    check_rows(result, [(0, 0., 1., "Al"), (0, 1., 2., "Cu"), (0, 2., 3., "vacuum"),
                        (0, 3., 5., "Al"), (0, 5., 6., "vacuum")])
    spans = occupancy(raw)
    active = spans.loc[spans["count"] == 1, COLUMNS[:-1]].sort_values(["lane", "s0_mm"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result.loc[result.material != "vacuum"].reset_index(drop=True), active.astype(policy._DTYPES))
    check_partition(result, [6.])


@pytest.mark.parametrize("rows", [
    [(0, 0., 3., "Al", 1), (0, 1., 2., "Al", 1)],
    [(0, 1., 3., "Al", 1), (0, 1., 3., "Al", 1)],
    [(0, 1., 2., "Al", -1)],
    [(0, 0., 3., "Al", 1), (0, 3., 4., "Al", -1)],
    [(0, 1., 3., "Al", 1), (0, 0., 2., "Al", -1)],
    [(0, 1., 2., "Al", -1), (0, 1., 2., "Cu", 1)],
    [(0, 1., 2., "Al", -1), (0, 1., 2., "Cu", 1), (0, 1., 2., "Cu", 1)],
])
def test_negative_or_multiple_material_occupancy_is_never_clipped(rows):
    raw = hits(rows)
    spans = occupancy(raw)
    assert (~spans["count"].isin([0, 1])).any()
    with pytest.raises(ValueError, match="signed occupancy"):
        policy.partition(raw, [5.])


def test_exact_cancellation_and_zero_width_sources():
    raw = hits([(0, 1., 3., "Al", 1), (0, 1., 3., "Al", -1), (0, 2., 2., "Cu", -1),
                (1, 0., 0., "Al", 1)])
    adjusted, _ = policy.coalesce(raw, [4., 0.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(adjusted, raw)
    result = policy.partition(adjusted, [4., 0.])
    check_rows(result, [(0, 0., 4., "vacuum")])
    check_partition(result, [4., 0.])


@pytest.mark.parametrize("lengths", [[], [0.], [0., 3., 0., 2.], [.5 * BUDGET]])
def test_empty_hits_and_zero_length_lanes(lengths):
    raw = hits()
    adjusted, evidence = policy.coalesce(raw, lengths, max_shift_mm=BUDGET)
    assert adjusted.empty and adjusted.columns.equals(raw.columns)
    assert evidence["endpoints"].empty
    assert evidence["summary"]["actual_max_shift_mm"] == 0.
    assert evidence["summary"]["total_length_change_mm"] == 0.
    check_partition(policy.partition(adjusted, lengths), lengths)


def test_empty_lane_between_hit_lanes_and_equal_coordinates_never_mix():
    raw = hits([(0, 0., 1., "Al", 1), (2, 1. + .5 * BUDGET, 2., "Cu", 1)])
    adjusted, evidence = policy.coalesce(raw, [3., 3., 3., 0.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(adjusted, raw)
    assert evidence["summary"]["actual_max_shift_mm"] == 0.
    check_partition(policy.partition(adjusted, [3., 3., 3., 0.]), [3., 3., 3., 0.])


def test_permutation_and_lane_batch_invariance():
    raw = hits([(0, 0., 1., "Al", 1), (0, 1. + .5 * BUDGET, 3., "Cu", 1),
                (2, .25 * BUDGET, 1., "Fe", 1), (2, 1. - .25 * BUDGET, 3., "Cu", 1)])
    raw["source"] = ["a", "b", "c", "d"]
    lengths = np.array([3., 2., 3., 0.])
    adjusted, evidence = policy.coalesce(raw, lengths, max_shift_mm=BUDGET)
    shuffled, shuffled_evidence = policy.coalesce(raw.sample(frac=1, random_state=4), lengths, max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(adjusted.sort_index(), shuffled.sort_index())
    pd.testing.assert_frame_equal(policy.partition(adjusted, lengths), policy.partition(shuffled, lengths))
    assert evidence["summary"] == shuffled_evidence["summary"]
    identity = ["row_index", "endpoint"]
    columns = [c for c in evidence["endpoints"] if c != "row_position"]
    pd.testing.assert_frame_equal(evidence["endpoints"][columns].sort_values(identity).reset_index(drop=True),
                                  shuffled_evidence["endpoints"][columns].sort_values(identity).reset_index(drop=True))
    batch_hits, batch_partitions, batch_evidence = [], [], []
    for start, stop in [(0, 2), (2, 4)]:
        subset = raw.loc[(raw.lane >= start) & (raw.lane < stop)].copy()
        subset["lane"] -= start
        changed, ev = policy.coalesce(subset, lengths[start:stop], max_shift_mm=BUDGET)
        part = policy.partition(changed, lengths[start:stop])
        changed["lane"] += start
        part["lane"] += start
        ev["endpoints"]["lane"] += start
        batch_hits.append(changed)
        batch_partitions.append(part)
        batch_evidence.append(ev["endpoints"])
    pd.testing.assert_frame_equal(adjusted, pd.concat(batch_hits))
    pd.testing.assert_frame_equal(policy.partition(adjusted, lengths), pd.concat(batch_partitions, ignore_index=True))
    pd.testing.assert_frame_equal(evidence["endpoints"][columns].sort_values(identity).reset_index(drop=True),
                                  pd.concat(batch_evidence)[columns].sort_values(identity).reset_index(drop=True))


@pytest.mark.parametrize("column,value", [("lane", -1), ("lane", 1), ("lane", .5), ("lane", np.nan),
    ("lane", np.inf), ("lane", True), ("lane", "0"), ("sign", 0), ("sign", 2), ("sign", -.5),
    ("sign", np.inf), ("sign", True), ("sign", "1"), ("s0_mm", np.nan), ("s1_mm", np.inf),
    ("s0_mm", -BUDGET / 2), ("s1_mm", 3. + BUDGET / 2), ("s0_mm", 2.5),
    ("s0_mm", "1"), ("s1_mm", complex(2, 1)), ("material", None), ("material", ""), ("material", 3)])
@pytest.mark.parametrize("operation", ["coalesce", "partition"])
def test_invalid_hits_are_rejected_without_tolerance(column, value, operation):
    raw = hits([(0, 1., 2., "Al", 1)])
    raw[column] = value
    kwargs = dict(max_shift_mm=BUDGET) if operation == "coalesce" else {}
    with pytest.raises(ValueError):
        getattr(policy, operation)(raw, [3.], **kwargs)


@pytest.mark.parametrize("lengths", [-1., 3., [[3.]], [-1.], [np.nan], [np.inf], ["3"], [True], []])
@pytest.mark.parametrize("operation", ["coalesce", "partition"])
def test_invalid_lengths_or_lane_domain(lengths, operation):
    raw = hits([(0, 1., 2., "Al", 1)])
    kwargs = dict(max_shift_mm=BUDGET) if operation == "coalesce" else {}
    with pytest.raises(ValueError):
        getattr(policy, operation)(raw, lengths, **kwargs)


@pytest.mark.parametrize("budget", [-BUDGET, np.inf, np.nan, [BUDGET], "1e-9", True, 1j])
def test_invalid_or_implicit_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="max_shift_mm"):
        policy.coalesce(hits(), [], max_shift_mm=budget)


def test_budget_must_be_explicit_and_keyword_only():
    with pytest.raises(TypeError):
        policy.coalesce(hits(), [])
    with pytest.raises(TypeError):
        policy.coalesce(hits(), [], BUDGET)


@pytest.mark.parametrize("raw", [None, [], pd.DataFrame(), pd.DataFrame(columns=COLUMNS[:-1]),
                                 pd.DataFrame(columns=COLUMNS + ["lane"])])
def test_invalid_schema(raw):
    with pytest.raises(ValueError, match="DataFrame|columns"):
        policy.coalesce(raw, [], max_shift_mm=BUDGET)
    with pytest.raises(ValueError, match="DataFrame|columns"):
        policy.partition(raw, [])


def test_above_budget_gap_is_preserved_and_policy_is_idempotent():
    raw = hits([(0, 0., 1., "Al", 1), (0, 1. + 2 * BUDGET, 2., "Cu", 1)])
    adjusted, evidence = policy.coalesce(raw, [2.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(adjusted, raw)
    result = policy.partition(adjusted, [2.])
    assert result.material.tolist() == ["Al", "vacuum", "Cu"]
    assert result.s0_mm.iloc[1] == 1.
    assert result.s1_mm.iloc[1] == 1. + 2 * BUDGET
    again, second = policy.coalesce(adjusted, [2.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(adjusted, again)
    assert evidence["summary"] == second["summary"]
    near = hits([(0, .25 * BUDGET, 1., "Al", 1), (0, 1. - .5 * BUDGET, 2., "Cu", 1)])
    changed, _ = policy.coalesce(near, [2.], max_shift_mm=BUDGET)
    again, second = policy.coalesce(changed, [2.], max_shift_mm=BUDGET)
    pd.testing.assert_frame_equal(changed, again)
    assert second["summary"]["actual_max_shift_mm"] == 0.
    assert second["summary"]["total_length_change_mm"] == 0.


@pytest.mark.parametrize("rows", [
    [(0, 0., 4., "Al", 1), (0, 1., 2., "Cu", 1)],
    [(0, 1., 3., "Al", 1), (0, 1., 3., "Cu", 1)],
    [(0, 0., 4., "Al", 1), (0, 1., 2., "vacuum", 1)],
])
def test_cross_material_overlap_is_not_union_or_density_precedence(rows):
    raw = hits(rows)
    assert occupancy(raw)["count"].isin([0, 1]).all()
    assert (occupancy(raw, by_material=False)["count"] > 1).any()
    with pytest.raises(ValueError, match="cross-material"):
        policy.partition(raw, [5.])


def test_many_lanes_equal_coordinates_are_independent():
    lanes = np.arange(2048)
    raw = pd.DataFrame(dict(lane=np.repeat(lanes, 2), s0_mm=np.tile([0., 1. - .5 * BUDGET], len(lanes)),
                            s1_mm=np.tile([1., 2.], len(lanes)), material="Al", sign=1))
    lengths = np.full(len(lanes), 2.)
    adjusted, evidence = policy.coalesce(raw, lengths, max_shift_mm=BUDGET)
    result = policy.partition(adjusted, lengths)
    check_partition(result, lengths)
    assert len(result) == 2 * len(lanes)
    assert evidence["endpoints"].group_size.max() == 2
    assert evidence["summary"]["changed_endpoints"] == 2 * len(lanes)
