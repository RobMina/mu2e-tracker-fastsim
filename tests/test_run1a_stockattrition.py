"""Saved-stage diagnosis only: no association, optimizer, or stock-conditioned recovery.

Priority is explaining the saved three missing-seed/seven nonpassing stock groups,
not retuning them. Synthetic fixtures below are independent of those observed counts.
Python loops here provide small independent census/counting oracles.
"""
from collections import Counter
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import run1a_stockattrition as attrition
from trkfastsim.validation import run1a_stockcompare as comparison
from test_run1a_stockcompare import FLAGS, _fast, _save_lr, _sha, _stock, _succeed


STAGES = ("unsupported_geometry", "wide_time_group", "fewer_than_four_candidates",
          "no_independent_quartet", "no_fitted_hypothesis", "no_supported_hypothesis",
          "supported", "supported", "supported", "supported", "supported")
STRING_LANES = ("certificate_status", "termination_status", "last_refit_status",
                "last_refit_termination_status")
COUNT_LANES = ("pending_lr_flips", "accepted_lr_rounds", "continuous_refits")


def _complete_lanes(fast):
    ng = len(fast["group_ev"])
    for name in STRING_LANES:
        fast["lr_" + name] = np.full((ng, 2), "not_evaluated", dtype="U40")
    fast["lr_analytic_differentiable"] = np.zeros((ng, 2), dtype=bool)
    for name in COUNT_LANES:
        fast["lr_" + name] = np.zeros((ng, 2), dtype=np.int64)
    return fast


def _association(fast, counts):
    ng, nh = len(fast["group_ev"]), len(fast["hit_group"])
    association = {k: fast[k].copy() for k in
                   ("evkey", "source_hit_ev", "source_hit_sid", "hit_group", "group_ev")}
    members = [np.flatnonzero(fast["hit_group"] == g) for g in range(ng)]
    pairs, groups = [], []
    for g, n in enumerate(counts):
        # Four independent pairs, then an alternative sharing original source hits.
        options = list(zip(members[g][::2], members[g][1::2]))
        if len(members[g]) >= 3:
            options.append((members[g][0], members[g][2]))
        pairs.extend(options[:n])
        groups.extend([g] * n)
    cg = np.asarray(groups, dtype=np.int64)
    association.update(
        association_source_used=fast["source_selected"].copy(),
        association_group_support=fast["group_support"].copy(),
        group_n=np.array([len(m) for m in members], dtype=np.int64),
        group_npoints=np.asarray(counts, dtype=np.int64),
        candidate_group=cg,
        candidate_ev=fast["group_ev"][cg],
        candidate_hit_index=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        candidate_positions=np.zeros((len(cg), 3), dtype=float),
        hit_geometry_supported=np.ones(nh, dtype=bool),
        group_candidates_evaluated=np.ones(ng, dtype=bool),
        group_span=np.full(ng, 100., dtype=float),
        group_reject_bits=np.full(ng, 8 | 16, dtype=np.int64),
        candidate_used=np.zeros(len(cg), dtype=bool),
        association_group_hypotheses=np.array([math.comb(n, 4) if n >= 4 else 0
                                               for n in counts], dtype=np.int64))
    for name in ("independent", "fit_success", "supported"):
        association["association_group_" + name] = np.zeros(ng, dtype=np.int64)
    return association


@pytest.fixture
def census():
    # Keys sharing the event number must not merge. e3 is a zero-hit event;
    # e4 has ungrouped hits but no time group. Stock omits e2, not e1.
    keys = [(10, 1, 7), (11, 1, 7), (10, 2, 7), (10, 1, 8), (10, 1, 9)]
    gev = [0] * 8 + [1, 2, 0]
    hits = [(ev, s, g, g >= 6) for g, ev in enumerate(gev) for s in range(8)]
    hits.extend([(0, 0, 3, False), (4, 90, -1, False), (4, 90, -1, False)])
    fast = _complete_lanes(_fast(keys, gev, hits))
    for g, lane in ((7, 0), (7, 1), (8, 1), (9, 0)):
        _succeed(fast, g, lane)
    fast["lr_refit_numerical_failure"][6, 0] = True
    fast["lr_status"][6, 0] = "numerical_failure"
    fast["lr_lr_stationary"][6, 1] = False
    fast["lr_status"][6, 1] = "nonstationary"
    fast["lr_pending_lr_flips"][6, 1] = 2
    fast["lr_certified"][6] = True  # continuous-only is NOT combined pass
    fast["lr_analytic_stationary"][6] = True
    fast["lr_analytic_differentiable"][6:] = True
    fast["lr_refit_error"][10, 0] = True
    fast["lr_status"][10, 0] = "refit_error"
    fast["lr_refit_numerical_failure"][10, 1] = True
    fast["lr_status"][10, 1] = "numerical_failure"
    fast["lr_accepted_lr_rounds"][6:] = [1, 2]
    fast["lr_continuous_refits"][6:] = [1, 2]
    fast["lr_certificate_status"][fast["lr_certified"]] = "adjacent_stencil_checked"
    fast["lr_termination_status"][6:] = ["iteration_limit", "stationary"]
    fast["lr_last_refit_status"][6:] = ["invalid", "converged"]
    fast["lr_last_refit_termination_status"][6:] = ["roundoff", "stationary"]
    fast["lr_objective"][6] = [np.inf, np.nan]
    association = _association(fast, [0, 0, 3, 4, 4, 5, 5, 4, 4, 4, 4])
    association["hit_geometry_supported"][0] = False
    association["group_candidates_evaluated"][0] = False
    association["group_span"][:2] = [101., 101.]
    association["association_group_independent"][4:] = [1, 3, 5, 1, 1, 1, 1]
    association["association_group_fit_success"][5:] = [2, 3, 1, 1, 1, 1]
    association["association_group_supported"][6:] = [2, 1, 1, 1, 1]
    # Group 3 genuinely has no independent quartet (shared original source hit).
    row = np.flatnonzero(association["candidate_group"] == 3)[-1]
    association["candidate_hit_index"][row] = [24, 30]
    stock = _stock([keys[3], keys[1], keys[0], keys[4], (99, 1, 7)],
                   [(2, 1, 2, 0), (2, 2, 2, 1), (1, -1, 2, 1), (4, 1, 2, 1)],
                   [(0, 0, 1), (0, 0, -1), (1, 7, 1), (2, 1, 0)])
    _, saved = comparison.overlap(fast, stock)
    return saved, association


def _assert_arrays(actual, expected):
    assert set(actual) == set(expected)
    for k, v in expected.items():
        assert actual[k].dtype == v.dtype, k
        np.testing.assert_array_equal(actual[k], v, err_msg=k)


def _hist(values):
    return dict(Counter(str(v) for v in values))


def _assert_oracle(fast, association, stages):
    report, out, table = attrition.diagnose(fast, association)
    ng = len(fast["group_ev"])
    assert len(table) == ng
    np.testing.assert_array_equal(table.group, np.arange(ng))
    np.testing.assert_array_equal(table.association_stage, stages)
    for g, ev in enumerate(fast["group_ev"]):
        members = [h for h, group in enumerate(fast["hit_group"]) if group == g]
        candidate_rows = [i for i, group in enumerate(association["candidate_group"]) if group == g]
        represented = {int(h) for i in candidate_rows for h in association["candidate_hit_index"][i]}
        row = table.iloc[g]
        assert tuple(row[["run", "subrun", "event"]]) == tuple(fast["evkey"][ev])
        assert row.members == len(members)
        assert row.distinct_member_straws == len({int(fast["source_hit_sid"][h]) for h in members})
        assert row.stereo_candidates == len(candidate_rows)
        assert row.candidate_source_hits == len(represented)
        assert row.fixed_support == sum(bool(fast["source_selected"][h]) for h in members)
        assert row.group_hypotheses == (math.comb(len(candidate_rows), 4) if len(candidate_rows) >= 4 else 0)
        for name in attrition.COUNTERS:
            assert row[name] == association["association_" + name][g]
        for lane, label in enumerate(("negative_z", "positive_z")):
            for name in attrition.LANES:
                assert row[label + "_" + name] == fast["lr_" + name][g, lane]
    selections = dict(all_groups=list(range(ng)))
    selections["no_support"] = [g for g in range(ng) if fast["group_support"][g] == 0]
    selections["supported_nonpassing"] = [g for g in range(ng) if fast["group_support"][g] > 0
                                           and not any(fast["lr_success"][g])]
    selections["recorded_ch"] = [g for g, ev in enumerate(fast["group_ev"])
                                 if fast["comparison_event_stock_ch_positive_status"][ev] > 0]
    selections["recorded_ch_nonpassing"] = [g for g in selections["recorded_ch"]
                                            if not any(fast["lr_success"][g])]
    assert set(report["populations"]) == set(selections)
    for name, groups in selections.items():
        expected = dict(groups=len(groups),
                        groups_any_combined_pass=sum(any(fast["lr_success"][g]) for g in groups),
                        groups_any_continuous_pass=sum(any(fast["lr_certified"][g]) for g in groups),
                        association_stages=_hist(stages[g] for g in groups),
                        final_lane_status=_hist(fast["lr_status"][g, lane] for g in groups for lane in (0, 1)),
                        certificate_status=_hist(fast["lr_certificate_status"][g, lane]
                                                 for g in groups for lane in (0, 1)))
        for output, flag in (("continuous_checked", "certified"), ("combined_pass", "success"),
                             ("sticky_numerical_failure", "refit_numerical_failure")):
            expected[output] = sum(int(fast["lr_" + flag][g, lane]) for g in groups for lane in (0, 1))
        assert report["populations"][name] == expected
    _assert_arrays({k: out[k] for k in fast}, fast)
    assert {k for k in out if k not in fast} == {"attrition_" + k for k in table.columns}
    for k in table.columns:
        np.testing.assert_array_equal(out["attrition_" + k], table[k].to_numpy())
    json.dumps(report, allow_nan=False)
    return report, out, table


def test_complete_census_categories_nested_counts_and_both_lr_lanes(census):
    report, out, table = _assert_oracle(*census, STAGES)
    assert len(out["evkey"]) == 5 and len(out["hit_group"]) == 91
    assert report["populations"]["all_groups"]["groups"] == 11
    assert report["populations"]["no_support"]["groups"] == 6
    assert report["populations"]["supported_nonpassing"]["groups"] == 2
    assert report["populations"]["recorded_ch"]["groups"] == 9
    assert report["populations"]["recorded_ch_nonpassing"]["groups"] == 8
    assert report["populations"]["all_groups"]["combined_pass"] == 4
    assert report["populations"]["all_groups"]["sticky_numerical_failure"] == 2
    # Every same-event group survives, including failed, unsupported and passing.
    np.testing.assert_array_equal(table.loc[table.stock_positive_status_ch > 0, "group"],
                                  [0, 1, 2, 3, 4, 5, 6, 7, 10])
    assert table.loc[8, "stock_event_present"] and table.loc[8, "stock_positive_status_ch"] == 0
    assert not table.loc[9, "stock_event_present"] and table.loc[9, "stock_positive_status_ch"] == -1
    assert not table.loc[6, "any_combined_pass"] and table.loc[7, "any_combined_pass"]
    assert table.loc[3, "members"] == 9 and table.loc[3, "distinct_member_straws"] == 8
    assert table.loc[6, "stereo_candidates"] == 5 and table.loc[6, "candidate_source_hits"] == 8
    assert "NOT association vetoes" in report["control_rejection_bits"]


def test_read_only_inputs_and_independent_preserved_arrays(census):
    snapshots = [{k: v.copy() for k, v in source.items()} for source in census]
    for source in census:
        for v in source.values():
            v.flags.writeable = False
    _, out, _ = _assert_oracle(*census, STAGES)
    for k in census[0]:
        assert not np.shares_memory(out[k], census[0][k]), k
    out["source_hit_sid"][0] = 5
    out["lr_status"][6, 0] = "changed"
    out["stock_hit_sid"][0] = 5
    for source, snapshot in zip(census, snapshots):
        _assert_arrays(source, snapshot)
        assert all(not v.flags.writeable for v in source.values())


@pytest.mark.parametrize("bits", [0, 8, 16, 8 | 16, 63, 2**31])
def test_control_rejection_bits_and_candidate_used_are_not_association_vetoes(census, bits):
    fast, association = census
    original, out, table = attrition.diagnose(fast, association)
    association["group_reject_bits"][:] = bits
    association["candidate_used"][:] = True
    current, new, frame = attrition.diagnose(fast, association)
    assert current == original
    _assert_arrays(new, out)
    pd.testing.assert_frame_equal(frame, table)


def test_candidate_order_and_endpoint_order_do_not_change_recounts(census):
    fast, association = census
    expected, out, table = attrition.diagnose(fast, association)
    permutation = np.random.default_rng(107).permutation(len(association["candidate_group"]))
    for name in ("candidate_group", "candidate_ev", "candidate_hit_index", "candidate_positions", "candidate_used"):
        association[name] = association[name][permutation]
    association["candidate_hit_index"] = association["candidate_hit_index"][:, ::-1]
    report, result, frame = attrition.diagnose(fast, association)
    assert report == expected
    _assert_arrays(result, out)
    pd.testing.assert_frame_equal(frame, table)


@pytest.mark.parametrize("events,groups,hits,stock_present", [
    (0, 0, 0, False), (2, 0, 0, False), (2, 0, 2, True),
    (2, 2, 0, False), (2, 2, 2, True),
])
def test_empty_censuses_zero_support_and_ungrouped_hits(events, groups, hits, stock_present):
    keys = [(1, 0, i) for i in range(events)]
    fast = _complete_lanes(_fast(keys, [0] * groups, [(0, 1, -1, False)] * hits))
    association = _association(fast, [0] * groups)
    _, saved = comparison.overlap(fast, _stock(keys if stock_present else []))
    report, out, table = _assert_oracle(saved, association, ["fewer_than_four_candidates"] * groups)
    assert report["populations"]["all_groups"]["combined_pass"] == 0
    assert len(out["evkey"]) == events and len(out["source_hit_sid"]) == hits
    assert len(table) == groups


@pytest.mark.parametrize("field", ["evkey", "source_hit_ev", "source_hit_sid", "hit_group", "group_ev"])
@pytest.mark.parametrize("problem", ["value", "dtype", "shape", "missing"])
def test_association_identity_must_match_exactly(census, field, problem):
    fast, association = census
    if problem == "missing":
        del association[field]
    elif problem == "dtype":
        association[field] = association[field].astype(float)
    elif problem == "shape":
        association[field] = association[field][:-1]
    else:
        association[field].flat[0] += 1
    with pytest.raises((ValueError, KeyError)):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("field", ["association_source_used", "association_group_support"])
@pytest.mark.parametrize("problem", ["value", "shape", "dtype", "missing"])
def test_association_support_contract(census, field, problem):
    fast, association = census
    if problem == "missing":
        del association[field]
    elif problem == "dtype":
        association[field] = association[field].astype(int if field.endswith("used") else float)
    elif problem == "shape":
        association[field] = association[field][:-1]
    else:
        association[field].flat[0] = 1
    with pytest.raises((ValueError, KeyError)):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("field", ["group_n", "group_npoints", "candidate_group", "candidate_hit_index"] +
                         ["association_" + name for name in attrition.COUNTERS])
@pytest.mark.parametrize("problem", ["negative", "float", "shape", "masked", "overflow"])
def test_integer_recounts_reject_malformed_values(census, field, problem):
    fast, association = census
    if problem == "negative":
        association[field].flat[0] = -1
    elif problem == "float":
        association[field] = association[field].astype(float)
    elif problem == "shape":
        association[field] = association[field][..., None]
    elif problem == "masked":
        association[field] = np.ma.array(association[field], mask=False)
        association[field].mask.flat[0] = True
    else:
        association[field] = association[field].astype(np.uint64)
        association[field].flat[0] = np.uint64(2**63)
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("problem", ["members", "candidate_count", "quartets", "independent", "fitted",
                                     "supported", "winner_absent", "winner_unreported", "candidate_limit",
                                     "candidate_group_bound", "source_bound", "self_pair", "wrong_group",
                                     "duplicate_pair", "reversed_duplicate_pair"])
def test_inconsistent_recounts_nested_counts_and_candidate_provenance(census, problem):
    fast, association = census
    key, row, value = {
        "members": ("group_n", 3, 8), "candidate_count": ("group_npoints", 2, 4),
        "quartets": ("association_group_hypotheses", 5, 4),
        "independent": ("association_group_independent", 3, 2),
        "fitted": ("association_group_fit_success", 4, 2),
        "supported": ("association_group_supported", 5, 3),
        "winner_absent": ("association_group_supported", 5, 1),
        "winner_unreported": ("association_group_supported", 6, 0),
        "candidate_limit": ("group_npoints", 2, 257),
        "candidate_group_bound": ("candidate_group", 0, 11),
        "source_bound": ("candidate_hit_index", (0, 0), len(fast["hit_group"])),
        "self_pair": ("candidate_hit_index", (0, 1), 16),
        "wrong_group": ("candidate_hit_index", (0, 0), 0),
        "duplicate_pair": ("candidate_hit_index", 1, [16, 17]),
        "reversed_duplicate_pair": ("candidate_hit_index", 1, [17, 16]),
    }[problem]
    association[key][row] = value
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("field", ["candidate_positions", "group_span"])
@pytest.mark.parametrize("problem", ["nan", "infinite", "integer", "shape", "masked"])
def test_geometry_and_span_contracts(census, field, problem):
    fast, association = census
    if problem in ("nan", "infinite"):
        association[field].flat[0] = np.nan if problem == "nan" else np.inf
    elif problem == "integer":
        association[field] = association[field].astype(int)
    elif problem == "shape":
        association[field] = association[field][..., None]
    else:
        association[field] = np.ma.array(association[field], mask=False)
        association[field].mask.flat[0] = True
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("problem", ["negative_span", "wide_candidates", "unsupported_candidates", "evaluated",
                                     "geometry_dtype", "evaluated_dtype"])
def test_candidate_eligibility_is_whole_group_geometry_and_fixed_span(census, problem):
    fast, association = census
    if problem == "negative_span":
        association["group_span"][0] = -1.
    elif problem == "wide_candidates":
        association["group_span"][2] = np.nextafter(100., np.inf)
    elif problem == "unsupported_candidates":
        association["hit_geometry_supported"][16] = False
        association["group_candidates_evaluated"][2] = False
    elif problem == "evaluated":
        association["group_candidates_evaluated"][0] = True
    else:
        field = "hit_geometry_supported" if problem == "geometry_dtype" else "group_candidates_evaluated"
        association[field] = association[field].astype(int)
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("field", FLAGS)
def test_combined_lr_contradictions_rejected(census, field):
    census[0]["lr_" + field][7, 0] = not census[0]["lr_" + field][7, 0]
    with pytest.raises(ValueError):
        attrition.diagnose(*census)


@pytest.mark.parametrize("field", attrition.LANES)
@pytest.mark.parametrize("problem", ["shape", "object", "masked", "missing"])
def test_every_lr_lane_is_required_unmasked_and_two_directional(census, field, problem):
    fast, association = census
    key = "lr_" + field
    if problem == "missing":
        del fast[key]
    elif problem == "shape":
        fast[key] = fast[key][:, :1]
    elif problem == "object":
        fast[key] = fast[key].astype(object)
    else:
        fast[key] = np.ma.array(fast[key], mask=False)
        fast[key].mask[0, 0] = True
    with pytest.raises((ValueError, KeyError)):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("field", STRING_LANES + COUNT_LANES + ("analytic_differentiable",))
def test_lr_diagnostic_lanes_have_semantic_dtypes(census, field):
    key = "lr_" + field
    census[0][key] = census[0][key].astype(bool if field in STRING_LANES else str)
    with pytest.raises(ValueError):
        attrition.diagnose(*census)


@pytest.mark.parametrize("field", COUNT_LANES)
def test_lr_work_and_pending_counts_cannot_be_negative(census, field):
    census[0]["lr_" + field][0, 0] = -1
    with pytest.raises(ValueError):
        attrition.diagnose(*census)


@pytest.mark.parametrize("problem", ["selected_integer", "selected_ungrouped", "mask_false", "mask_integer",
                                     "mask_shape", "duplicate_support", "support_recount", "subminimum_support",
                                     "invalid_empty_objective", "status", "masked_opaque"])
def test_fixed_support_masks_and_saved_fast_contracts(census, problem):
    fast, association = census
    if problem == "selected_integer":
        fast["source_selected"] = fast["source_selected"].astype(int)
    elif problem == "selected_ungrouped":
        fast["source_selected"][-1] = True
    elif problem == "mask_false":
        fast["source_lr_mask"][48, 0] = False
    elif problem == "mask_integer":
        fast["source_lr_mask"] = fast["source_lr_mask"].astype(int)
    elif problem == "mask_shape":
        fast["source_lr_mask"] = fast["source_lr_mask"][:, :1]
    elif problem == "duplicate_support":
        fast["source_hit_sid"][49] = fast["source_hit_sid"][48]
    elif problem == "support_recount":
        fast["group_support"][6] += 1
    elif problem == "subminimum_support":
        fast["source_selected"][48] = False
        fast["source_lr_mask"][48] = False
        fast["group_support"][6] -= 1
        association["association_source_used"][48] = False
        association["association_group_support"][6] -= 1
    elif problem == "invalid_empty_objective":
        fast["lr_objective_valid"][0, 0] = True
    elif problem == "status":
        fast["lr_status"][7, 0] = "failed"
    else:
        fast["lr_objective"] = np.ma.array(fast["lr_objective"], mask=False)
        fast["lr_objective"].mask[6, 1] = True
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("problem", ["presence_integer", "presence_shape", "counts_float", "counts_shape",
                                     "unknown_zero", "covered_unknown", "less_than_unknown"])
def test_stock_coverage_unknown_is_not_zero(census, problem):
    fast, association = census
    presence, counts = "comparison_event_stock_present", "comparison_event_stock_ch_positive_status"
    if problem == "presence_integer":
        fast[presence] = fast[presence].astype(int)
    elif problem == "presence_shape":
        fast[presence] = fast[presence][:, None]
    elif problem == "counts_float":
        fast[counts] = fast[counts].astype(float)
    elif problem == "counts_shape":
        fast[counts] = fast[counts][:, None]
    elif problem == "unknown_zero":
        fast[counts][2] = 0
    elif problem == "covered_unknown":
        fast[counts][1] = -1
    else:
        fast[counts][2] = -2
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)


def _cli(tmp_path, census):
    saved, association = census
    original = {k: v.copy() for k, v in saved.items()
                if not k.startswith(("stock_", "comparison_"))}
    lrdir, lr_report = _save_lr(tmp_path, original)
    assocpath = tmp_path / "association.npz"
    np.savez(assocpath, **association)
    lr_report["inputs"]["association"] = dict(path=str(assocpath), sha256=_sha(assocpath))
    (lrdir / "report.json").write_text(json.dumps(lr_report))
    stockpath = tmp_path / "stock.root"
    stockpath.write_bytes(b"recorded census already copied; ROOT is never reopened")
    directory = tmp_path / "comparison"
    directory.mkdir()
    payload = directory / "diagnostics.npz"
    np.savez(payload, **saved)
    paths = dict(lr_report=lrdir / "report.json", lr_payload=lrdir / "diagnostics.npz", stock=stockpath)
    report = dict(output_payload_sha256=_sha(payload),
                  inputs={k: dict(path=str(p), sha256=_sha(p)) for k, p in paths.items()})
    (directory / "report.json").write_text(json.dumps(report))
    return ["--comparison", str(directory)], directory, report


def _rehash_comparison(directory, report):
    report["output_payload_sha256"] = _sha(directory / "diagnostics.npz")
    (directory / "report.json").write_text(json.dumps(report))


def test_cli_hashes_copy_structural_replay_repeat_and_no_overwrite(tmp_path, monkeypatch, census, capsys):
    argv, directory, _ = _cli(tmp_path, census)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    calls = []
    overlap = comparison.overlap

    def replay_only(fast, stock, **kwargs):
        calls.append("structural replay")
        return overlap(fast, stock, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("saved attrition must not rerun association, optimization, or ROOT reading")

    from trkfastsim.reco import seedassociation, lrdriftfit, stabledriftfit
    from trkfastsim.validation import run1a_lrdrift
    monkeypatch.setattr(comparison, "overlap", replay_only)
    monkeypatch.setattr(comparison.stockcensus, "read_stock", forbidden)
    monkeypatch.setattr(seedassociation, "associate_wires", forbidden)
    monkeypatch.setattr(lrdriftfit, "fit_lr_drift_helix", forbidden)
    monkeypatch.setattr(stabledriftfit, "fit_stable_drift_helix", forbidden)
    monkeypatch.setattr(run1a_lrdrift, "main", forbidden)
    first = tmp_path / "first"
    report = attrition.main(argv + ["--outdir", str(first)])
    assert json.loads(capsys.readouterr().out) == report
    assert json.loads((first / "report.json").read_text()) == report
    assert report["output_payload_sha256"] == _sha(first / "diagnostics.npz")
    assert report["implementation_sha256"] == _sha(Path(attrition.__file__))
    for category in ("inputs", "implementation"):
        for info in report[category].values():
            assert info["sha256"] == _sha(Path(info["path"]))
    expected_report, expected, table = attrition.diagnose(*census)
    assert report["populations"] == expected_report["populations"]
    with np.load(first / "diagnostics.npz", allow_pickle=False) as actual:
        _assert_arrays(dict(actual), expected)
    csv = pd.read_csv(first / "groups.csv")
    pd.testing.assert_frame_equal(csv, table, check_dtype=False)
    assert [row["group"] for row in report["recorded_ch_groups"]] == [0, 1, 2, 3, 4, 5, 6, 7, 10]
    saved = {p: p.read_bytes() for p in first.iterdir()}
    with pytest.raises(FileExistsError):
        attrition.main(argv + ["--outdir", str(first)])
    assert calls == ["structural replay"]
    assert all(p.read_bytes() == contents for p, contents in saved.items())
    second = tmp_path / "second"
    replay = attrition.main(argv + ["--outdir", str(second), "--compare-saved", str(first / "diagnostics.npz")])
    assert json.loads(capsys.readouterr().out) == replay
    assert replay["saved_comparison"] == dict(exact=True, sha256=_sha(first / "diagnostics.npz"))
    assert {k: v for k, v in replay.items() if k not in ("saved_comparison", "output_payload_sha256")} == {
        k: v for k, v in report.items() if k != "output_payload_sha256"}
    with np.load(second / "diagnostics.npz", allow_pickle=False) as actual:
        _assert_arrays(dict(actual), expected)
    assert (first / "groups.csv").read_bytes() == (second / "groups.csv").read_bytes()
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert calls == ["structural replay"] * 2


@pytest.mark.parametrize("problem", ["payload", "missing_payload_hash", "lr_report", "lr_payload", "stock", "association"])
def test_cli_hash_tampering_fails_before_creating_output(tmp_path, census, problem):
    argv, directory, report = _cli(tmp_path, census)
    if problem == "missing_payload_hash":
        del report["output_payload_sha256"]
        (directory / "report.json").write_text(json.dumps(report))
    else:
        path = (directory / "diagnostics.npz" if problem == "payload" else
                tmp_path / "association.npz" if problem == "association" else
                Path(report["inputs"][problem]["path"]))
        path.write_bytes(b"tampered")
    destination = tmp_path / "never-created"
    with pytest.raises(ValueError, match="hash"):
        attrition.main(argv + ["--outdir", str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize("problem", ["lr_failed_lane", "lr_missing", "pair_value", "pair_dtype", "pair_shape",
                                     "extra", "missing", "stock_count", "stock_identity"])
def test_cli_rehashed_comparison_tampering_fails_structural_replay(tmp_path, census, problem):
    argv, directory, report = _cli(tmp_path, census)
    payload = {k: v.copy() for k, v in census[0].items()}
    if problem == "lr_failed_lane":
        payload["lr_objective"][6, 0] = 123.
    elif problem == "lr_missing":
        del payload["lr_objective"]
    elif problem == "pair_value":
        payload["comparison_pair_shared_straws"][0] += 1
    elif problem == "pair_dtype":
        payload["comparison_pair_shared_straws"] = payload["comparison_pair_shared_straws"].astype(float)
    elif problem == "pair_shape":
        payload["comparison_pair_group"] = payload["comparison_pair_group"][:, None]
    elif problem == "extra":
        payload["unexpected"] = np.array([0])
    elif problem == "missing":
        del payload["comparison_pair_group"]
    elif problem == "stock_count":
        payload["comparison_event_stock_ch_positive_status"][0] += 1
    else:
        payload["stock_evkey"][2, 0] += 100
    np.savez(directory / "diagnostics.npz", **payload)
    _rehash_comparison(directory, report)
    destination = tmp_path / "never-created"
    with pytest.raises(ValueError, match="changed saved LR field|table replay mismatch"):
        attrition.main(argv + ["--outdir", str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize("problem", ["value", "missing", "extra", "shape", "dtype", "failed_lane"])
def test_cli_exact_repeat_rejects_changed_payload(tmp_path, census, problem):
    argv, _, _ = _cli(tmp_path, census)
    _, saved, _ = attrition.diagnose(*census)
    if problem == "value":
        saved["attrition_members"][0] += 1
    elif problem == "missing":
        del saved["source_time"]
    elif problem == "extra":
        saved["extra"] = np.array([0])
    elif problem == "shape":
        saved["attrition_group"] = saved["attrition_group"][:, None]
    elif problem == "dtype":
        saved["attrition_members"] = saved["attrition_members"].astype(float)
    else:
        saved["lr_objective"][6, 1] = 0.
    previous = tmp_path / "previous.npz"
    np.savez(previous, **saved)
    destination = tmp_path / "never-created"
    with pytest.raises(ValueError, match="attrition repeat differs"):
        attrition.main(argv + ["--outdir", str(destination), "--compare-saved", str(previous)])
    assert not destination.exists()


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 256])
def test_candidate_combinatorial_boundary_without_enumerating_hypotheses(count):
    fast = _complete_lanes(_fast([(1, 2, 3)], [0], [(0, s, 0, False) for s in range(24)]))
    association = _association(fast, [0])
    pairs = np.asarray(list(itertools.combinations(range(24), 2))[:count], dtype=np.int64).reshape(-1, 2)
    association.update(candidate_group=np.zeros(count, dtype=np.int64),
                       candidate_ev=np.zeros(count, dtype=np.int64),
                       candidate_hit_index=pairs, candidate_positions=np.zeros((count, 3)),
                       candidate_used=np.zeros(count, dtype=bool),
                       group_npoints=np.array([count], dtype=np.int64),
                       association_group_hypotheses=np.array([math.comb(count, 4) if count >= 4 else 0], dtype=np.int64))
    _, saved = comparison.overlap(fast, _stock([(1, 2, 3)]))
    stage = "fewer_than_four_candidates" if count < 4 else "no_independent_quartet"
    _, out, _ = _assert_oracle(saved, association, [stage])
    assert out["attrition_group_hypotheses"][0] == (math.comb(count, 4) if count >= 4 else 0)


@pytest.mark.parametrize("problem", ["certificate_unknown", "certificate_flag", "certified_nondifferentiable",
                                     "certified_nonstationary", "certified_invalid_objective", "round_refit_count"])
def test_certificate_prerequisites_and_refit_accounting(census, problem):
    fast, association = census
    if problem == "certificate_unknown":
        fast["lr_certificate_status"][0, 0] = "arbitrary"
    elif problem == "certificate_flag":
        fast["lr_certificate_status"][7, 0] = "not_evaluated"
    elif problem == "certified_nondifferentiable":
        fast["lr_analytic_differentiable"][6, 0] = False
    elif problem == "certified_nonstationary":
        fast["lr_analytic_stationary"][6, 0] = False
    elif problem == "certified_invalid_objective":
        fast["lr_objective_valid"][6, 0] = False
    else:
        fast["lr_continuous_refits"][6, 0] += 1
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize("field", ["association_source_used", "hit_geometry_supported", "group_candidates_evaluated",
                                   "candidate_used", "group_reject_bits"])
def test_association_masks_cannot_hide_missing_values_even_in_control_diagnostics(census, field):
    fast, association = census
    association[field] = np.ma.array(association[field], mask=False)
    association[field].mask.flat[0] = True
    with pytest.raises(ValueError, match="masked"):
        attrition.diagnose(fast, association)


def test_cli_existing_output_rejected_before_any_input_read(tmp_path, monkeypatch):
    destination = tmp_path / "exists"
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"unchanged")

    def forbidden(*args, **kwargs):
        pytest.fail("existing destination must be rejected before input IO")

    monkeypatch.setattr(attrition, "_load", forbidden)
    monkeypatch.setattr(attrition, "_sha256", forbidden)
    with pytest.raises(FileExistsError):
        attrition.main(["--comparison", str(tmp_path / "missing-input"), "--outdir", str(destination)])
    assert list(destination.iterdir()) == [destination / "sentinel"]
    assert (destination / "sentinel").read_bytes() == b"unchanged"


@pytest.mark.parametrize("problem", ["identity", "candidate_count", "candidate_pair", "source_mask", "support", "nested_count"])
def test_cli_rehashed_association_is_still_recounted(tmp_path, census, problem):
    argv, directory, report = _cli(tmp_path, census)
    association = {k: v.copy() for k, v in census[1].items()}
    if problem == "identity":
        association["source_hit_sid"][0] += 1
    elif problem == "candidate_count":
        association["group_npoints"][2] += 1
    elif problem == "candidate_pair":
        association["candidate_hit_index"][0, 0] = 0
    elif problem == "source_mask":
        association["association_source_used"] = association["association_source_used"].astype(int)
    elif problem == "support":
        association["association_group_support"][6] += 1
    else:
        association["association_group_fit_success"][4] = 2
    assocpath = tmp_path / "association.npz"
    np.savez(assocpath, **association)
    lrpath = Path(report["inputs"]["lr_report"]["path"])
    lr_report = json.loads(lrpath.read_text())
    lr_report["inputs"]["association"]["sha256"] = _sha(assocpath)
    lrpath.write_text(json.dumps(lr_report))
    report["inputs"]["lr_report"]["sha256"] = _sha(lrpath)
    _rehash_comparison(directory, report)
    destination = tmp_path / "never-created"
    with pytest.raises(ValueError):
        attrition.main(argv + ["--outdir", str(destination)])
    assert not destination.exists()


def test_candidate_event_contradiction_rejected(census):
    fast, association = census
    association['candidate_ev'][0] = (association['candidate_ev'][0] + 1) % len(fast['evkey'])
    with pytest.raises(ValueError, match='candidate event'):
        attrition.diagnose(fast, association)


@pytest.mark.parametrize('name', ['pending_lr_flips', 'accepted_lr_rounds', 'continuous_refits'])
def test_lane_work_count_uint64_overflow_rejected(census, name):
    fast, association = census
    a = fast['lr_'+name].astype(np.uint64)
    a[0, 0] = np.uint64(2**63)
    fast['lr_'+name] = a
    with pytest.raises(ValueError):
        attrition.diagnose(fast, association)
