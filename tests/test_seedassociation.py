"""Analytic trajectory association checks; no stock tracks or simulation truth input."""
import itertools
import json
import math

import numpy as np
import pytest

from trkfastsim.reco.seedassociation import _wire_votes, associate_wires


def fixture(n=6):
    phase = np.linspace(.25, 1.75, n)
    pos = np.column_stack((30 + 200 * np.cos(phase), -20 + 200 * np.sin(phase), 250 * phase))
    hp = np.repeat(pos, 2, axis=0)
    a = np.arange(2 * n) * 1.1
    direction = np.column_stack((np.cos(a), np.sin(a), np.full(2 * n, .02)))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    mid = hp - 10 * direction
    cond = {"straw": {**dict(zip(("midx", "midy", "midz"), mid.T)),
                      **dict(zip(("dirx", "diry", "dirz"), direction.T)), "hlen": np.full(2 * n, 1000.)}}
    sh = dict(ev=np.zeros(2 * n, int), sid=np.arange(2 * n),
              uvar=np.full(2 * n, 100.), **dict(zip(("x", "y", "z"), hp.T)))
    points = dict(positions=pos, group=np.zeros(n, int), hit_index=np.arange(2 * n).reshape(n, 2))
    return points, sh, cond, np.zeros(2 * n, int)


def call(points, sh, cond, hg, **kw):
    return associate_wires(points, sh, cond, hit_group=hg, ngroups=3, bz=1.0117, **kw)


def test_analytic_curve_and_full_group_census():
    args = fixture()
    out = call(*args)
    np.testing.assert_array_equal(out["seed"]["success"], [True, False, False])
    np.testing.assert_array_equal(out["group_hypotheses"], [15, 0, 0])
    np.testing.assert_array_equal(out["group_independent"], [15, 0, 0])
    np.testing.assert_array_equal(out["group_support"], [12, 0, 0])
    np.testing.assert_allclose(out["seed"]["radius"][0], 200., atol=1e-8)
    np.testing.assert_allclose(out["seed"]["pitch"][0], 250., atol=1e-8)
    np.testing.assert_allclose(out["seed"]["center_xy"][0], [30, -20], atol=1e-8)
    assert out["source_used"].all()
    np.testing.assert_allclose(out["source_doca"], 0., atol=1e-7)
    assert np.isnan(out["seed"]["p_abs"][1:]).all()
    assert out["runner_support"][0] == 12


@pytest.mark.parametrize("batch", [1, 2, 7, 1024])
def test_batch_invariance_and_all_combinations(batch):
    args = fixture(7)
    reference = call(*args)
    test = call(*args, hypothesis_batch=batch, max_score_cells=14 * batch)
    for k in reference:
        if k == "seed":
            for name in reference[k]:
                np.testing.assert_allclose(reference[k][name], test[k][name], rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_allclose(reference[k], test[k], rtol=1e-12, atol=1e-12)
    assert test["group_hypotheses"][0] == math.comb(7, 4)


def test_shared_alternatives_scored_not_blanket_rejected():
    points, sh, cond, hg = fixture()
    points["positions"] = np.r_[points["positions"], points["positions"][:3] + [100., -75., 15.]]
    points["hit_index"] = np.r_[points["hit_index"], [[0, 3], [2, 5], [4, 7]]]
    points["group"] = np.zeros(9, int)
    out = call(points, sh, cond, hg)
    assert out["group_hypotheses"][0] == math.comb(9, 4)
    assert 0 < out["group_independent"][0] < math.comb(9, 4)
    assert out["group_support"][0] == 12
    assert out["seed"]["success"][0]
    assert np.all(out["group_candidate"][0] < 6)
    np.testing.assert_allclose(out["seed"]["radius"][0], 200., atol=1e-8)
    expected = sum(len(np.unique(points["hit_index"][list(c)])) == 8
                   for c in itertools.combinations(range(9), 4))
    assert out["group_independent"][0] == expected


def test_insufficient_independent_support_not_fabricated():
    points, sh, cond, hg = fixture(4)
    points["hit_index"][:, 0] = 0
    out = call(points, sh, cond, hg)
    assert out["group_hypotheses"][0] == 1
    assert out["group_independent"].sum() == 0
    assert not out["seed"]["success"].any()
    assert not out["source_used"].any()


def test_physical_straw_duplicates_never_count_as_independent():
    points, sh, cond, hg = fixture(4)
    sh["sid"][1] = sh["sid"][0]
    for key in ("x", "y", "z"):
        sh[key][1] = sh[key][0]
    out = call(points, sh, cond, hg)
    assert out["group_independent"].sum() == 0
    assert not out["seed"]["success"].any()


def test_one_vote_per_straw_minimum_cost_and_first_tie():
    inlier = np.array([[True, True, True, False, True], [True, False, True, True, False]])
    cost = np.array([[.4, .1, .1, 0, .9], [0, 0, .8, .3, 0]])
    sid = np.array([[1, 1, 1, 2, 2], [0, 1, 1, 1, 2]])
    np.testing.assert_array_equal(_wire_votes(inlier, cost, sid),
                                  [[False, True, False, False, True], [True, False, False, True, False]])


def test_outlier_wire_is_not_selected_and_seed_not_refitted():
    points, sh, cond, hg = fixture()
    n = len(sh["sid"])
    for key in sh:
        sh[key] = np.r_[sh[key], sh[key][-1]]
    sh["sid"][-1] = n
    sh["x"][-1] += 500.
    for key in cond["straw"]:
        cond["straw"][key] = np.r_[cond["straw"][key], cond["straw"][key][-1]]
    cond["straw"]["midx"][-1] += 500.
    out = call(points, sh, cond, np.r_[hg, 0])
    assert out["group_support"][0] == n
    assert not out["source_used"][-1]
    assert np.isnan(out["source_doca"][-1])


def test_measured_longitudinal_incompatibility_rejects_wires():
    points, sh, cond, hg = fixture(4)
    for axis in ("x", "y", "z"):
        sh[axis] += 150 * cond["straw"]["dir" + axis]
    out = call(points, sh, cond, hg)
    assert out["group_fit_success"][0] == 1
    assert not out["seed"]["success"].any()


def test_finite_wire_length_required():
    points, sh, cond, hg = fixture(4)
    cond["straw"]["hlen"][:] = 5.
    out = call(points, sh, cond, hg)
    assert out["group_fit_success"][0] == 1
    assert not out["seed"]["success"].any()


def test_repeated_candidate_z_fails_without_jitter():
    points, sh, cond, hg = fixture(4)
    points["positions"][1, 2] = points["positions"][0, 2]
    out = call(points, sh, cond, hg)
    assert out["group_independent"][0] == 1
    assert out["group_fit_success"][0] == 0


def test_group_and_candidate_permutation_and_original_indices():
    points, sh, cond, hg = fixture(6)
    reference = call(points, sh, cond, hg)
    points = {k: v[[3, 5, 0, 1, 4, 2]] for k, v in points.items()}
    hp = np.array([9, 5, 2, 4, 0, 7, 8, 6, 1, 3, 11, 10])
    inverse = np.argsort(hp)
    sh = {k: v[hp] for k, v in sh.items()}
    points["hit_index"] = inverse[points["hit_index"]]
    out = call(points, sh, cond, hg[hp])
    np.testing.assert_allclose(out["seed"]["p_abs"], reference["seed"]["p_abs"], rtol=1e-10)
    np.testing.assert_array_equal(np.sort(sh["sid"][out["source_used"]]), np.arange(12))
    selected = points["hit_index"][out["group_candidate"][0]]
    assert len(np.unique(sh["sid"][selected])) == 8


@pytest.mark.parametrize("kw", [{"max_hypotheses": 14}, {"max_score_cells": 11}])
def test_resource_limits_raise_without_truncation(kw):
    with pytest.raises(ValueError, match="no .*truncated"):
        call(*fixture(), **kw)


@pytest.mark.parametrize("kw", [{"max_hypotheses": True}, {"hypothesis_batch": 0},
                                {"max_score_cells": False}, {"min_hits": 7},
                                {"max_doca": 0}, {"max_wire_pull": np.nan}])
def test_malformed_settings(kw):
    with pytest.raises(ValueError):
        call(*fixture(), **kw)


@pytest.mark.parametrize("problem", ["source", "group", "duplicate", "nan", "cross_event", "negative_sid"])
def test_invalid_candidate_or_hit_provenance(problem):
    points, sh, cond, hg = fixture()
    if problem == "source":
        points["hit_index"][0, 0] = 99
    elif problem == "group":
        points["group"][0] = 2
    elif problem == "duplicate":
        points["hit_index"][0] = points["hit_index"][1]
    elif problem == "nan":
        points["positions"][0, 0] = np.nan
    elif problem == "cross_event":
        sh["ev"][0] = 1
    else:
        sh["sid"][0] = -1
    with pytest.raises(ValueError):
        call(points, sh, cond, hg)


def test_empty_input_and_zero_groups():
    points, sh, cond, hg = fixture(0)
    out = associate_wires(points, sh, cond, hit_group=hg, ngroups=0, bz=1.0117)
    assert out["seed"]["success"].shape == (0,)
    assert out["group_candidate"].shape == (0, 4)
    assert out["source_used"].shape == (0,)


def test_no_input_mutation():
    points, sh, cond, hg = fixture()
    pc = {k: v.copy() for k, v in points.items()}
    hc = {k: v.copy() for k, v in sh.items()}
    gc = {k: v.copy() for k, v in cond["straw"].items()}
    call(points, sh, cond, hg)
    for k in pc:
        np.testing.assert_array_equal(points[k], pc[k])
    for k in hc:
        np.testing.assert_array_equal(sh[k], hc[k])
    for k in gc:
        np.testing.assert_array_equal(cond["straw"][k], gc[k])


def test_validation_driver_preserves_empty_events_and_safe_json():
    from trkfastsim.validation.run1a_association import run
    points, sh, cond, hg = fixture(0)
    sh.update(time=np.empty(0), timevar=np.empty(0), flag=np.empty(0, np.uint32))
    keys = np.array([[1430, 0, 1], [1430, 0, 2]])
    report, payload = run(keys, sh, cond)
    assert report["events"] == report["zero_hit_events"] == 2
    assert report["successful_seeds"] == 0
    np.testing.assert_array_equal(payload["evkey"], keys)
    np.testing.assert_array_equal(payload["event_associated_seeds"], [0, 0])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("ev", [-1, np.uint64(2**64 - 1)])
def test_negative_or_unsigned_overflow_event_index(ev):
    points, sh, cond, hg = fixture()
    sh["ev"] = np.full(len(hg), ev)
    with pytest.raises(ValueError, match="negative event"):
        call(points, sh, cond, hg)


def test_derived_geometry_overflow_rejected():
    points, sh, cond, hg = fixture()
    sh["x"][:] = 1e308
    cond["straw"]["midx"][:] = -1e308
    with pytest.raises(ValueError, match="inconsistent"):
        call(points, sh, cond, hg)


def test_incremental_ranking_storage_bound(monkeypatch):
    import trkfastsim.reco.seedassociation as module
    original = module._first_two
    sizes = []
    def recording(group, count, score, rank):
        sizes.append(len(group))
        return original(group, count, score, rank)
    monkeypatch.setattr(module, "_first_two", recording)
    out = call(*fixture(10), hypothesis_batch=1)
    assert out["group_supported"][0] == math.comb(10, 4)
    assert max(sizes) <= 3


def test_multiple_groups_duplicate_pulses_rejected_hits_and_permutations():
    from trkfastsim.reco.hitreco import FLAG
    from trkfastsim.validation.run1a_association import run
    n = 6
    z = np.arange(n) * 100.
    phi = .3 + z / 250.
    xy = np.column_stack((30 + 200 * np.cos(phi), -20 + 200 * np.sin(phi)))
    pos = np.column_stack((np.repeat(xy, 2, axis=0), np.repeat(z, 2) + np.tile([-5., 5.], n)))
    angle = np.tile([0., np.pi / 2], n)
    direction = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(2 * n)))
    mid = pos - 10 * direction
    cond = {"straw": {**dict(zip(("midx", "midy", "midz"), mid.T)),
                      **dict(zip(("dirx", "diry", "dirz"), direction.T)), "hlen": np.full(2 * n, 1000.)}}
    sid = np.r_[np.tile(np.arange(12), 3), 0, 0]
    ev = np.r_[np.repeat([1, 1, 3], 12), 1, 1]
    times = np.r_[np.repeat([0., 200., 0.], 12), 0., 0.]
    sh = dict(ev=ev, sid=sid, time=times, timevar=np.full(len(sid), 4.),
              uvar=np.full(len(sid), 100.), flag=np.full(len(sid), 1 << FLAG["tdiv"], np.uint32),
              **dict(zip(("x", "y", "z"), pos[sid].T)))
    sh["flag"][-1] = 0
    keys = np.column_stack((np.full(5, 1430), np.zeros(5, int), np.arange(5)))
    report, payload = run(keys, sh, cond, hypothesis_batch=np.int64(2), max_doca=np.array(10.))
    json.dumps(report, allow_nan=False)
    assert report["groups"] == report["successful_seeds"] == 3
    assert report["zero_hit_events"] == 3
    np.testing.assert_array_equal(payload["event_associated_seeds"], [0, 2, 0, 1, 0])
    np.testing.assert_array_equal(payload["association_group_support"], [12, 12, 12])
    used = payload["association_source_used"]
    assert not used[-2:].any()
    assert payload["hit_group"][-1] == -1
    np.testing.assert_array_equal(np.bincount(payload["hit_group"][used], minlength=3), [12, 12, 12])
    assert len(np.unique(np.column_stack((payload["hit_group"][used], sid[used])), axis=0)) == used.sum()
    permutation = np.random.default_rng(26).permutation(len(sid))
    _, repeat = run(keys, {k: v[permutation] for k, v in sh.items()}, cond, hypothesis_batch=1)
    np.testing.assert_allclose(repeat["associated_seed_p_abs"], payload["associated_seed_p_abs"])
    np.testing.assert_array_equal(repeat["association_group_support"], payload["association_group_support"])
    for tag in ("group_candidate", "runner_candidate"):
        a = payload["candidate_hit_index"][payload["association_" + tag]]
        b = repeat["candidate_hit_index"][repeat["association_" + tag]]
        np.testing.assert_array_equal(np.sort(sid[a], axis=-1), np.sort(sid[permutation][b], axis=-1))
    np.testing.assert_array_equal(repeat["evkey"], keys)


def test_two_tracks_in_one_time_group_still_returns_only_one_seed():
    points, sh, cond, hg = fixture(6)
    points["positions"] = np.r_[points["positions"], points["positions"] + [1500., 0., 0.]]
    points["hit_index"] = np.r_[points["hit_index"], points["hit_index"] + 12]
    points["group"] = np.zeros(12, int)
    for k, v in tuple(sh.items()):
        sh[k] = np.r_[v, v + (12 if k == "sid" else 1500. if k == "x" else 0)]
    for k, v in tuple(cond["straw"].items()):
        cond["straw"][k] = np.r_[v, v + (1500. if k == "midx" else 0)]
    out = call(points, sh, cond, np.zeros(24, int))
    assert out["seed"]["success"].sum() == 1
    assert out["group_support"][0] == 12
    used = np.flatnonzero(out["source_used"])
    assert np.all(used < 12) or np.all(used >= 12)


@pytest.mark.parametrize("tamper", [False, True])
def test_cli_saved_payload_audit_and_no_overwrite(tmp_path, tamper):
    from trkfastsim.validation.run1a_association import main
    points, sh, cond, hg = fixture(0)
    sh.update(time=np.empty(0), timevar=np.empty(0), flag=np.empty(0, np.uint32))
    np.savez(tmp_path / "hits.npz", evkey=np.array([[1430, 0, 1]]), **{"s_" + k: v for k, v in sh.items()})
    raw = {"straw": {k: v.tolist() for k, v in cond["straw"].items()},
           **{k: [] for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror")}}
    (tmp_path / "conditions.json").write_text(json.dumps(raw))
    args = ["--hits", str(tmp_path / "hits.npz"), "--conditions", str(tmp_path / "conditions.json")]
    main(args + ["--outdir", str(tmp_path / "first")])
    with pytest.raises(FileExistsError):
        main(args + ["--outdir", str(tmp_path / "first")])
    previous = tmp_path / "first" / "diagnostics.npz"
    if tamper:
        with np.load(previous) as data:
            payload = dict(data)
        payload["event_associated_seeds"][0] = 1
        previous = tmp_path / "tampered.npz"
        np.savez(previous, **payload)
        with pytest.raises(RuntimeError, match="comparison differs"):
            main(args + ["--outdir", str(tmp_path / "repeat"), "--compare-saved", str(previous)])
        assert not (tmp_path / "repeat").exists()
    else:
        report = main(args + ["--outdir", str(tmp_path / "repeat"), "--compare-saved", str(previous)])
        assert report["saved_comparison"]["exact"]
        assert report["saved_comparison"]["different_arrays"] == []


def test_unequal_group_sizes_have_bit_exact_batch_reductions():
    points, sh, cond, hg = fixture(6)
    points["positions"] = np.r_[points["positions"], points["positions"][:4]]
    points["hit_index"] = np.r_[points["hit_index"], points["hit_index"][:4] + 12]
    points["group"] = np.r_[np.zeros(6, int), np.ones(4, int)]
    for k, v in tuple(sh.items()):
        sh[k] = np.r_[v, v[:8] + (1 if k == "ev" else 0)]
    hg = np.r_[hg, np.ones(8, int)]
    ref = call(points, sh, cond, hg)
    test = call(points, sh, cond, hg, hypothesis_batch=1)
    for k in ref:
        if k == "seed":
            for name in ref[k]:
                np.testing.assert_array_equal(ref[k][name], test[k][name])
        else:
            np.testing.assert_array_equal(ref[k], test[k])
