"""Saved-membership wire-fit driver provenance and complete-census tests."""
import json

import numpy as np
import pytest

from trkfastsim.reco.helixwire import wire_residuals
from trkfastsim.reco.hitreco import FLAG
from trkfastsim.validation.run1a_wirefit import prepare, run, main


def fixture(empty=False):
    n = 0 if empty else 12
    z = np.linspace(-200, 200, n)
    phase = .4 + z / 250
    pos = np.column_stack((30 + 200*np.cos(phase), -20 + 200*np.sin(phase), z))
    a = np.arange(n) * 1.1
    d = np.column_stack((np.cos(a), np.sin(a), np.full(n, .001)))
    d /= np.linalg.norm(d, axis=1)[:, None]
    mid = pos - 10*d
    sh = dict(ev=np.full(n, 2, int), sid=np.arange(n), time=np.zeros(n), uvar=np.full(n, 100.),
              timevar=np.ones(n), flag=np.full(n, 1 << FLAG["tdiv"], np.uint32),
              **dict(zip(("x", "y", "z"), pos.T)))
    cond = {"straw": {**dict(zip(("midx", "midy", "midz"), mid.T)),
                       **dict(zip(("dirx", "diry", "dirz"), d.T)), "hlen": np.full(n, 1000.)}}
    keys = np.column_stack((np.full(4, 1430), np.zeros(4, int), np.arange(4)))
    seed = dict(success=np.array([False, not empty, False]),
                center_xy=np.array([[np.nan, np.nan], [32., -21.], [np.nan, np.nan]]),
                radius=np.array([np.nan, 199., np.nan]), pitch=np.array([np.nan, 252., np.nan]),
                phase_ref=np.array([np.nan, .4, np.nan]), z_ref=np.array([np.nan, 0., np.nan]))
    residuals = wire_residuals({k: v[1:2] for k, v in seed.items()}, mid, d,
                               np.full(n, 10.), np.full(n, 100.), np.full(n, 1000.))
    saved = dict(evkey=keys.copy(), source_hit_ev=sh["ev"].copy(), source_hit_sid=sh["sid"].copy(),
                 hit_group=np.ones(n, int), group_ev=np.array([0, 2, 3]),
                 association_source_used=np.ones(n, bool), association_group_support=np.array([0, n, 0]))
    saved.update({"associated_seed_" + k: v for k, v in seed.items()})
    saved.update({"association_source_" + k: residuals[field][0] for k, field in
                  (("doca", "doca"), ("wire_pull", "wire_pull"), ("poca_z", "z"))})
    return keys, sh, cond, saved


def test_fixed_support_improves_and_preserves_census():
    keys, sh, cond, saved = fixture()
    report, out = run(keys, sh, cond, saved)
    assert report["events"] == 4
    assert report["zero_hit_events"] == 3
    assert report["input_seed_groups"] == report["successful_fits"] == 1
    assert report["fixed_selected_hits"] == 12
    np.testing.assert_array_equal(out["evkey"], keys)
    np.testing.assert_array_equal(out["fit_success"], [False, True, False])
    np.testing.assert_array_equal(out["event_nfits"], [0, 0, 1, 0])
    assert out["fit_final_objective"][1] < out["fit_initial_objective"][1]
    np.testing.assert_allclose(out["fit_radius"][1], 200., atol=.001)
    np.testing.assert_allclose(out["fit_pitch"][1], 250., atol=.001)
    assert out["source_selected"].all()
    assert out["source_poca_converged"].all()
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("batch", [1, 2, 128])
def test_group_chunking_preserves_failed_lanes_and_outputs(batch):
    args = fixture()
    _, baseline = run(*args)
    _, out = run(*args, group_batch=batch)
    for k in baseline:
        np.testing.assert_array_equal(out[k], baseline[k], err_msg=k)


def test_all_empty_and_no_group():
    keys, sh, cond, saved = fixture(empty=True)
    report, out = run(keys, sh, cond, saved)
    assert report["events"] == report["zero_hit_events"] == 4
    assert report["successful_fits"] == 0
    assert out["fit_success"].shape == (3,)
    assert out["source_doca"].shape == (0,)
    for k in list(saved):
        if k.startswith("associated_seed_") or k in ("group_ev", "association_group_support"):
            saved[k] = saved[k][:0]
    report, out = run(keys, sh, cond, saved)
    assert report["groups"] == 0
    assert out["fit_success"].shape == (0,)
    np.testing.assert_array_equal(out["event_nfits"], [0, 0, 0, 0])


@pytest.mark.parametrize("field", ["evkey", "source_hit_ev", "source_hit_sid"])
def test_changed_census_rejected(field):
    keys, sh, cond, saved = fixture()
    saved[field] = saved[field] + 1
    with pytest.raises(ValueError, match="census"):
        run(keys, sh, cond, saved)


@pytest.mark.parametrize("field,value", [("association_group_support", [0, 11, 0]),
                                          ("group_ev", [0, 1, 3]),
                                          ("associated_seed_success", [False, False, False])])
def test_changed_grouping_or_seed_support_rejected(field, value):
    keys, sh, cond, saved = fixture()
    saved[field] = np.asarray(value)
    with pytest.raises(ValueError):
        run(keys, sh, cond, saved)


@pytest.mark.parametrize("field", ["doca", "wire_pull", "poca_z"])
def test_saved_residual_mismatch_rejected(field):
    keys, sh, cond, saved = fixture()
    saved["association_source_" + field][0] += .1
    with pytest.raises(ValueError, match="saved support"):
        run(keys, sh, cond, saved)


def test_duplicate_straw_support_rejected():
    keys, sh, cond, saved = fixture()
    sh["sid"][1] = sh["sid"][0]
    saved["source_hit_sid"] = sh["sid"].copy()
    with pytest.raises(ValueError, match="distinct-straw"):
        prepare(keys, sh, cond, saved)


def test_fixed_membership_not_dropped_on_exhaustion():
    args = fixture()
    report, out = run(*args, max_iterations=0)
    assert report["input_seed_groups"] == 1
    assert report["successful_fits"] == 0
    assert out["source_selected"].sum() == 12
    assert out["fit_n_hits"][1] == 12
    assert out["fit_status"][1] == "max_iterations"
    assert np.isnan(out["fit_p_abs"]).all()


@pytest.mark.parametrize("settings", [{"group_batch": 0}, {"group_batch": True}, {"max_cells": 11}, {"bz": 0}])
def test_bounds_and_invalid_settings(settings):
    with pytest.raises(ValueError):
        run(*fixture(), **settings)


def test_cli_inputs_hash_and_no_overwrite(tmp_path):
    keys, sh, cond, saved = fixture()
    np.savez(tmp_path / "hits.npz", evkey=keys, **{"s_" + k: v for k, v in sh.items()})
    np.savez(tmp_path / "association.npz", **saved)
    raw = {"straw": {k: v.tolist() for k, v in cond["straw"].items()},
           **{k: [] for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror")}}
    (tmp_path / "conditions.json").write_text(json.dumps(raw))
    args = ["--hits", str(tmp_path / "hits.npz"), "--conditions", str(tmp_path / "conditions.json"),
            "--association", str(tmp_path / "association.npz")]
    report = main(args + ["--outdir", str(tmp_path / "first")])
    assert report["successful_fits"] == 1
    assert all(len(v["sha256"]) == 64 for v in report["inputs"].values())
    with pytest.raises(FileExistsError):
        main(args + ["--outdir", str(tmp_path / "first")])
    report = main(args + ["--outdir", str(tmp_path / "repeat"), "--group-batch", "1",
                          "--compare-saved", str(tmp_path / "first" / "diagnostics.npz")])
    assert report["saved_comparison"]["exact"]


def test_alternate_local_seed_cannot_bypass_replayed_seed():
    keys, sh, cond, saved = fixture()
    saved["associated_seed_local_parameters"] = np.zeros((3, 5))
    with pytest.raises(ValueError, match="alternate saved seed"):
        run(keys, sh, cond, saved)


def test_uint64_support_normalized_to_safe_indices():
    keys, sh, cond, saved = fixture()
    report, expected = run(keys, sh, cond, saved)
    saved["association_group_support"] = saved["association_group_support"].astype(np.uint64)
    _, out = run(keys, sh, cond, saved)
    for k in expected:
        np.testing.assert_array_equal(out[k], expected[k])


def test_residual_replay_not_claimed_as_complete_original_input_hash():
    keys, sh, cond, saved = fixture()
    sh["time"] += 1e6
    cond["straw"]["hlen"][:] = 50.
    report, out = run(keys, sh, cond, saved)
    assert report["selected_seed_residuals_replayed"]
    assert report["original_association_inputs_hash_bound"] is False
    assert "not equality" in report["provenance_caveat"]


def test_uint64_event_straw_and_group_indices():
    keys, sh, cond, saved = fixture()
    for k in ("ev", "sid", "flag"):
        sh[k] = sh[k].astype(np.uint64)
    for k in ("hit_group", "group_ev", "association_group_support"):
        saved[k] = saved[k].astype(np.uint64)
    report, out = run(keys.astype(np.uint64), sh, cond, saved)
    assert report["successful_fits"] == 1
    np.testing.assert_array_equal(out["event_nfits"], [0, 0, 1, 0])
