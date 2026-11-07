"""Measured synthetic wires: driver identity, rejection and bounded seed checks."""
import json

import numpy as np
import pytest

from trkfastsim.reco.hitreco import FLAG
from trkfastsim.validation.run1a_seed import REASONS, associate, main, run


def fixture(z=(), angle=None, xy=None, ev=None, times=None):
    z = np.asarray(z, float)
    n = len(z)
    angle = np.asarray(angle if angle is not None else np.arange(n) * np.pi / 2)
    xy = np.asarray(xy if xy is not None else np.tile([10., 20.], (n, 1)))
    d = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n)))
    pos = np.column_stack((xy, z))
    mid = pos - 10 * d
    cond = {"straw": {**dict(zip(("midx", "midy", "midz"), mid.T)),
                      **dict(zip(("dirx", "diry", "dirz"), d.T)), "hlen": np.full(n, 1000.)}}
    sh = dict(ev=np.zeros(n, int) if ev is None else np.asarray(ev, int), sid=np.arange(n),
              flag=np.full(n, 1 << FLAG["tdiv"], np.uint32),
              time=np.zeros(n) if times is None else np.asarray(times, float),
              uvar=np.full(n, 100.), timevar=np.full(n, 4.), **dict(zip(("x", "y", "z"), pos.T)))
    return sh, cond


def keys(n):
    return np.column_stack((np.full(n, 1430), np.zeros(n, int), np.arange(n)))


@pytest.mark.parametrize("ne", [0, 5])
def test_all_empty_census_and_json(ne):
    sh, cond = fixture()
    report, out = run(keys(ne), sh, cond)
    assert report["events"] == ne
    assert report["successful_seeds"] == report["attempted_groups"] == 0
    assert report["zero_hit_events"] == ne
    assert out["candidate_hit_index"].shape == (0, 2)
    assert out["candidate_positions"].shape == (0, 3)
    assert out["seed_center_xy"].shape == (0, 2)
    np.testing.assert_array_equal(out["evkey"], keys(ne))
    np.testing.assert_array_equal(out["event_nhits"], np.zeros(ne))
    json.dumps(report, allow_nan=False)


def test_duplicate_full_keys_rejected_but_same_event_other_run_allowed():
    sh, cond = fixture()
    with pytest.raises(ValueError, match="duplicate full"):
        run(np.array([[1430, 0, 1], [1430, 0, 1]]), sh, cond)
    report, out = run(np.array([[1430, 0, 1], [1431, 0, 1]]), sh, cond)
    assert report["events"] == 2
    assert out["evkey"][1, 0] == 1431


@pytest.mark.parametrize("field", ["x", "time", "uvar", "unused_extra_column"])
def test_nonfinite_hit_rejected(field):
    sh, cond = fixture([0., 10.])
    sh[field] = np.array([np.nan, 0.])
    with pytest.raises(ValueError, match="nonfinite"):
        run(keys(1), sh, cond)


@pytest.mark.parametrize("field,value", [("ev", np.array([0, 1])), ("sid", np.array([0, 96])),
                                           ("flag", np.array([-1, 0]))])
def test_invalid_identity(field, value):
    sh, cond = fixture([0., 10.])
    sh[field] = value
    with pytest.raises(ValueError):
        run(keys(1), sh, cond)


def test_geometry_finite_and_wire_consistency_validation():
    sh, cond = fixture([0., 10.])
    cond["straw"]["dirz"][0] = np.nan
    with pytest.raises(ValueError, match="geometry"):
        run(keys(1), sh, cond)
    cond["straw"]["dirz"][0] = 0.
    sh["z"][0] += 1.
    with pytest.raises(ValueError, match="inconsistent"):
        run(keys(1), sh, cond)


def test_original_source_indices_and_rejected_ambiguous_groups():
    # Rows reordered; two candidate pairs share original row 2, at repeated z=5.
    sh, cond = fixture([0., 0., 10.], angle=[0., np.pi / 2, np.pi / 4], ev=[2, 2, 2])
    permutation = np.array([2, 0, 1])
    sh = {k: v[permutation] for k, v in sh.items()}
    report, out = run(keys(5), sh, cond)
    assert report["candidates"] == 2
    assert report["rejected_ambiguous_groups"] == 1
    assert report["attempted_groups"] == 0
    np.testing.assert_array_equal(out["candidate_hit_index"], [[1, 0], [2, 0]])
    np.testing.assert_array_equal(sh["ev"][out["candidate_hit_index"]], [[2, 2], [2, 2]])
    np.testing.assert_array_equal(out["candidate_group"], [0, 0])
    np.testing.assert_array_equal(out["group_npoints"], [2])
    assert out["group_reject_bits"][0] & REASONS["shared_source_hits"]
    assert out["group_reject_bits"][0] & REASONS["repeated_z"]
    assert not out["candidate_used"].any()
    assert len(out["seed_success"]) == 1
    assert not out["seed_success"].any()
    np.testing.assert_array_equal(out["event_npoints"], [0, 0, 2, 0, 0])


def test_geometry_rejects_whole_original_group_and_preserves_other_mapping():
    sh, cond = fixture([0., 10., 100., 110.], ev=[0, 0, 2, 2])
    # A truly axial wire, not a supported alignment tilt.
    cond["straw"]["dirx"][0] = 0.
    cond["straw"]["dirz"][0] = 1.
    cond["straw"]["midx"][0] = sh["x"][0]
    report, out = run(keys(4), sh, cond)
    assert report["unsupported_geometry_hits"] == 1
    assert report["groups"] == 2
    assert report["group_rejection_counts"]["unsupported_geometry"] == 1
    np.testing.assert_array_equal(out["group_n"], [2, 2])
    np.testing.assert_array_equal(out["hit_group"], [0, 0, 1, 1])
    np.testing.assert_array_equal(out["candidate_hit_index"], [[2, 3]])
    np.testing.assert_array_equal(out["candidate_group"], [1])
    np.testing.assert_array_equal(out["event_nhits"], [2, 0, 2, 0])
    assert report["stereo_supported_groups_only"]["input_hits"] == 2


def test_known_helix_unambiguous_measured_pairs():
    z = np.arange(6) * 100.
    phase = .3 + z / 250.
    xy = np.column_stack((30 + 200 * np.cos(phase), -20 + 200 * np.sin(phase)))
    sh, cond = fixture(np.repeat(z, 2) + np.tile([-5., 5.], 6),
                       xy=np.repeat(xy, 2, axis=0), ev=np.full(12, 2))
    perm = np.array([11, 3, 0, 6, 9, 2, 5, 8, 1, 10, 4, 7])
    sh = {k: v[perm] for k, v in sh.items()}
    report, out = run(keys(5), sh, cond)
    assert report["candidates"] == 6
    assert report["attempted_groups"] == report["successful_seeds"] == 1
    assert report["events_with_seeds"] == 1
    np.testing.assert_array_equal(out["event_nseeds"], [0, 0, 1, 0, 0])
    np.testing.assert_allclose(out["seed_radius"], [200.], rtol=1e-12)
    np.testing.assert_allclose(out["seed_pitch"], [250.], rtol=1e-12)
    np.testing.assert_allclose(out["seed_center_xy"], [[30., -20.]], atol=1e-10)
    np.testing.assert_allclose(out["seed_p_abs"], [.299792458 * 1.0117 * np.hypot(200, 250)])
    # Candidate/source identities still address the input flat rows, not geometry order.
    pair = out["candidate_hit_index"]
    np.testing.assert_array_equal(np.sort(sh["sid"][pair], axis=1), np.arange(12).reshape(6, 2))
    np.testing.assert_allclose(out["candidate_positions"][:, 2], z)
    assert out["candidate_used"].all()


def test_repeated_z_without_shared_hits_and_bound_no_truncation():
    n = 257
    points = dict(group=np.r_[np.zeros(4, int), np.ones(n, int)],
                  positions=np.column_stack((np.zeros(n + 4), np.zeros(n + 4), np.r_[0, 0, 1, 2, np.arange(n)])),
                  hit_index=np.arange(2 * (n + 4)).reshape(-1, 2))
    bits, used = associate(points, np.array([4, n]), 2 * (n + 4), np.zeros(2, int))
    np.testing.assert_array_equal(bits, [REASONS["repeated_z"], REASONS["over_256_points"]])
    assert not used.any()
    assert len(used) == n + 4


def test_flag_rejection_preserves_source_and_event_census():
    sh, cond = fixture([0., 10.], ev=[1, 1])
    sh["flag"][:] = 0
    report, out = run(keys(4), sh, cond)
    assert report["rejected_flag_hits"] == 2
    assert report["groups"] == 0
    np.testing.assert_array_equal(out["hit_group"], [-1, -1])
    np.testing.assert_array_equal(out["event_nhits"], [0, 2, 0, 0])


@pytest.mark.parametrize("bz", [0., np.nan, np.inf])
def test_bad_field_even_if_empty(bz):
    sh, cond = fixture()
    with pytest.raises(ValueError):
        run(keys(2), sh, cond, bz=bz)


def test_cli_safe_output_and_no_overwrite(tmp_path):
    sh, cond = fixture([0., 10.])
    np.savez(tmp_path / "hits.npz", evkey=keys(3), **{"s_" + k: v for k, v in sh.items()})
    raw = {"straw": {k: v.tolist() for k, v in cond["straw"].items()},
           **{k: [] for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror")}}
    (tmp_path / "conditions.json").write_text(json.dumps(raw))
    argv = ["--hits", str(tmp_path / "hits.npz"), "--conditions", str(tmp_path / "conditions.json"),
            "--outdir", str(tmp_path / "output"), "--bz", "1.0117"]
    report = main(argv)
    text = (tmp_path / "output" / "report.json").read_text()
    def reject_constant(value):
        raise AssertionError(f"nonfinite JSON number {value}")
    assert json.loads(text, parse_constant=reject_constant) == report
    with np.load(tmp_path / "output" / "diagnostics.npz", allow_pickle=False) as out:
        assert len(out["evkey"]) == 3
        assert len(out["seed_success"]) == 1
        assert np.isnan(out["seed_radius"]).all()
    with pytest.raises(FileExistsError):
        main(argv)


def test_uint64_identity_columns_are_safe_after_validation():
    sh, cond = fixture([0., 10.], ev=[2, 2])
    for k in ("ev", "sid", "flag"):
        sh[k] = sh[k].astype(np.uint64)
    report, out = run(keys(4).astype(np.uint64), sh, cond)
    assert report["candidates"] == 1
    np.testing.assert_array_equal(out["event_nhits"], [0, 0, 2, 0])


def test_wide_time_group_retained_without_pairs():
    sh, cond = fixture(np.arange(8) * 10., times=np.arange(8) * 15.)
    report, out = run(keys(2), sh, cond)
    assert report["groups"] == 1
    assert out["group_span"][0] == 105.
    assert out["group_reject_bits"][0] & REASONS["wide_time"]
    assert out["group_candidates_evaluated"][0]
    assert report["candidates"] == 0
    assert len(out["member"]) == 8


def test_geometry_rejected_ambiguity_is_explicitly_unevaluated():
    sh, cond = fixture([0., 10.])
    cond["straw"]["dirx"][0] = 0.
    cond["straw"]["dirz"][0] = 1.
    cond["straw"]["midx"][0] = sh["x"][0]
    report, out = run(keys(3), sh, cond)
    assert report["groups_candidates_evaluated"] == 0
    assert report["groups_ambiguity_unevaluated"] == 1
    assert report["candidate_count_histogram_evaluated"] == {}
    np.testing.assert_array_equal(out["group_reject_bits"], [REASONS["unsupported_geometry"]])
    assert not out["group_candidates_evaluated"].any()


def test_chunked_fit_preserves_rejected_lane_and_candidate_order():
    from trkfastsim.validation.run1a_seed import fit_groups
    ng = 1027
    g = np.tile(np.arange(ng), 4)
    phase = np.repeat(np.linspace(.2, 1.2, 4), ng)
    points = dict(group=g, positions=np.column_stack((100 * np.cos(phase),
                                                      100 * np.sin(phase), 200 * phase)))
    bits = np.zeros(ng, int)
    bits[512] = REASONS["shared_source_hits"]
    result = fit_groups(points, np.full(ng, 4), bits, 1.0117)
    np.testing.assert_array_equal(result["success"], bits == 0)
    np.testing.assert_allclose(result["radius"][bits == 0], 100., rtol=1e-12)
    np.testing.assert_allclose(result["pitch"][bits == 0], 200., rtol=1e-12)
    assert result["n_hits"][512] == 0
    assert np.isnan(result["radius"][512])


@pytest.mark.parametrize("tilt", [.001, .8])
def test_tilted_geometry_evaluated_without_flattening(tilt):
    sh, cond = fixture([0., 10.])
    d = np.array([np.sqrt(1 - tilt**2), 0., tilt])
    crossing = np.array([sh[k][0] for k in ("x", "y", "z")])
    for k, v in zip(("dirx", "diry", "dirz"), d):
        cond["straw"][k][0] = v
    for k, v in zip(("midx", "midy", "midz"), crossing - 10 * d):
        cond["straw"][k][0] = v
    for k, v in zip(("x", "y", "z"), crossing + 4 * d):
        sh[k][0] = v
    report, out = run(keys(3), sh, cond)
    assert report["unsupported_geometry_hits"] == 0
    assert report["groups_candidates_evaluated"] == 1
    assert report["groups_ambiguity_unevaluated"] == 0
    assert report["candidates"] == 1
    np.testing.assert_allclose(out["candidate_positions"], [[10., 20., 5.]], atol=1e-12)
    np.testing.assert_allclose(out["candidate_wire_pull"], [[-.4, 0.]], atol=1e-12)
    np.testing.assert_array_equal(out["group_reject_bits"], [REASONS["fewer_than_four"]])
    assert out["hit_geometry_supported"].all()


def test_alignment_tilts_do_not_resolve_shared_hit_ambiguity():
    sh, cond = fixture([0., 0., 10.], angle=[0., np.pi / 2, np.pi / 4])
    cond["straw"]["dirz"][:] = .001
    cond["straw"]["midz"] -= .01
    report, out = run(keys(2), sh, cond)
    assert report["groups_candidates_evaluated"] == 1
    assert report["candidates"] == 2
    assert report["group_rejection_counts"]["shared_source_hits"] == 1
    assert report["attempted_groups"] == report["successful_seeds"] == 0
    assert not out["candidate_used"].any()


def test_near_axial_projection_retains_group_rejection_diagnostic():
    sh, cond = fixture([0., 10.])
    cond["straw"]["dirx"][0] = 1e-12
    cond["straw"]["dirz"][0] = 1.
    cond["straw"]["midx"][0] = sh["x"][0]
    report, out = run(keys(2), sh, cond)
    assert report["unsupported_geometry_hits"] == 1
    assert report["groups_ambiguity_unevaluated"] == 1
    np.testing.assert_array_equal(out["group_reject_bits"], [REASONS["unsupported_geometry"]])
    np.testing.assert_array_equal(out["hit_geometry_supported"], [False, True])
