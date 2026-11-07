"""Synthetic geometry only: candidate construction is not tracking validation."""
import copy

import numpy as np
import pytest

from trkfastsim.reco.hitreco import FLAG
from trkfastsim.reco.stereoseed import stereo_points


def fixture(z=(0., 10.), angle=(0., np.pi / 2), ev=None, time=None):
    z, angle = np.asarray(z, float), np.asarray(angle, float)
    n = len(z)
    d = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n)))
    crossing = np.column_stack((np.full(n, 10.), np.full(n, 20.), z))
    mid = crossing - 10 * d
    cond = {"straw": {**dict(zip(("midx", "midy", "midz"), mid.T)),
                      **dict(zip(("dirx", "diry", "dirz"), d.T)), "hlen": np.full(n, 100.)}}
    sh = dict(ev=np.zeros(n, int) if ev is None else np.asarray(ev, int), sid=np.arange(n),
              flag=np.full(n, 1 << FLAG["tdiv"], np.uint32),
              time=np.zeros(n) if time is None else np.asarray(time, float),
              uvar=np.full(n, 100.), timevar=np.full(n, 4.), **dict(zip(("x", "y", "z"), crossing.T)))
    return sh, cond


def check_accounting(out):
    diag = out["diagnostics"]
    rejected = sum(v for k, v in diag.items() if k.startswith("rejected_") and k != "rejected_flags")
    assert rejected + diag["accepted"] == diag["pairs_considered"]
    assert out["group_npairs"].sum() == diag["pairs_considered"]
    assert out["event_nhits"].shape == (out["nevents"],)
    assert out["group_npoints"].sum() == len(out["positions"])
    assert out["event_npoints"].sum() == len(out["positions"])


def test_known_intersection_original_identity_and_census():
    sh, cond = fixture(ev=[2, 2], time=[11, 10])
    sh["wdist"] = np.array([9999., -9999.])  # not used; measured xyz supplies wire distance
    sh["udx"] = np.zeros(2)  # geometry must come from conditions, not hit direction
    out = stereo_points(sh, cond, nevents=5)
    np.testing.assert_allclose(out["positions"], [[10., 20., 5.]])
    np.testing.assert_allclose(out["wire_pull"], 0, atol=1e-14)
    assert out["hit_index"].tolist() == [[1, 0]]
    assert out["ev"].tolist() == [2]
    assert out["member"].tolist() == [1, 0]
    assert out["member_group"].tolist() == [0, 0]
    assert out["hit_group"].tolist() == [0, 0]
    assert out["event_nhits"].tolist() == [0, 0, 2, 0, 0]
    assert out["event_ngroups"].tolist() == [0, 0, 1, 0, 0]
    assert out["event_npoints"].tolist() == [0, 0, 1, 0, 0]
    check_accounting(out)


@pytest.mark.parametrize("angles", [(0., np.pi / 3), (0.2, 1.8), (np.pi, np.pi / 2)])
def test_oblique_and_reversed_geometry(angles):
    sh, cond = fixture(angle=angles)
    # Non-unit directions are normalized; lengths/variances remain in mm.
    for key in ("dirx", "diry", "dirz"):
        cond["straw"][key] *= 2
    out = stereo_points(sh, cond, nevents=1)
    np.testing.assert_allclose(out["positions"], [[10, 20, 5]], atol=1e-12)


@pytest.mark.parametrize("angle", [0., np.pi, 0.01])
def test_parallel_and_ill_conditioned_rejected(angle):
    sh, cond = fixture(angle=(0., angle))
    out = stereo_points(sh, cond, nevents=1)
    assert out["positions"].shape == (0, 3)
    assert out["diagnostics"]["rejected_parallel"] == 1
    check_accounting(out)


def test_same_plane_duplicates_not_jittered_or_deduplicated_arbitrarily():
    sh, cond = fixture(z=(0., 0., 10.), angle=(0., np.pi / 2, np.pi / 4))
    out = stereo_points(sh, cond, nevents=1)
    assert out["diagnostics"]["rejected_same_z"] == 1
    assert out["hit_index"].tolist() == [[0, 2], [1, 2]]
    np.testing.assert_allclose(out["positions"], [[10, 20, 5], [10, 20, 5]])
    check_accounting(out)


def test_nearby_z_boundary():
    sh, cond = fixture(z=(0., 50.))
    assert len(stereo_points(sh, cond, nevents=1)["positions"]) == 1
    out = stereo_points(sh, cond, nevents=1, max_dz=49.9)
    assert out["diagnostics"]["rejected_far_z"] == 1
    sh, cond = fixture(z=(0., 1e-4))
    assert stereo_points(sh, cond, nevents=1)["diagnostics"]["rejected_same_z"] == 1


@pytest.mark.parametrize("cut", ["wire_length", "wire_distance", "time"])
def test_measured_constraints(cut):
    sh, cond = fixture()
    if cut == "wire_length":
        cond["straw"]["hlen"][0] = 9.  # intersection is 10 mm from midpoint
    elif cut == "wire_distance":
        sh["x"][0] += 4.
        sh["uvar"][0] = 1.
    else:
        sh["time"][1] = 10.  # same time group, but >3*sqrt(4+4)
    out = stereo_points(sh, cond, nevents=1)
    assert out["diagnostics"]["rejected_" + cut] == 1
    assert len(out["positions"]) == 0
    check_accounting(out)


def test_zero_time_variance_equal_time_only():
    sh, cond = fixture()
    sh["timevar"][:] = 0
    assert len(stereo_points(sh, cond, nevents=1)["positions"]) == 1
    sh["time"][1] = 0.001
    assert stereo_points(sh, cond, nevents=1)["diagnostics"]["rejected_time"] == 1


def test_no_cross_event_or_time_group_pairs_unsorted():
    sh, cond = fixture(z=(0., 10., 0., 10., 0., 10.), angle=(0., np.pi / 2) * 3,
                       ev=[3, 3, 0, 0, 3, 3], time=[0, 0, 0, 0, 80, 80])
    out = stereo_points(sh, cond, nevents=6, pair_batch=1)
    assert out["diagnostics"]["pairs_considered"] == 3
    assert out["group_ev"].tolist() == [0, 3, 3]
    i, j = out["hit_index"].T
    np.testing.assert_array_equal(sh["ev"][i], sh["ev"][j])
    np.testing.assert_array_equal(sh["time"][i], sh["time"][j])
    np.testing.assert_array_equal(out["hit_group"][i], out["hit_group"][j])
    check_accounting(out)


def test_single_link_span_rejection_retains_membership_and_singletons():
    sh, cond = fixture(z=np.arange(8) * 3., angle=np.arange(8) * 0.5,
                       ev=[0] * 7 + [2], time=list(np.arange(7) * 19.) + [0.])
    out = stereo_points(sh, cond, nevents=4)
    assert out["group_n"].tolist() == [7, 1]
    assert out["member"].tolist() == list(range(8))
    assert out["diagnostics"]["wide_time_groups"] == 1
    assert out["diagnostics"]["singleton_groups"] == 1
    assert out["diagnostics"]["pairs_considered"] == 0
    check_accounting(out)


def test_pair_time_cap_applies_even_in_connected_cluster():
    sh, cond = fixture(z=(0., 10., 20.), angle=(0., 0.6, 1.2), time=[0., 19., 38.])
    sh["timevar"][:] = 10000.
    out = stereo_points(sh, cond, nevents=1)
    assert out["diagnostics"]["rejected_time"] == 1
    assert out["hit_index"].tolist() == [[0, 1], [1, 2]]


@pytest.mark.parametrize("bit", [6, FLAG["dead"], FLAG["noisy"], FLAG["tdiv"]])
def test_flags(bit):
    sh, cond = fixture()
    sh["flag"][0] ^= np.uint32(1 << bit)
    out = stereo_points(sh, cond, nevents=2)
    assert out["diagnostics"]["rejected_flags"] == 1
    assert out["member"].tolist() == [1]
    assert out["hit_group"].tolist() == [-1, 0]
    assert out["event_ngood"].tolist() == [1, 0]
    check_accounting(out)


@pytest.mark.parametrize("nevents", [0, 4])
def test_empty(nevents):
    sh, cond = fixture(z=(), angle=())
    out = stereo_points(sh, cond, nevents=nevents)
    assert out["positions"].shape == (0, 3)
    assert out["hit_index"].shape == (0, 2)
    assert out["hit_index"].dtype.kind == "i"
    assert out["wire_pull"].shape == (0, 2)
    assert out["member"].dtype.kind == "i"
    assert out["event_nhits"].tolist() == [0] * nevents
    check_accounting(out)


def test_hit_and_event_permutation_equivariance():
    sh, cond = fixture(z=(0., 10., 0., 10.), angle=(0., np.pi / 2) * 2,
                       ev=[0, 0, 3, 3], time=[1, 2, 1, 2])
    reference = stereo_points(sh, cond, nevents=5)
    order = np.array([3, 0, 2, 1])
    permuted = {k: v[order] for k, v in sh.items()}
    out = stereo_points(permuted, cond, nevents=5)
    np.testing.assert_array_equal(order[out["hit_index"]], reference["hit_index"])
    np.testing.assert_allclose(out["positions"], reference["positions"])
    event_map = np.array([4, 2, 1, 0, 3])
    permuted["ev"] = event_map[permuted["ev"]]
    out = stereo_points(permuted, cond, nevents=5)
    np.testing.assert_array_equal(out["event_npoints"][event_map], reference["event_npoints"])
    pairs = np.sort(order[out["hit_index"]], axis=1)
    assert sorted(map(tuple, pairs)) == sorted(map(tuple, reference["hit_index"]))


def test_pair_batches_complete_and_preflight_limit_failure():
    sh, cond = fixture(z=np.arange(9) * 2., angle=np.arange(9) * 0.3)
    reference = stereo_points(sh, cond, nevents=1, max_pairs=36)
    assert reference["diagnostics"]["pairs_considered"] == 36
    for batch in (1, 2, 7, 35, 36, 37):
        out = stereo_points(sh, cond, nevents=1, max_pairs=36, pair_batch=batch)
        for key in reference:
            if key == "diagnostics":
                assert out[key] == reference[key]
            else:
                np.testing.assert_array_equal(out[key], reference[key])
    with pytest.raises(ValueError, match="pair limit exceeded: 36.*no pairs truncated"):
        stereo_points(sh, cond, nevents=1, max_pairs=35, pair_batch=1)
    # Limits count even parallel/same-z pairs, not just survivors.
    sh, cond = fixture(z=(0.,) * 9, angle=(0.,) * 9)
    with pytest.raises(ValueError, match="pair limit"):
        stereo_points(sh, cond, nevents=1, max_pairs=35)


@pytest.mark.parametrize("field,value", [("ev", [-1, 0]), ("ev", [0, 2]), ("ev", [0., 0.]),
    ("sid", [0, 96]), ("sid", [0, 768]), ("sid", [0, 2]), ("sid", [0, 65536]),
    ("flag", [-1, 0]), ("flag", [0, 2**32]), ("x", [np.nan, 0]), ("x", [0.]),
    ("x", [[0., 0.]]), ("time", [0, np.inf]), ("uvar", [0., 1.]), ("timevar", [-1., 1.]),
    ("z", [1., 10.]), ("y", [21., 20.])])
def test_malformed_hits(field, value):
    sh, cond = fixture()
    sh[field] = np.asarray(value)
    with pytest.raises(ValueError):
        stereo_points(sh, cond, nevents=2)


@pytest.mark.parametrize("field,value", [("dirx", [0., 0.]), ("dirz", [0.1, 0.]),
    ("hlen", [0., 100.]), ("midx", [np.nan, 0.]), ("diry", [0.]), ("dirx", [1e308, 0.])])
def test_malformed_geometry(field, value):
    sh, cond = fixture()
    cond["straw"][field] = np.asarray(value)
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        stereo_points(sh, cond, nevents=1)


@pytest.mark.parametrize("kwargs", [dict(nevents=-1), dict(nevents=True), dict(nevents=1.5),
    dict(max_pairs=0), dict(pair_batch=0), dict(pair_batch=1.5), dict(max_pairs=True),
    dict(time_gap=0), dict(min_dz=0), dict(max_dz=1e-4), dict(min_sin=1.1),
    dict(wire_nsigma=np.nan), dict(time_nsigma=-1), dict(max_dt=np.inf), dict(max_time_span="x")])
def test_malformed_parameters(kwargs):
    sh, cond = fixture()
    with pytest.raises(ValueError):
        stereo_points(sh, cond, **(dict(nevents=1) | kwargs))


@pytest.mark.parametrize("missing", ["ev", "timevar", "straw", "dirx"])
def test_missing_columns(missing):
    sh, cond = fixture()
    if missing in sh:
        del sh[missing]
    elif missing == "straw":
        del cond[missing]
    else:
        del cond["straw"][missing]
    with pytest.raises(ValueError):
        stereo_points(sh, cond, nevents=1)


def test_no_input_mutation():
    sh, cond = fixture()
    before = copy.deepcopy((sh, cond))
    stereo_points(sh, cond, nevents=2)
    for key in sh:
        np.testing.assert_array_equal(sh[key], before[0][key])
    for key in cond["straw"]:
        np.testing.assert_array_equal(cond["straw"][key], before[1]["straw"][key])


def test_sloped_track_bias_is_not_a_fit():
    # True trajectory (x,y,z)=(z,0,z). Horizontal wire at z=0 and vertical
    # wire at z=10 intersect in projection at (10,0), not true midpoint (5,0).
    sh, cond = fixture()
    cond["straw"].update(midx=np.array([0., 10.]), midy=np.zeros(2))
    sh.update(x=np.array([0., 10.]), y=np.zeros(2))
    out = stereo_points(sh, cond, nevents=1)
    np.testing.assert_allclose(out["positions"], [[10., 0., 5.]], atol=1e-12)
    assert not np.allclose(out["positions"][0], [5., 0., 5.])


def test_pair_enumeration_mixed_groups_matches_exhaustive_reference():
    n = 14
    sh, cond = fixture(z=np.arange(n) * 2., angle=np.arange(n) * 0.4,
                       ev=[0] + [1] * 3 + [3] * 4 + [4] * 6,
                       time=[0.] * 8 + list(np.arange(6) * 19.))
    sh["timevar"][:] = 10000.
    out = stereo_points(sh, cond, nevents=6, max_time_span=90., pair_batch=3)
    i, j = np.triu_indices(n, 1)
    same_group = (sh["ev"][i] == sh["ev"][j]) & (sh["ev"][i] != 4)
    i, j = i[same_group], j[same_group]
    assert out["diagnostics"]["pairs_considered"] == len(i) == 9
    sine = np.abs(np.sin((i - j) * 0.4))
    expected = np.column_stack((i[sine >= 0.25], j[sine >= 0.25]))
    np.testing.assert_array_equal(out["hit_index"], expected)
    assert out["group_npairs"].tolist() == [0, 3, 6, 0]
    assert out["diagnostics"]["wide_time_groups"] == 1
    check_accounting(out)


def test_packed_straw_identity_not_collection_index():
    sh, cond = fixture()
    # Different planes/panels; geometry deliberately reordered and sparse.
    sh["sid"] = np.array([1024 + 128 + 3, 2 * 1024 + 5])
    indices = np.array([7 * 96 + 3, 12 * 96 + 5])
    expanded = {}
    for key, values in cond["straw"].items():
        expanded[key] = np.zeros(indices.max() + 1)
        expanded[key][indices] = values
    out = stereo_points(sh, {"straw": expanded}, nevents=1)
    np.testing.assert_allclose(out["positions"], [[10., 20., 5.]])
    assert out["hit_index"].tolist() == [[0, 1]]


def test_all_flags_rejected_preserves_census():
    sh, cond = fixture(ev=[0, 2])
    sh["flag"][:] = 0
    out = stereo_points(sh, cond, nevents=4)
    assert out["event_nhits"].tolist() == [1, 0, 1, 0]
    assert out["event_ngood"].tolist() == [0, 0, 0, 0]
    assert out["hit_group"].tolist() == [-1, -1]
    assert out["member"].size == out["group_ev"].size == 0
    check_accounting(out)


def tilted_fixture(z=(0., 10.), measured_offset=(0., 0.), angle=(0., np.pi / 2)):
    """Known crossing at wire distances (10,-20) with unequal XY norms."""
    sh, cond = fixture(z=z, angle=angle)
    d = np.column_stack((.6 * np.cos(angle), .6 * np.sin(angle), [.8, -.8]))
    # Distinct tilt magnitudes expose accidental XY renormalization/scaling.
    d[1] = [.8 * np.cos(angle[1]), .8 * np.sin(angle[1]), -.6]
    crossing = np.column_stack(([10., 10.], [20., 20.], z))
    u = np.array([10., -20.])
    mid = crossing - u[:, None] * d
    measured = crossing + np.asarray(measured_offset)[:, None] * d
    cond["straw"].update(dict(zip(("midx", "midy", "midz"), mid.T)))
    cond["straw"].update(dict(zip(("dirx", "diry", "dirz"), d.T)))
    sh.update(dict(zip(("x", "y", "z"), measured.T)))
    return sh, cond


@pytest.mark.parametrize("angle", [(0., np.pi / 2), (.2, 1.8), (np.pi, np.pi / 3)])
@pytest.mark.parametrize("scales", [(1., 1.), (3., .2), (-2., 4.)])
def test_tilted_analytic_intersection_and_original_3d_pull(angle, scales):
    sh, cond = tilted_fixture(measured_offset=(4., -6.), angle=angle)
    sh["uvar"] = np.array([4., 9.])
    for k in ("dirx", "diry", "dirz"):
        cond["straw"][k] *= scales
    out = stereo_points(sh, cond, nevents=2)
    np.testing.assert_allclose(out["positions"], [[10., 20., 5.]], atol=1e-12)
    np.testing.assert_allclose(out["wire_pull"], [[-2. * np.sign(scales[0]), 2. * np.sign(scales[1])]], atol=1e-12)
    # Measured z mean is 8.4, deliberately distinct from intersection z mean 5.
    assert not np.isclose(np.mean(sh["z"]), out["positions"][0, 2])
    check_accounting(out)


@pytest.mark.parametrize("wire,half_length", [(0, 8.), (1, 18.)])
def test_tilted_finite_length_uses_wire_mm_not_xy_mm(wire, half_length):
    sh, cond = tilted_fixture()
    # Projected distances are only 6 and 16 mm, but actual distances are 10,20.
    cond["straw"]["hlen"][wire] = half_length
    out = stereo_points(sh, cond, nevents=1)
    assert out["diagnostics"]["rejected_wire_length"] == 1
    check_accounting(out)


@pytest.mark.parametrize("offset,rejected", [(3., False), (3.1, True)])
def test_tilted_pull_cut_is_along_wire_not_projected(offset, rejected):
    sh, cond = tilted_fixture(measured_offset=(offset, 0.))
    sh["uvar"][:] = 1.
    out = stereo_points(sh, cond, nevents=1)
    assert out["diagnostics"]["rejected_wire_distance"] == int(rejected)
    if not rejected:
        np.testing.assert_allclose(out["wire_pull"], [[-3., 0.]], atol=1e-12)
    check_accounting(out)


@pytest.mark.parametrize("z,offset,reason", [((0., 10.), (12.5, 0.), None),
    ((0., 0.), (0., -20.), "same_z"), ((0., 60.), (75., 0.), "far_z")])
def test_tilted_z_window_uses_intersection_not_measured_z(z, offset, reason):
    sh, cond = tilted_fixture(z=z, measured_offset=offset)
    sh["uvar"][:] = 10000.
    out = stereo_points(sh, cond, nevents=1)
    if reason is None:
        assert len(out["positions"]) == 1  # measured hit z equal, crossing z distinct
    else:
        assert out["diagnostics"]["rejected_" + reason] == 1
    check_accounting(out)


def test_tilted_angle_cut_is_projected_angle_not_3d_cross_magnitude():
    sh, cond = tilted_fixture()
    crossing = np.column_stack([sh[k] for k in ("x", "y", "z")])
    d = np.array([[.1, 0., np.sqrt(.99)], [0., .1, -np.sqrt(.99)]])
    mid = crossing - np.array([10., -20.])[:, None] * d
    cond["straw"].update(dict(zip(("midx", "midy", "midz"), mid.T)))
    cond["straw"].update(dict(zip(("dirx", "diry", "dirz"), d.T)))
    out = stereo_points(sh, cond, nevents=1)
    np.testing.assert_allclose(out["positions"], [[10., 20., 5.]], atol=1e-12)
    assert out["diagnostics"]["rejected_parallel"] == 0


@pytest.mark.parametrize("xy", [0., 1e-12])
def test_axial_and_numerically_singular_projection_rejected_explicitly(xy):
    sh, cond = fixture()
    for k, value in zip(("dirx", "diry", "dirz"), (xy, 0., 1.)):
        cond["straw"][k][0] = value
    for k in ("x", "y", "z"):
        cond["straw"]["mid" + k][0] = sh[k][0]
    with pytest.raises(ValueError, match="axial or ill-conditioned XY"):
        stereo_points(sh, cond, nevents=1)


def test_singular_intersection_rejection_precedes_undefined_z():
    sh, cond = fixture(z=(0., 0.), angle=(0., 0.))
    out = stereo_points(sh, cond, nevents=1)
    assert out["diagnostics"]["rejected_parallel"] == 1
    assert out["diagnostics"]["rejected_same_z"] == 0
    check_accounting(out)


@pytest.mark.parametrize("angle", [0., .01, np.pi])
def test_tilted_parallel_and_ill_conditioned_pair_explicit_rejection(angle):
    sh, cond = tilted_fixture(angle=(0., angle))
    out = stereo_points(sh, cond, nevents=1)
    assert out["diagnostics"]["rejected_parallel"] == 1
    assert len(out["positions"]) == 0
    check_accounting(out)


def test_tilted_batching_matches_independent_batched_linear_solve():
    n = 9
    sh, cond = fixture(z=np.arange(n) * 2., angle=np.arange(n) * .3)
    angle = np.arange(n) * .3
    tilt = np.linspace(-.8, .8, n)
    d = np.column_stack((np.sqrt(1 - tilt**2) * np.cos(angle),
                         np.sqrt(1 - tilt**2) * np.sin(angle), tilt))
    crossing = np.column_stack([sh[k] for k in ("x", "y", "z")])
    mid = crossing - np.linspace(-30., 30., n)[:, None] * d
    measured = crossing + np.linspace(-2., 2., n)[:, None] * d
    cond["straw"].update(dict(zip(("midx", "midy", "midz"), mid.T)))
    cond["straw"].update(dict(zip(("dirx", "diry", "dirz"), d.T)))
    sh.update(dict(zip(("x", "y", "z"), measured.T)))
    out = stereo_points(sh, cond, nevents=3, pair_batch=2)
    reference = stereo_points(sh, cond, nevents=3, pair_batch=100)
    for k in out:
        if k == "diagnostics":
            assert out[k] == reference[k]
        else:
            np.testing.assert_array_equal(out[k], reference[k])
    i, j = out["hit_index"].T
    matrix = np.stack((d[i, :2], -d[j, :2]), axis=2)
    u = np.linalg.solve(matrix, (mid[j, :2] - mid[i, :2])[..., None])[..., 0]
    pi, pj = mid[i] + u[:, 0, None] * d[i], mid[j] + u[:, 1, None] * d[j]
    np.testing.assert_allclose(out["positions"], (pi + pj) / 2, atol=1e-12)
    measured_u = np.sum((measured - mid) * d, axis=1)
    expected_pull = (u - np.column_stack((measured_u[i], measured_u[j]))) / 10.
    np.testing.assert_allclose(out["wire_pull"], expected_pull, atol=1e-12)
    check_accounting(out)
