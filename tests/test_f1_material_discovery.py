"""Synthetic material-discovery contracts; no captured discovery input or fixtures.

All 36 planes, 216 panels and 20,736 straws per target are constructed in
memory. These tests exercise this Python implementation, not fit equivalence.
"""
from copy import deepcopy

import awkward as ak
import numpy as np
import pytest

import f1_material_discovery as M
import f1_material_discovery_compare as D


DEFAULTS = dict(AddStrawMinDz=0., MaxDStraw=2, MaxStrawDOCA=5.,
                MaxStrawDOCAConsistency=1., MaxStrawUposBuffer=0.,
                StrawNBuffer=0, SkipStrawCheck=False)


def trajectory(n=1, pieces=1):
    q = np.zeros((n, pieces, 6))
    q[:, :, 0] = 30.25 + np.arange(n)[:, None]
    q[:, :, 5] = 100.
    ranges = np.broadcast_to(np.column_stack((np.arange(pieces),
                                             np.arange(1, pieces + 1))),
                             (n, pieces, 2)).astype(float).copy()
    return dict(keys=np.column_stack((np.ones(n, dtype=np.int64),
                                      np.full(n, 2), np.arange(n) + 3,
                                      np.full(n, 4), np.full(n, 7))),
                q=q, ranges=ranges, counts=np.full(n, pieces, dtype=np.int64),
                mass=np.full(n, 100.),
                covariance=np.broadcast_to(np.eye(6), (n, pieces, 6, 6)).copy())


def trajectory_table(t):
    n, width = t["q"].shape[:2]
    data = {name: np.repeat(t["keys"][:, i], width)
            for i, name in enumerate(M.B.TRAJECTORY)}
    data.update(piece_index=np.tile(np.arange(width), n),
                category=np.full(n * width, 2, dtype=np.int64),
                parameters=t["q"].reshape(-1, 6),
                range=t["ranges"].reshape(-1, 2),
                mass=np.repeat(t["mass"], width),
                covariance=t["covariance"].reshape(-1, 36))
    return ak.Array(data)


def full_geometry(t):
    """Columnar fixture: no Python loop over individual geometric objects."""
    n = len(t["keys"])
    bindings = {name: t["keys"][:, i] for i, name in enumerate(M.B.KEYS)}
    bindings.update(source_status=np.ones(n, dtype=np.int64),
                    geometry_index=np.arange(n))
    tables = dict(bindings=ak.Array(bindings),
                  tracker=ak.Array(dict(geometry_index=np.arange(n),
                                        straw_outer_radius=np.full(n, 2.5),
                                        wire_radius=np.full(n, .0125))))
    for kind, count, ordinal_name in (("planes", 36, "plane_ordinal"),
                                      ("panels", 216, "tracker_panel_ordinal"),
                                      ("straws", 20736, "tracker_straw_ordinal")):
        ordinal = np.tile(np.arange(count), n)
        rows = n * count
        data = dict(geometry_index=np.repeat(np.arange(n), count),
                    fullsid=ordinal.copy())
        data[ordinal_name] = ordinal
        if kind == "planes":
            data.update(plane_exists=np.ones(rows, dtype=bool),
                        origin=np.column_stack((np.zeros(rows), np.zeros(rows),
                                                ordinal * 4.)))
        elif kind == "panels":
            data.update(origin=np.column_stack((np.zeros(rows), np.zeros(rows),
                                                (ordinal // 6) * 4.)),
                        v_direction=np.tile([0., 1., 0.], (rows, 1)),
                        ds_to_panel_rotation=np.tile(np.eye(3).ravel(), (rows, 1)),
                        ds_to_panel_translation=np.zeros((rows, 3)),
                        panel_ordinal=ordinal % 6)
        else:
            y = 10. + ordinal % 96
            z = (ordinal // 576) * 4.
            # Deliberately distinct tube endpoints, wire endpoints and midpoints.
            data.update(tube_cal=np.column_stack((np.full(rows, 200.), y + .1, z + .1)),
                        tube_hv=np.column_stack((np.full(rows, -200.), y + .1, z + .1)),
                        wire_cal=np.column_stack((np.full(rows, 150.), y + .5, z)),
                        wire_midpoint=np.column_stack((np.zeros(rows), y, z)),
                        wire_direction=np.tile([1., 0., 0.], (rows, 1)),
                        half_length=np.full(rows, 200., dtype=np.float32),
                        straw_ordinal=ordinal % 96)
        tables[kind] = ak.Array(data)
    return tables


@pytest.fixture(scope="module")
def world():
    t = trajectory(n=2)
    t["ranges"][:] = [-10., 10.]
    tables = full_geometry(t)
    g = M.geometry(tables)
    old = np.zeros((2, 65536), dtype=bool)
    out = M.generate(t, g, old)
    return t, tables, g, old, out


def receipt(keys, strawids):
    return dict(identities=dict(object_keys=np.asarray(keys, dtype=np.int64).reshape(-1, 5),
                                strawid=np.asarray(strawids, dtype=np.int64)))


def fake_linear_particle(q, time, mass):
    """Exact unit-z motion with q[z0] used as the requested root time."""
    p = np.zeros((len(time), 3))
    p[:, 2] = time - q[:, 2]
    return p, np.tile([0., 0., 1.], (len(time), 1)), np.ones(len(time))


def test_configuration_exact_defaults_and_fresh_copy(tmp_path):
    path = tmp_path / "config.fcl"
    text = "\n".join(f"{key}: {'false' if value is False else value}"
                     for key, value in DEFAULTS.items())
    path.write_text(text + "\nMaxStrawDOCA: 5.0\n")
    first = M.configuration(path)
    assert first == DEFAULTS
    first["MaxStrawDOCA"] = 99
    assert M.configuration(path) == DEFAULTS


@pytest.mark.parametrize("change", ["missing", "duplicate", "boolean"])
def test_configuration_rejects_unsupported_source(tmp_path, change):
    text = "\n".join(f"{key}: {'false' if value is False else value}"
                     for key, value in DEFAULTS.items())
    if change == "missing":
        text = text.replace("MaxDStraw: 2", "OtherValue: 2")
    elif change == "duplicate":
        text += "\nMaxDStraw: 3"
    else:
        text = text.replace("SkipStrawCheck: false", "SkipStrawCheck: true")
    path = tmp_path / "config.fcl"
    path.write_text(text)
    with pytest.raises(ValueError, match="source config|unsupported config"):
        M.configuration(path)


def test_transform_preserves_source_grouping_and_row_major_rotation():
    rotation = np.array([np.eye(3), [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    position = np.array([[1.e16, -1.e16, 1.], [2., 3., 4.]])
    rotation[0, 0] = [1., 1., 1.]
    translation = np.array([[2., 3., 4.], [5., 6., 7.]])
    result = M.transform(position, rotation.reshape(2, 9), translation)
    # ((1e16 + -1e16) + 1) + 2 == 3; regrouping can produce 2.
    np.testing.assert_array_equal(result, [[3., -1.e16 + 3., 5.], [2., 8., 11.]])


def test_tzero_shared_endpoints_zero_duration_and_counted_padding():
    t = trajectory(n=3, pieces=3)
    t["q"][:, :, 4] = [[1., 9., 9.], [0., 9., 9.], [.25, 9., 9.]]
    t["ranges"][1] = [[0., 0.], [0., 0.], [0., 0.]]
    t["counts"][2] = 1
    t["q"][2, 1:] = np.nan
    time, index, moves = M.tzero(t)
    np.testing.assert_array_equal(time, [1., 0., .25])
    np.testing.assert_array_equal(index, [0, 0, 0])
    np.testing.assert_array_equal(moves, [1, 1, 1])


def test_tzero_caps_at_ten_pointer_changes_without_extra_resolve():
    t = trajectory(n=2, pieces=2)
    t["q"][:, :, 4] = [[1.5, -.5], [.25, .25]]
    time, index, moves = M.tzero(t)
    np.testing.assert_array_equal(time, [1.5, .25])
    np.testing.assert_array_equal(index, [1, 0])
    np.testing.assert_array_equal(moves, [10, 1])
    assert time[0] != t["q"][0, index[0], 4]  # Returned pointer is unsolved.


def test_ztime_shared_boundary_and_one_ulp_sides(monkeypatch):
    monkeypatch.setattr(M.C, "particle", fake_linear_particle)
    t = trajectory(n=3, pieces=2)
    roots = np.array([np.nextafter(1., -np.inf), 1., np.nextafter(1., np.inf)])
    t["q"][:, :, 2] = roots[:, None]
    time, index, tries, active = M.ztime(t, np.arange(3), np.zeros(3))
    np.testing.assert_array_equal(time, roots)
    np.testing.assert_array_equal(index, [0, 0, 1])
    np.testing.assert_array_equal(tries, [1, 1, 2])
    assert not active.any()


def test_ztime_begins_at_range_begin_not_parameter_tzero(monkeypatch):
    seen = []

    def particle(q, time, mass):
        seen.append(time.copy())
        return fake_linear_particle(q, time, mass)

    monkeypatch.setattr(M.C, "particle", particle)
    t = trajectory(pieces=2)
    t["ranges"][:] = [[7., 8.], [8., 9.]]
    t["q"][0, :, 4] = 99.
    t["q"][0, :, 2] = [8.5, 8.75]
    time, index, tries, active = M.ztime(t, np.array([0]), np.array([0.]))
    np.testing.assert_array_equal(np.concatenate(seen), [7., 8.5])
    assert (time[0], index[0], tries[0], active[0]) == (8.75, 1, 2, False)


def test_ztime_cap_retains_last_time_but_advances_unsolved_pointer(monkeypatch):
    monkeypatch.setattr(M.C, "particle", fake_linear_particle)
    t = trajectory(n=2, pieces=2)
    t["q"][:, :, 2] = [[2., -1.], [.25, .25]]
    time, index, tries, active = M.ztime(t, np.arange(2), np.zeros(2))
    np.testing.assert_array_equal(time, [-1., .25])
    np.testing.assert_array_equal(index, [0, 0])
    np.testing.assert_array_equal(tries, [100, 1])
    np.testing.assert_array_equal(active, [True, False])
    assert time[0] != t["q"][0, index[0], 2]


@pytest.mark.parametrize("parent,z", [(np.array([0.]), np.array([0.])),
                                       (np.array([True]), np.array([0.])),
                                       (np.array([-1]), np.array([0.])),
                                       (np.array([1]), np.array([0.])),
                                       (np.array([0]), np.array([np.nan]))])
def test_ztime_invalid_parent_or_nonfinite_target(parent, z):
    with pytest.raises(ValueError):
        M.ztime(trajectory(), parent, z)


@pytest.mark.parametrize("doca,variance,du,half,expected", [
    (5., 100., 0., 10., False),
    (-5., 100., 0., 10., False),
    (np.nextafter(5., 0.), 100., 0., 10., True),
    (3.5, 1., 0., 10., False),
    (np.nextafter(3.5, 0.), 1., 0., 10., True),
    (2.5, -1., 0., 10., True),
    (2.6, -1., 0., 10., False),
    (1., 1., float(np.float32(1.00000006)), 1.00000006, False),
    (1., 1., 1.00000008, 1.00000006, True),
])
def test_material_strict_cuts_and_float32_half_length(doca, variance, du, half, expected):
    dsig, passed = M.material_cut(np.array([doca]), np.array([variance]),
                                  np.array([du]), np.array([2.5]),
                                  np.array([half], dtype=np.float32))
    expected_dsig = max(0., abs(doca) - 2.5) / np.sqrt(max(variance, np.finfo(float).tiny))
    assert dsig[0] == expected_dsig
    assert bool(passed[0]) is expected
    assert np.isfinite(dsig).all()


def test_similarity_signed_quadratic_and_source_sum_order():
    row = np.ones((2, 6))
    covariance = np.zeros((2, 6, 6))
    covariance[0] = np.diag([1.e16, -1.e16, 1., 0., 0., 0.])
    covariance[1] = -np.eye(6)
    np.testing.assert_array_equal(M.similarity(row, covariance), [1., -6.])
    covariance[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite variance"):
        M.similarity(row, covariance)


def test_pack_filters_nonmaterial_sorts_pieces_and_preserves_covariance():
    t = trajectory(n=2, pieces=2)
    t["ranges"][:, 0] = [0., 0.]
    t["ranges"][:, 1] = [0., 1.]
    t["covariance"][1, 1] *= 3.
    table = trajectory_table(t)
    ignored = ak.with_field(table[:1], np.array([1]), "category")
    packed = M.pack(ak.concatenate([table[::-1], ignored]))
    for name in ("keys", "q", "ranges", "counts", "mass", "covariance"):
        np.testing.assert_array_equal(packed[name], t[name])


@pytest.mark.parametrize("problem", ["contexts", "category_type", "covariance"])
def test_pack_rejects_ambiguous_context_or_invalid_dtype_and_covariance(problem):
    t = trajectory(n=2)
    table = trajectory_table(t)
    if problem == "contexts":
        t["keys"][1, :4] = t["keys"][0, :4]
        t["keys"][1, 4] += 1
        table = trajectory_table(t)
    elif problem == "category_type":
        table = ak.with_field(table, np.full(2, 2.), "category")
    else:
        t["covariance"][0, 0, 0, 1] = 1.
        table = trajectory_table(t)
    with pytest.raises(ValueError):
        M.pack(table)


def test_geometry_full_universe_is_ordered_and_keeps_float32(world):
    _, tables, g, _, _ = world
    reordered = M.geometry({name: value[::-1] for name, value in tables.items()})
    # Binding order controls parent order, but ordinals always control local order.
    for kind, count in (("planes", 36), ("panels", 216), ("straws", 20736)):
        assert g[kind]["fullsid"].shape == (2, count)
        np.testing.assert_array_equal(g[kind]["fullsid"], np.tile(np.arange(count), (2, 1)))
        for field in g[kind]:
            np.testing.assert_array_equal(reordered[kind][field][::-1], g[kind][field])
    assert g["straws"]["half_length"].dtype == np.float32
    np.testing.assert_array_equal(reordered["target"][::-1], g["target"])


@pytest.mark.parametrize("problem", ["missing_straw", "duplicate_panel", "parent", "half_dtype"])
def test_geometry_rejects_incomplete_or_invalid_source(world, problem):
    _, original, _, _, _ = world
    tables = dict(original)
    if problem == "missing_straw":
        tables["straws"] = tables["straws"][:-1]
    elif problem == "duplicate_panel":
        tables["panels"] = ak.concatenate([tables["panels"][:-1], tables["panels"][:1]])
    elif problem == "parent":
        tables["planes"] = ak.with_field(tables["planes"], np.full(72, 99), "geometry_index")
    else:
        tables["straws"] = ak.with_field(tables["straws"],
                                         np.full(41472, 200., dtype=np.float64), "half_length")
    with pytest.raises(ValueError):
        M.geometry(tables)


def test_exclusions_union_is_full_target_scoped_and_ignores_captured_fields(world):
    t, _, g, _, _ = world
    a, b = t["keys"][:, :4]
    initial = receipt([np.r_[a, 5], np.r_[a, 6], np.r_[b, 5]], [18, 18, 19])
    associated = receipt([np.r_[a, 90], np.r_[b, 91]], [20, 19])
    initial["captured_discovery"] = object()
    initial["generated"] = object()
    old = M.exclusions(t, g, initial, associated)
    assert old.dtype == bool and old.shape == (2, 65536)
    np.testing.assert_array_equal(np.flatnonzero(old[0]), [18, 20])
    np.testing.assert_array_equal(np.flatnonzero(old[1]), [19])


@pytest.mark.parametrize("problem", ["straw_type", "parent"])
def test_exclusions_rejects_bad_membership_input(world, problem):
    t, _, g, _, _ = world
    r = receipt([np.r_[t["keys"][0, :4], 0]], [18])
    if problem == "straw_type":
        r["identities"]["strawid"] = np.array([18.])
    else:
        r["identities"]["object_keys"][0, 0] = 99
    with pytest.raises(ValueError):
        M.exclusions(t, g, r, receipt([], []))


def test_generate_full_hierarchy_wire_limits_and_tube_line_without_capture(world):
    t, _, g, old, out = world
    assert out["universe"] == dict(contexts=2, planes=72, panels=432, straws=41472)
    expected_counts = [2, 72, 72, 432, 432, 432, 2160, 2160, 2160]
    assert [len(out["stages"][i]["parent"]) for i in range(9)] == expected_counts
    assert all(stage["passed"].all() for stage in out["stages"].values())
    attempt = out["stages"][7]
    sid, parent = attempt["straw"], attempt["parent"]
    np.testing.assert_array_equal(attempt["plane"], sid // 576)
    np.testing.assert_array_equal(attempt["panel"], sid // 96)
    np.testing.assert_array_equal(np.bincount(parent), [1080, 1080])
    np.testing.assert_array_equal(sid[parent == 0] % 96, np.tile(np.arange(18, 23), 216))
    np.testing.assert_array_equal(sid[parent == 1] % 96, np.tile(np.arange(19, 24), 216))
    np.testing.assert_array_equal(out["stages"][0]["values"][:, 7:13],
                                  np.tile([2.5, 5., np.sqrt(150.**2 + 105.5**2) + 5.,
                                           10., 105., 1.], (2, 1)))
    np.testing.assert_array_equal(out["generated"]["line"]["end"], g["straws"]["tube_cal"][parent, sid])
    np.testing.assert_array_equal(out["generated"]["line_start"], g["straws"]["tube_hv"][parent, sid])
    assert not np.array_equal(out["generated"]["line"]["end"], g["straws"]["wire_cal"][parent, sid])
    np.testing.assert_allclose(out["generated"]["radius"], .0125, rtol=0, atol=0)
    assert not old.any()
    np.testing.assert_array_equal(M.array(ak.num(out["stages"][0]["old"])), [0, 0])
    # Generated membership depends only on the three explicit inputs.
    poisoned = dict(g, discovery=object(), constructors=object(), accepted=object())
    repeated = M.generate(t, poisoned, old)
    np.testing.assert_array_equal(repeated["stages"][8]["straw"], out["stages"][8]["straw"])


def test_generate_old_membership_removes_only_that_target_attempt(world):
    t, _, g, old, baseline = world
    old = old.copy()
    old[0, 19] = True
    old[1, 65535] = True  # Membership outside a reached window remains recorded.
    out = M.generate(t, g, old)
    stage6, stage7 = out["stages"][6], out["stages"][7]
    assert len(stage6["parent"]) == 2160 and stage6["passed"].sum() == 2159
    assert not np.any((stage7["parent"] == 0) & (stage7["straw"] == 19))
    assert np.any((stage7["parent"] == 1) & (stage7["straw"] == 19))
    np.testing.assert_array_equal(M.array(out["stages"][0]["old"]), [[19], [65535]])
    for number in range(1, 6):
        np.testing.assert_array_equal(out["stages"][number]["passed"], baseline["stages"][number]["passed"])


def test_generate_missing_plane_gates_children_not_universe(world):
    t, _, original, old, _ = world
    g = deepcopy(original)
    g["planes"]["plane_exists"][0, 5] = False
    out = M.generate(t, g, old)
    assert out["universe"]["straws"] == 41472
    assert len(out["stages"][1]["parent"]) == 72
    assert out["stages"][1]["passed"].sum() == 71
    assert len(out["stages"][3]["parent"]) == 426
    assert len(out["stages"][6]["parent"]) == 2130
    for number in range(2, 9):
        stage = out["stages"][number]
        assert not np.any((stage["parent"] == 0) & (stage["plane"] == 5))


def test_comparison_keys_keep_context_and_reject_bad_capture_without_feedback(world):
    t, _, g, old, out = world
    stage = out["stages"][6]
    keys = D.stage_keys(stage, t)
    np.testing.assert_array_equal(keys[:, :5], t["keys"][stage["parent"]])
    np.testing.assert_array_equal(keys[:, 5:], np.column_stack((stage["plane"], stage["panel"], stage["straw"])))
    assert len(np.unique(keys, axis=0)) == 2160
    with pytest.raises(ValueError, match="invalid discovery stage"):
        D.compare(out, t, ak.Array(dict(stage=np.array([99]))), ak.Array([]))
    with pytest.raises(ValueError, match="stage census differs"):
        D.compare(out, t, ak.Array(dict(stage=np.array([0]))), ak.Array([]))
    repeated = M.generate(t, g, old)
    np.testing.assert_array_equal(repeated["accepted"], out["accepted"])
