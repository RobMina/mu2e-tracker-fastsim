"""Finite straw-wall intervals and unnormalised geometric lever-arm columns."""
import numpy as np
import pytest

from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs, fieldwall as fw


def single_straw():
    return geometry.StrawGeometry(
        plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.zeros((1, 3)), dir=np.array([[0., 0., 1.]]), halflen=np.array([2.]), radius=1.,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None],
        plane_t=np.zeros((1, 3)), wall_rmax=1.2)


def tilted_straws():
    return geometry.StrawGeometry(
        plane=np.zeros(6, dtype=np.int64), panel=np.repeat(np.arange(3), 2), straw=np.tile([0, 1], 3),
        mid=np.array([[0., 0., 0.], [0., 3., 0.], [0., 0., 5.], [0., 3., 5.],
                      [0., 0., 12.], [3., 0., 12.]]),
        dir=np.array([[1., 0., 0.], [1., 0., 0.], [.8, 0., .6], [.8, 0., .6],
                      [0., 1., 0.], [0., 1., 0.]]),
        halflen=np.array([4., 3., 7., 5., 2., 4.]), radius=1., wall_rmax=1.2,
        panel_R=np.tile(np.eye(3), (3, 1, 1)), panel_t=np.zeros((3, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def secants(start, end):
    start, end = np.atleast_2d(start).astype(float), np.atleast_2d(end).astype(float)
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    assert np.all(length > 0)
    return start, delta / length[:, None], length


def sorted_rows(rows):
    return rows[np.lexsort((rows[:, 1], rows[:, 2], rows[:, 0]))]


def check_rows(rows, length, nstraws):
    assert isinstance(rows, np.ndarray) and rows.ndim == 2 and rows.shape[1] == 4
    assert np.issubdtype(rows.dtype, np.floating) and np.isfinite(rows).all()
    lane, straw = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
    np.testing.assert_array_equal(rows[:, :2], np.column_stack((lane, straw)))
    assert np.all((lane >= 0) & (lane < len(length)))
    assert np.all((straw >= 0) & (straw < nstraws))
    assert np.all((rows[:, 2] >= 0) & (rows[:, 3] <= length[lane]))
    assert np.all(rows[:, 3] > rows[:, 2])
    np.testing.assert_array_equal(rows, sorted_rows(rows))
    _, counts = np.unique(rows[:, :2], axis=0, return_counts=True)
    assert np.all(counts <= 2)
    grouped = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    same = np.all(grouped[1:, :2] == grouped[:-1, :2], axis=1)
    assert np.all(grouped[1:, 2][same] >= grouped[:-1, 3][same])


def cylinder_interval(x, c, axis, half, radius):
    """Scalar finite-cylinder oracle using transverse projection and quadratic roots."""
    projector = np.eye(3) - np.outer(axis, axis)
    v, u = projector @ x, projector @ c
    aa, bb, cc = u @ u, u @ v, v @ v - radius**2
    if aa <= 1e-15:
        if cc > 0:
            return np.inf, -np.inf
        lo, hi = -np.inf, np.inf
    else:
        disc = bb**2 - aa*cc
        if disc <= 0:
            return np.inf, -np.inf
        lo, hi = (-bb - np.sqrt(disc))/aa, (-bb + np.sqrt(disc))/aa
    axial_x, axial_c = x @ axis, c @ axis
    if abs(axial_c) <= 1e-12:
        if abs(axial_x) > half:
            return np.inf, -np.inf
    else:
        cap0, cap1 = (-half - axial_x)/axial_c, (half - axial_x)/axial_c
        lo, hi = max(lo, min(cap0, cap1)), min(hi, max(cap0, cap1))
    return lo, hi


def oracle_crossings(x, c, length, geom, radius=None):
    """Check every synthetic lane/straw pair without panel bounds or production intersections."""
    radius = geom.radius if radius is None else radius
    rows = []
    for lane in range(len(x)):
        for straw in range(geom.nstraws):
            a, b = cylinder_interval(x[lane] - geom.mid[straw], c[lane], geom.dir[straw],
                                     geom.halflen[straw], radius)
            a, b = max(a, 0.), min(b, length[lane])
            if b > a:
                rows.append((lane, straw, a, b))
    return sorted_rows(np.asarray(rows, dtype=float).reshape(-1, 4))


def all_straw_oracle(x, c, length, geom, r_out=1.2):
    rows = []
    columns = tuple(np.zeros(len(length)) for _ in range(3))
    for lane, straw, a, b in oracle_crossings(x, c, length, geom, r_out):
        lane, straw = int(lane), int(straw)
        g0, g1 = cylinder_interval(x[lane] - geom.mid[straw], c[lane], geom.dir[straw],
                                   geom.halflen[straw], geom.radius)
        g0, g1 = max(g0, a), min(g1, b)
        if g1 > g0:
            if g0 > a:
                rows.append((lane, straw, a, g0))
            if b > g1:
                rows.append((lane, straw, g1, b))
        else:
            rows.append((lane, straw, a, b))
            g0 = g1 = a
        # Integrate outer minus gas independently of the returned wall partition.
        remaining = length[lane]
        for power, out in enumerate(columns, 1):
            out[lane] += ((remaining-a)**power - (remaining-b)**power
                          - (remaining-g0)**power + (remaining-g1)**power)/power
    return sorted_rows(np.asarray(rows, dtype=float).reshape(-1, 4)), columns


def check_columns(actual, expected, atol=1e-11):
    assert isinstance(actual, tuple) and len(actual) == 3
    for got, want in zip(actual, expected):
        assert isinstance(got, np.ndarray) and got.shape == want.shape
        assert np.issubdtype(got.dtype, np.floating) and np.isfinite(got).all()
        np.testing.assert_allclose(got, want, atol=atol, rtol=1e-12)


def test_analytic_central_crossing_and_geometric_not_radiation_length_columns():
    x, c, length = secants([-2., 0., 0.], [2., 0., 0.])
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    rows = index.intervals(x, c, length)
    check_rows(rows, length, 1)
    np.testing.assert_allclose(rows, [[0., 0., .8, 1.], [0., 0., 3., 3.2]], atol=1e-14, rtol=0)
    expected = (np.array([.4]), np.array([.8]), np.array([6.256 / 3]))
    check_columns(index.columns(x, c, length), expected)
    check_columns(fw.interval_columns(rows, length), expected)


def test_gas_free_graze_uses_expanded_bounds_and_cached_outer_index_without_mutation():
    geom = single_straw()
    x, c, length = secants([-2., 1.1, 0.], [2., 1.1, 0.])
    assert fs.ChordIndex(geom).crossings(x, c, length, geom).shape == (0, 4)
    constructed = []

    class RecordingIndex(fs.ChordIndex):
        def __init__(self, outer):
            constructed.append(outer)
            assert outer is not geom and outer.radius == 1.2
            super().__init__(outer)

        def crossings(self, *args):
            raise AssertionError('wall search must not use legacy near-parallel gas cylinder filter')

    original = [value.copy() for value in (x, c, length, geom.mid, geom.dir, geom.halflen)]
    index = fw.WallIndex(geom, 1.2, RecordingIndex)
    root = np.sqrt(1.2 ** 2 - 1.1 ** 2)
    expected = np.array([[0., 0., 2 - root, 2 + root]])
    for batch in (1, 2, 65536):
        rows = index.intervals(x, c, length, batch=batch)
        np.testing.assert_allclose(rows, expected, atol=1e-14, rtol=0)
    check_columns(index.columns(x, c, length), all_straw_oracle(x, c, length, geom)[1])
    assert len(constructed) == 1 and index.outer is constructed[0]
    assert geom.radius == 1. and geom.wall_rmax == 1.2
    for before, after in zip(original, (x, c, length, geom.mid, geom.dir, geom.halflen)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("start,end,expected", [
    ([-1.1, 0., 0.], [-1.05, 0., 0.], [[0., 0., 0., .05]]),
    ([-2., 0., 0.], [-1.1, 0., 0.], [[0., 0., .8, .9]]),
    ([1.1, 0., 0.], [2., 0., 0.], [[0., 0., 0., .1]]),
    ([-1.1, 0., 0.], [0., 0., 0.], [[0., 0., 0., .1]]),
    ([0., 0., 0.], [1.1, 0., 0.], [[0., 0., 1., 1.1]]),
    ([-.5, 0., 0.], [.5, 0., 0.], []),
])
def test_partial_start_end_and_wholly_wall_or_gas_segments(start, end, expected):
    x, c, length = secants(start, end)
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    rows = index.intervals(x, c, length)
    check_rows(rows, length, 1)
    np.testing.assert_allclose(rows, np.asarray(expected).reshape(-1, 4), atol=1e-14, rtol=0)
    check_columns(index.columns(x, c, length), all_straw_oracle(x, c, length, single_straw())[1])


def test_axis_parallel_forward_reverse_caps_inside_and_outside():
    start = np.array([[1.1, 0., -3.], [1.1, 0., 3.], [1.1, 0., -1.], [1.1, 0., 3.],
                      [0., 0., -3.], [.5, 0., -3.], [1.3, 0., -3.]])
    end = np.array([[1.1, 0., 3.], [1.1, 0., -3.], [1.1, 0., 1.], [1.1, 0., 4.],
                    [0., 0., 3.], [.5, 0., 3.], [1.3, 0., 3.]])
    x, c, length = secants(start, end)
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    rows = index.intervals(x, c, length, batch=1)
    check_rows(rows, length, 1)
    np.testing.assert_allclose(rows, [[0., 0., 1., 5.], [1., 0., 1., 5.], [2., 0., 0., 2.]], atol=1e-14)
    check_columns(index.columns(x, c, length),
                  (np.array([4., 4., 2., 0., 0., 0., 0.]),
                   np.array([12., 12., 2., 0., 0., 0., 0.]),
                   np.array([124 / 3, 124 / 3, 8 / 3, 0., 0., 0., 0.])))


def test_oblique_caps_and_outer_tangent_inner_tangent_and_cap_miss():
    start = np.array([[-1.5, 0., .5], [-2., 0., 2.1], [-2., 1.2, 0.], [-2., 1., 0.],
                      [1.1, 0., 1.8]])
    end = np.array([[.5, 0., 2.5], [2., 0., 2.1], [2., 1.2, 0.], [2., 1., 0.], [0., 0., 4.]])
    x, c, length = secants(start, end)
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    rows = index.intervals(x, c, length)
    check_rows(rows, length, 1)
    root = np.sqrt(1.2 ** 2 - 1.)
    cap_exit = .2 * np.sqrt(1.1 ** 2 + 2.2 ** 2) / 2.2
    expected = [[0., 0., .3 * np.sqrt(2), .5 * np.sqrt(2)],
                [3., 0., 2 - root, 2 + root], [4., 0., 0., cap_exit]]
    np.testing.assert_allclose(rows, expected, atol=1e-13, rtol=0)
    # Analytic zero tangencies avoid discriminant roundoff in the independent oracle.
    a, b = np.zeros(len(length)), np.zeros(len(length))
    a[[0, 3, 4]], b[[0, 3, 4]] = np.asarray(expected)[:, 2], np.asarray(expected)[:, 3]
    expected_columns = (b - a, ((length - a) ** 2 - (length - b) ** 2) / 2,
                        ((length - a) ** 3 - (length - b) ** 3) / 3)
    check_columns(index.columns(x, c, length), expected_columns)


def mixed_rays():
    rng = np.random.default_rng(480215)
    start = rng.uniform([-12., -6., -7.], [12., 6., 19.], (173, 3))
    end = rng.uniform([-12., -6., -7.], [12., 6., 19.], (173, 3))
    fixed_start = np.array([[0., 0., -5.], [0., 0., 18.], [-8., -.2, 0.], [-8., -.2, 0.],
                            [0., 0., 0.], [0., 0., -2.], [0., 0., -2.], [0., 0., 5.],
                            [-3., 1.1, -3.], [20., 20., 20.], [0., 0., -.1]])
    fixed_end = np.array([[0., 0., 18.], [0., 0., -5.], [8., -.2, 0.], [-2., -.2, 0.],
                          [0., 0., .25], [0., 0., -1.], [0., 0., -.5], [5.6, 0., 9.2],
                          [-3., 1.1, 3.], [21., 20., 20.], [0., 0., .1]])
    return secants(np.concatenate((fixed_start, start)), np.concatenate((fixed_end, end)))


def test_tilted_unequal_z_panels_mixed_random_all_straw_oracle_and_batch_invariance():
    geom = tilted_straws()
    lo, hi, _ = crossings.panel_index(geom)
    assert np.unique(hi[:, 2] - lo[:, 2]).size > 1
    x, c, length = mixed_rays()
    expected, expected_columns = all_straw_oracle(x, c, length, geom)
    gas = oracle_crossings(x, c, length, geom)
    wall_lanes = np.unique(expected[:, 0].astype(int))
    gas_lanes = np.unique(gas[:, 0].astype(int))
    assert np.setdiff1d(wall_lanes, gas_lanes).size > 0
    assert np.setdiff1d(np.arange(len(length)), np.union1d(wall_lanes, gas_lanes)).size > 0
    assert 8 in wall_lanes and 8 not in gas_lanes and 9 not in wall_lanes and 10 not in wall_lanes
    index = fw.WallIndex(geom, 1.2, fs.ChordIndex)
    reference = index.intervals(x, c, length)
    for batch in (1, 2, 7, 65536):
        rows = index.intervals(x, c, length, batch=batch)
        check_rows(rows, length, geom.nstraws)
        np.testing.assert_array_equal(rows, reference)
        np.testing.assert_allclose(rows, expected, atol=2e-11, rtol=0)
    check_columns(index.columns(x, c, length), expected_columns, atol=2e-9)
    check_columns(fw.interval_columns(reference, length), expected_columns, atol=2e-9)


def test_segmentation_preserves_length_and_global_lever_moments():
    breaks = np.array([-2., -1.1, -1., -.9, .9, 1., 1.1, 2.])
    start, end = np.zeros((len(breaks) - 1, 3)), np.zeros((len(breaks) - 1, 3))
    start[:, 0], end[:, 0] = breaks[:-1], breaks[1:]
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    x, c, length = secants(start, end)
    ln, first, second = index.columns(x, c, length)
    whole = index.columns(*secants([-2., 0., 0.], [2., 0., 0.]))
    downstream = breaks[-1] - breaks[1:]
    combined = (ln.sum(), (first + downstream * ln).sum(),
                (second + 2 * downstream * first + downstream ** 2 * ln).sum())
    np.testing.assert_allclose(combined, np.asarray(whole)[:, 0], atol=1e-13, rtol=0)
    np.testing.assert_allclose(ln, [.1, .1, 0., 0., 0., .1, .1], atol=1e-14, rtol=0)
    rows = index.intervals(x, c, length, batch=1)
    offsets = breaks[:-1] - breaks[0]
    global_rows = rows.copy()
    global_rows[:, 2:] += offsets[rows[:, 0].astype(int), None]
    global_rows[:, 0] = 0
    check_columns(fw.interval_columns(global_rows, np.array([4.])), whole)


def test_forward_reversal_reflects_intervals_and_transforms_both_moments():
    geom = tilted_straws()
    x, c, length = mixed_rays()
    index = fw.WallIndex(geom, 1.2, fs.ChordIndex)
    forward = index.intervals(x, c, length, batch=3)
    reverse = index.intervals(x + c * length[:, None], -c, length, batch=5)
    check_rows(forward, length, geom.nstraws)
    check_rows(reverse, length, geom.nstraws)
    reflected = forward.copy()
    lane = forward[:, 0].astype(int)
    reflected[:, 2] = length[lane] - forward[:, 3]
    reflected[:, 3] = length[lane] - forward[:, 2]
    np.testing.assert_allclose(reverse, sorted_rows(reflected), atol=2e-11, rtol=0)
    ln, first, second = index.columns(x, c, length)
    expected = (ln, length * ln - first, length ** 2 * ln - 2 * length * first + second)
    check_columns(index.columns(x + c * length[:, None], -c, length), expected, atol=2e-9)


def test_interval_columns_accumulates_unsorted_duplicate_lanes_and_zero_lanes():
    rows = np.array([[2., 7., 1., 2.], [0., 3., 0., 1.], [2., 1., 3., 4.]])
    length = np.array([2., 7., 5., 9.])
    before = rows.copy()
    expected = (np.array([1., 0., 2., 0.]), np.array([1.5, 0., 5., 0.]),
                np.array([7 / 3, 0., 44 / 3, 0.]))
    check_columns(fw.interval_columns(rows, length), expected)
    np.testing.assert_array_equal(rows, before)


def test_empty_and_all_miss_batches_keep_full_lane_denominator():
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    for x, c, length in ((np.empty((0, 3)), np.empty((0, 3)), np.empty(0)),
                          secants([[10., 0., 0.], [0., 0., 3.]], [[11., 0., 0.], [1., 0., 3.]])):
        rows = index.intervals(x, c, length, batch=1)
        check_rows(rows, length, 1)
        assert rows.shape == (0, 4)
        expected = tuple(np.zeros(len(length)) for _ in range(3))
        check_columns(index.columns(x, c, length), expected)
        check_columns(fw.interval_columns(rows, length), expected)


@pytest.mark.parametrize('radius', [0., .9, -1., np.inf, np.nan])
def test_rejects_invalid_outer_radius(radius):
    with pytest.raises(ValueError, match='radius'):
        fw.WallIndex(single_straw(), radius, fs.ChordIndex)


@pytest.mark.parametrize('batch', [0, -1, True, 1.5])
def test_rejects_nonpositive_or_noninteger_batch(batch):
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    with pytest.raises(ValueError, match='batch'):
        index.intervals(*secants([-2., 0., 0.], [2., 0., 0.]), batch=batch)


def test_zero_wall_thickness_returns_no_material_columns():
    index = fw.WallIndex(single_straw(), 1., fs.ChordIndex)
    x, c, length = secants([-2., 0., 0.], [2., 0., 0.])
    assert index.intervals(x, c, length).shape == (0, 4)
    check_columns(index.columns(x, c, length), tuple(np.zeros(1) for _ in range(3)))


@pytest.mark.parametrize('radius', [1., 1.2])
def test_near_axis_nonzero_transverse_motion_crosses_inner_or_outer_boundary(radius):
    x = np.array([[radius + 1e-8, 0., -1.]])
    c = np.array([[-2e-8, 0., np.sqrt(1.-4e-16)]])
    length = np.array([2.])
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    forward = index.intervals(x, c, length)
    backward = index.intervals(x + c*length[:, None], -c, length)
    expected = [[0., 0., 0., .5]] if radius == 1. else [[0., 0., .5, 2.]]
    np.testing.assert_allclose(forward, expected, rtol=0, atol=2e-8)
    reflected = np.array(expected)
    reflected[:, 2], reflected[:, 3] = 2.-reflected[:, 3].copy(), 2.-reflected[:, 2].copy()
    np.testing.assert_allclose(backward, reflected, rtol=0, atol=2e-8)


def test_nearly_cap_parallel_nonzero_axial_motion_crosses_finite_straw_end():
    x = np.array([[-.5, 1.1, 2.+1e-13]])
    c = np.array([[1., 0., -2e-13]])
    length = np.array([.9])
    index = fw.WallIndex(single_straw(), 1.2, fs.ChordIndex)
    rows = index.intervals(x, c, length)
    cap = (2.-x[0, 2])/c[0, 2]
    outer = .5 + np.sqrt(1.2**2-1.1**2)
    np.testing.assert_allclose(rows, [[0., 0., cap, min(outer, .9)]], rtol=0, atol=1e-12)
