"""Analytic first-slice navigation checks on synthetic rings, not full extracted geometry."""
import numpy as np
import pytest

from mu2efastsim.world import CoaxialShells
from trkfastsim.materialpath import support_intervals
from trkfastsim.steps import InteriorColumn


def tube(material="Al", ri=2., ro=3., z=0., hz=10., dp=2 * np.pi, center=None, rotation=None):
    return (material, np.eye(3) if rotation is None else rotation,
            np.array([0., 0., z]) if center is None else np.asarray(center), ri, ro, hz, 0., dp)


def shells(*tubes):
    return CoaxialShells(tubes, axis_point=[0., 0., 0.])


def ray(geometry, o=(-5., 0., 0.), d=(1., 0., 0.), length=10.):
    return support_intervals(geometry, [o], [d], length)


def assert_spans(frame, expected):
    assert frame.material.tolist() == [m for _, _, m in expected]
    np.testing.assert_allclose(frame[["s0_mm", "s1_mm"]], [[a, b] for a, b, _ in expected], atol=1e-12)


def test_near_far_sides_with_explicit_vacuum():
    result = ray(shells(tube()))
    assert result.columns.tolist() == ["lane", "s0_mm", "s1_mm", "material"]
    assert result.lane.tolist() == [0] * 5
    assert_spans(result, [(0, 2, "vacuum"), (2, 3, "Al"), (3, 7, "vacuum"),
                          (7, 8, "Al"), (8, 10, "vacuum")])


def test_start_inside_and_bounded_stop():
    assert_spans(ray(shells(tube()), o=(2.5, 0, 0), length=2),
                 [(0, .5, "Al"), (.5, 2, "vacuum")])
    assert_spans(ray(shells(tube()), length=2.5), [(0, 2, "vacuum"), (2, 2.5, "Al")])


def test_oblique_chords_and_tangencies():
    a, b = np.sqrt(8), np.sqrt(3)
    assert_spans(ray(shells(tube()), o=(-5, 1, 0)),
                 [(0, 5-a, "vacuum"), (5-a, 5-b, "Al"), (5-b, 5+b, "vacuum"),
                  (5+b, 5+a, "Al"), (5+a, 10, "vacuum")])
    assert_spans(ray(shells(tube()), o=(-5, 3, 0)), [(0, 10, "vacuum")])
    a = np.sqrt(5)
    assert_spans(ray(shells(tube()), o=(-5, 2, 0)),
                 [(0, 5-a, "vacuum"), (5-a, 5+a, "Al"), (5+a, 10, "vacuum")])


def test_z_clipping_and_reverse_direction():
    g = shells(tube(hz=.5))
    root2 = np.sqrt(2)
    expected = [(0, 1.5*root2, "vacuum"), (1.5*root2, 2*root2, "Al"),
                (2*root2, 6, "vacuum")]
    assert_spans(ray(g, o=(-4, 0, -2), d=(1/root2, 0, 1/root2), length=6), expected)
    assert_spans(ray(g, o=(-4, 0, 2), d=(1/root2, 0, -1/root2), length=6), expected)


def test_axis_parallel_inside_hole_outside_and_z_parallel_miss():
    g = shells(tube(hz=1))
    assert_spans(ray(g, o=(2.5, 0, -3), d=(0, 0, 1), length=6),
                 [(0, 2, "vacuum"), (2, 4, "Al"), (4, 6, "vacuum")])
    for radius in (0, 1, 4):
        assert_spans(ray(g, o=(radius, 0, -3), d=(0, 0, 1), length=6), [(0, 6, "vacuum")])
    for z in (1, 2):
        assert_spans(ray(g, o=(-5, 0, z)), [(0, 10, "vacuum")])


def test_sequential_materials_gaps_and_touching_shells():
    g = shells(tube("Cu", z=3, hz=1), tube("Al", z=0, hz=1), tube("Al", z=5, hz=1))
    assert_spans(ray(g, o=(2.5, 0, -2), d=(0, 0, 1), length=9),
                 [(0, 1, "vacuum"), (1, 3, "Al"), (3, 4, "vacuum"),
                  (4, 6, "Cu"), (6, 8, "Al"), (8, 9, "vacuum")])
    g = shells(tube("Al", ri=0, ro=2), tube("Cu", ri=2, ro=3))
    assert_spans(ray(g, o=(0, 0, 0), length=4),
                 [(0, 2, "Al"), (2, 3, "Cu"), (3, 4, "vacuum")])


def test_multiple_lanes_zero_length_empty_and_wrapper():
    g = InteriorColumn(shells(tube()), calo=None)
    result = support_intervals(g, [[-5, 0, 0], [-5, 0, 0], [0, 0, 0]],
                               [[1, 0, 0]] * 3, [10, 0, 1])
    assert result.lane.tolist() == [0] * 5 + [2]
    assert_spans(result[result.lane == 2], [(0, 1, "vacuum")])
    empty = support_intervals(g, np.empty((0, 3)), np.empty((0, 3)), [])
    zero = ray(g, length=0)
    assert empty.empty and zero.empty
    assert empty.dtypes.to_dict() == zero.dtypes.to_dict() == result.dtypes.to_dict()
    assert support_intervals(g, np.empty((0, 3)), np.empty((0, 3)), 1.).empty
    assert_spans(ray(shells()), [(0, 10, "vacuum")])


def test_translated_rotated_coaxial_frame():
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    center = np.array([20., 30., 40.])
    g = CoaxialShells([tube(center=center, rotation=rotation)], center, axis_dir=[1., 0., 0.])
    result = ray(g, o=center + rotation @ [-5., 0., 0.], d=rotation @ [1., 0., 0.])
    assert_spans(result, [(0, 2, "vacuum"), (2, 3, "Al"), (3, 7, "vacuum"),
                          (7, 8, "Al"), (8, 10, "vacuum")])


@pytest.mark.parametrize("material", ["Al", "Cu"])
def test_reject_overlapping_volumes_even_for_same_material_and_missed_rays(material):
    g = shells(tube(), tube(material, ri=2.5, ro=4))
    with pytest.raises(ValueError, match="overlapping support volumes"):
        ray(g, o=(100, 0, 0))
    with pytest.raises(ValueError, match="overlapping support volumes"):
        ray(shells(tube(), tube(material)))


def test_reject_unsupported_geometry_even_for_empty_rays():
    g = shells(tube(dp=np.pi))
    with pytest.raises(ValueError, match="sectors"):
        support_intervals(g, np.empty((0, 3)), np.empty((0, 3)), [])
    with pytest.raises(ValueError, match="noncoaxial"):
        ray(shells(tube(center=[5., 0., 0.])))
    g = InteriorColumn(shells(tube()))
    g.calo = object()
    with pytest.raises(ValueError, match="calorimeter"):
        ray(g)
    with pytest.raises(TypeError, match="CoaxialShells"):
        ray(object())
    assert "Al" in ray(shells(tube(dp=2*np.pi - .5e-9))).material.tolist()


@pytest.mark.parametrize("length", [-1, np.inf, np.nan, [1, 2], [[1]]])
def test_reject_bad_lengths(length):
    with pytest.raises(ValueError, match="length"):
        ray(shells(), length=length)


@pytest.mark.parametrize("direction", [(0, 0, 0), (2, 0, 0), (1.000001, 0, 0), (np.nan, 0, 0)])
def test_reject_nonunit_or_nonfinite_directions(direction):
    with pytest.raises(ValueError, match="unit|finite"):
        ray(shells(), d=direction)


def test_reject_invalid_origins_shapes_and_empty_scalar_length():
    with pytest.raises(ValueError, match="finite"):
        ray(shells(), o=(np.inf, 0, 0))
    for o, d in [([0, 0, 0], [1, 0, 0]), ([[0, 0, 0]], [[1, 0]]), ([], [])]:
        with pytest.raises(ValueError, match="shapes"):
            support_intervals(shells(), o, d, 1)
    with pytest.raises(ValueError, match="finite"):
        support_intervals(shells(), np.empty((0, 3)), np.empty((0, 3)), np.nan)


def test_lengths_match_current_column_and_partition_each_lane():
    g = shells(tube("Al", ri=1, ro=2, z=-3, hz=1), tube("Cu", ri=1, ro=2, z=2, hz=2),
               tube("Al", ri=3, ro=4, z=0, hz=5), tube("C", ri=0, ro=.5, z=0, hz=5))
    rng = np.random.default_rng(716)
    origins = rng.uniform(-6, 6, size=(300, 3))
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    origins = np.vstack([origins, [[1.5, 0, -6], [-6, 0, 0], [0, 0, 0]]])
    directions = np.vstack([directions, [[0, 0, 1], [1, 0, 0], [1, 0, 0]]])
    lengths = np.r_[rng.uniform(0, 20, size=300), [12, 12, 0]]
    result = support_intervals(g, origins, directions, lengths)
    column = g.column(origins, directions, lengths)
    for material in set(result.material) - {"vacuum"}:
        selected = result[result.material == material]
        actual = np.bincount(selected.lane, weights=selected.s1_mm - selected.s0_mm, minlength=len(origins))
        np.testing.assert_allclose(actual, column[material], atol=1e-11, rtol=1e-11)
    assert set(column) == set(result.material) - {"vacuum"}
    for lane, spans in result.groupby("lane", sort=False):
        assert spans.s0_mm.iloc[0] == 0
        assert spans.s1_mm.iloc[-1] == lengths[lane]
        np.testing.assert_array_equal(spans.s1_mm.to_numpy()[:-1], spans.s0_mm.to_numpy()[1:])
        assert (spans.s1_mm > spans.s0_mm).all()
    assert 302 not in result.lane.to_numpy()


# General tube navigation is a separate API: the full-ring gate above stays intact.
from trkfastsim.materialpath import tube_intervals
from mu2efastsim.world import column_tubes


def sector(p0, dp, **kwargs):
    t = tube(dp=dp, **kwargs)
    return (*t[:6], p0, dp)


def general_ray(geometry, o=(-5., 0., 0.), d=(1., 0., 0.), length=10.):
    return tube_intervals(geometry, [o], [d], length)


@pytest.mark.parametrize('p0,dp,expected', [
    (-np.pi/4, np.pi/2, [(0, 5, 'vacuum'), (5, 8, 'Al'), (8, 10, 'vacuum')]),
    (7*np.pi/4, np.pi/2, [(0, 5, 'vacuum'), (5, 8, 'Al'), (8, 10, 'vacuum')]),
    (np.pi/4, 3*np.pi/2, [(0, 2, 'vacuum'), (2, 5, 'Al'), (5, 10, 'vacuum')]),
    (-np.pi/4, 3*np.pi/2, [(0, 2, 'vacuum'), (2, 8, 'Al'), (8, 10, 'vacuum')]),
])
def test_general_sectors_wrap_reflex_axis_and_reverse(p0, dp, expected):
    g = shells(sector(p0, dp, ri=0))
    assert_spans(general_ray(g), expected)
    reverse = [(10-b, 10-a, mat) for a, b, mat in expected[::-1]]
    assert_spans(general_ray(g, o=(5, 0, 0), d=(-1, 0, 0)), reverse)


def test_general_exact_phi_planes_and_annular_hole():
    # y=1 enters the wedge y<=x at x=1, not a sampled approximation.
    g = shells(sector(0, np.pi/4, ri=0, ro=4))
    end = 5 + np.sqrt(15)
    assert_spans(general_ray(g, o=(-5, 1, 0)),
                 [(0, 6, 'vacuum'), (6, end, 'Al'), (end, 10, 'vacuum')])
    g = shells(sector(0, np.pi/4, ri=2, ro=4))
    start = 5 + np.sqrt(3)
    assert_spans(general_ray(g, o=(-5, 1, 0)),
                 [(0, start, 'vacuum'), (start, end, 'Al'), (end, 10, 'vacuum')])


def test_general_boundary_parallel_axis_crossing():
    g = shells(sector(0, np.pi/2, ri=0))
    assert_spans(general_ray(g), [(0, 5, 'vacuum'), (5, 8, 'Al'), (8, 10, 'vacuum')])
    assert_spans(general_ray(g, o=(1, 1, -12), d=(0, 0, 1), length=24),
                 [(0, 2, 'vacuum'), (2, 22, 'Al'), (22, 24, 'vacuum')])
    assert_spans(general_ray(g, o=(-1, 1, -12), d=(0, 0, 1), length=24),
                 [(0, 24, 'vacuum')])


def test_general_rotated_translated_noncoax_and_mixed_chronology():
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    center = np.array([20., 30., 40.])
    t = sector(-np.pi/4, np.pi/2, ri=0, center=center, rotation=rotation)
    g = shells(t)
    assert len(g.other) == 1
    assert_spans(general_ray(g, o=center+rotation@[-5., 0., 0.], d=rotation@[1., 0., 0.]),
                 [(0, 5, 'vacuum'), (5, 8, 'Al'), (8, 10, 'vacuum')])
    g = shells(tube('Cu', ri=0, ro=1, center=[8., 0., 0.]),
               sector(-np.pi/4, np.pi/2, ri=0, ro=2),
               tube('C', ri=0, ro=1, center=[4., 0., 0.]))
    assert_spans(general_ray(InteriorColumn(g, calo=None), length=16),
                 [(0, 5, 'vacuum'), (5, 7, 'Al'), (7, 8, 'vacuum'), (8, 10, 'C'),
                  (10, 12, 'vacuum'), (12, 14, 'Cu'), (14, 16, 'vacuum')])


def test_general_random_full_rings_identical_to_existing():
    g = shells(tube('Al', ri=1, ro=2, hz=2), tube('Cu', ri=3, ro=4, hz=5))
    rng = np.random.default_rng(880)
    o = rng.uniform(-6, 6, (500, 3))
    d = rng.normal(size=o.shape)
    d /= np.linalg.norm(d, axis=1)[:, None]
    length = rng.uniform(0, 20, len(o))
    import pandas as pd
    pd.testing.assert_frame_equal(tube_intervals(g, o, d, length), support_intervals(g, o, d, length),
                                  check_exact=False, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('noncoax', [False, True])
def test_general_sector_core_sampling_converges(noncoax):
    rng = np.random.default_rng(178)
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]) if noncoax else np.eye(3)
    center = np.array([5., -3., 2.]) if noncoax else np.zeros(3)
    t = sector(5.7, 4.2, ri=.7, ro=3., hz=2, center=center, rotation=rotation)
    g = shells(t)
    o = rng.uniform(-4, 4, (80, 3)) @ rotation.T + center
    d = rng.normal(size=o.shape)
    d = (d / np.linalg.norm(d, axis=1)[:, None]) @ rotation.T
    result = tube_intervals(g, o, d, 12.)
    solid = result[result.material == 'Al']
    exact = np.bincount(solid.lane, weights=solid.s1_mm-solid.s0_mm, minlength=len(o))
    coarse = g.column(o, d, 12., nsub=8)['Al']
    fine = g.column(o, d, 12., nsub=8192)['Al']
    assert np.linalg.norm(fine-exact) < np.linalg.norm(coarse-exact)/100
    np.testing.assert_allclose(fine, exact, atol=24/8192, rtol=0)
    np.testing.assert_allclose(column_tubes([t], o, d, 12., nsub=8192)['Al'], exact, atol=24/8192, rtol=0)
    total = np.bincount(result.lane, weights=result.s1_mm-result.s0_mm, minlength=len(o))
    np.testing.assert_allclose(total, 12.)


def test_general_overlap_checks_only_requested_rays_and_keeps_disjoint_sectors():
    g = shells(tube(), tube('Cu'))
    with pytest.raises(ValueError, match='overlapping'):
        general_ray(g)
    assert_spans(general_ray(g, o=(100, 0, 0)), [(0, 10, 'vacuum')])
    with pytest.raises(ValueError, match='overlapping'):
        general_ray(shells(tube(), tube()))
    with pytest.raises(ValueError, match='overlapping'):
        general_ray(shells(tube(), tube('Cu', ri=0, ro=3, center=[2., 0., 0.])))
    g = shells(sector(-np.pi/4, np.pi/2), sector(3*np.pi/4, np.pi/2, material='Cu'))
    assert_spans(general_ray(g), [(0, 2, 'vacuum'), (2, 3, 'Cu'), (3, 7, 'vacuum'),
                                 (7, 8, 'Al'), (8, 10, 'vacuum')])


@pytest.mark.parametrize('api', [tube_intervals])
def test_general_invalid_inputs_and_empty(api):
    g = shells()
    for o, d, length in [([[np.nan, 0, 0]], [[1, 0, 0]], 1),
                         ([[0, 0, 0]], [[np.inf, 0, 0]], 1),
                         ([[0, 0, 0]], [[2, 0, 0]], 1),
                         ([[0, 0, 0]], [[1, 0, 0]], -1),
                         ([[0, 0, 0]], [[1, 0, 0]], np.inf),
                         ([[0, 0, 0]], [[1, 0, 0]], [1, 2]),
                         ([0, 0, 0], [1, 0, 0], 1)]:
        with pytest.raises(ValueError):
            api(g, o, d, length)
    empty = api(g, np.empty((0, 3)), np.empty((0, 3)), [])
    assert empty.empty
    assert empty.dtypes.to_dict() == support_intervals(g, np.empty((0, 3)), np.empty((0, 3)), []).dtypes.to_dict()
    assert api(g, [[0, 0, 0]], [[1, 0, 0]], 0).empty
    with pytest.raises(ValueError, match='finite'):
        api(g, np.empty((0, 3)), np.empty((0, 3)), np.nan)
    with pytest.raises(TypeError, match='CoaxialShells'):
        general_ray(object())
    wrapped = InteriorColumn(g)
    wrapped.calo = object()
    with pytest.raises(ValueError, match='calorimeter'):
        general_ray(wrapped)


@pytest.mark.parametrize('field,value', [('dp', np.nan), ('dp', 0), ('dp', 7), ('p0', np.inf),
                                         ('z0', -np.inf), ('rmin', -1), ('rmax', np.inf)])
def test_general_reject_invalid_geometry_even_without_rays(field, value):
    g = shells(tube())
    if field in ('rmin', 'rmax'):
        g.groups[0][field] = value
    else:
        g.groups[0][field][0] = value
    with pytest.raises(ValueError, match='bounds'):
        tube_intervals(g, np.empty((0, 3)), np.empty((0, 3)), [])


def test_general_antiparallel_coax_sector_and_endcaps():
    rotation = np.diag([1., -1., -1.])
    center = np.array([0., 0., 4.])
    g = shells(sector(0., np.pi/2, ri=0, ro=3, hz=1, center=center, rotation=rotation))
    assert not g.other
    # Local first quadrant becomes global fourth quadrant; local z is reversed.
    assert_spans(general_ray(g, o=(1, -1, 1), d=(0, 0, 1), length=6),
                 [(0, 2, 'vacuum'), (2, 4, 'Al'), (4, 6, 'vacuum')])
    assert_spans(general_ray(g, o=(1, 1, 1), d=(0, 0, 1), length=6), [(0, 6, 'vacuum')])
    assert_spans(general_ray(g, o=(-5, -1, 5)), [(0, 10, 'vacuum')])
    o = np.array([[-5, -1, 4], [-5, -1, 4], [1, 1, 4]])
    result = tube_intervals(g, o, [[1, 0, 0]]*3, [10, 0, 2])
    assert set(result.lane) == {0, 2}
    assert_spans(result[result.lane == 2], [(0, 2, 'vacuum')])


def test_general_half_sector_axis_parallel_and_noncoax_full_ring():
    g = shells(sector(0, np.pi, ri=0))
    # Both sides of the diameter belong to the closed half-sector.
    assert_spans(general_ray(g), [(0, 2, 'vacuum'), (2, 8, 'Al'), (8, 10, 'vacuum')])
    center = np.array([7., 2., -1.])
    g = shells(tube(center=center))
    assert len(g.other) == 1
    assert_spans(general_ray(g, o=center+[-5, 0, 0]),
                 [(0, 2, 'vacuum'), (2, 3, 'Al'), (3, 7, 'vacuum'),
                  (7, 8, 'Al'), (8, 10, 'vacuum')])


@pytest.mark.parametrize('bad', ['rotation_finite', 'center_finite', 'rotation_orthogonal', 'bounds'])
def test_general_noncoax_geometry_validation(bad):
    g = shells(tube(center=[5., 0., 0.]))
    t = list(g.other[0])
    if bad == 'rotation_finite':
        t[1][0, 0] = np.nan
    elif bad == 'center_finite':
        t[2][0] = np.inf
    elif bad == 'rotation_orthogonal':
        t[1][0, 0] = 2
    else:
        t[5] = 0
    g.other[0] = tuple(t)
    with pytest.raises(ValueError, match='finite|orthogonal|bounds'):
        tube_intervals(g, np.empty((0, 3)), np.empty((0, 3)), [])


def test_general_reflex_sector_two_disconnected_spans_and_reverse():
    # Exclude the upper 90-degree wedge: y=1 leaves at x=-1 and reenters at x=1.
    g = shells(sector(3*np.pi/4, 3*np.pi/2, ri=0, ro=4))
    root = np.sqrt(15)
    expected = [(0, 5-root, 'vacuum'), (5-root, 4, 'Al'), (4, 6, 'vacuum'),
                (6, 5+root, 'Al'), (5+root, 10, 'vacuum')]
    assert_spans(general_ray(g, o=(-5, 1, 0)), expected)
    assert_spans(general_ray(g, o=(5, 1, 0), d=(-1, 0, 0)), expected)


def test_general_touching_tubes_are_separate_and_nearly_full_matches_core():
    g = shells(tube('Al', ri=0, ro=2), tube('Al', ri=2, ro=3))
    assert_spans(general_ray(g, o=(0, 0, 0), length=4),
                 [(0, 2, 'Al'), (2, 3, 'Al'), (3, 4, 'vacuum')])
    g = shells(sector(1.2, 2*np.pi-.5e-9))
    assert_spans(general_ray(g), [(0, 2, 'vacuum'), (2, 3, 'Al'), (3, 7, 'vacuum'),
                                 (7, 8, 'Al'), (8, 10, 'vacuum')])


def test_phi_face_inclusion_is_stable_under_frame_conversion():
    from trkfastsim.materialpath import tube_intervals
    p0, dp = 2., .5
    g = shells(("Al", np.eye(3), np.zeros(3), 0., 3., 10., p0, dp))
    for angle in (p0, p0 + dp):
        out = tube_intervals(g, [[np.cos(angle), np.sin(angle), 0.]], [[0., 0., 1.]], 1.)
        assert out.material.tolist() == ["Al"]
    other = shells(("Al", np.eye(3), np.zeros(3), 0., 3., 10., p0, dp),
                   ("Cu", np.eye(3), np.zeros(3), 0., 3., 10., p0 + dp, .5))
    with pytest.raises(ValueError, match="overlap"):
        tube_intervals(other, [[np.cos(p0+dp), np.sin(p0+dp), 0.]], [[0., 0., 1.]], 1.)
