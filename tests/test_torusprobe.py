"""Synthetic actual-torus bounds; no GDML, remote data, or transport admission."""
import numpy as np
import pytest

from trkfastsim.torusprobe import TorusProbeResult, torus_intervals


BASE = dict(rtor=10.0, rmin=0.0, rmax=2.0)


def probe(o, d, length, **kwargs):
    return torus_intervals(np.asarray(o, dtype=float).reshape(-1, 3),
                           np.asarray(d, dtype=float).reshape(-1, 3), length,
                           **(BASE | kwargs))


def encloses(result, truth, gap=2e-6):
    truth = np.broadcast_to(truth, result.lower_mm.shape)
    assert np.all(result.lower_mm <= truth), (result.lower_mm, truth)
    assert np.all(result.upper_mm >= truth), (result.upper_mm, truth)
    if gap is not None:
        assert np.all(result.upper_mm - result.lower_mm < gap)
    rows = result.intervals
    assert (rows.s1_mm > rows.s0_mm).all()
    assert set(rows.status) <= {"inside", "unresolved"}
    assert rows.equals(rows.sort_values(["lane", "s0_mm"], kind="stable").reset_index(drop=True))
    previous = rows.groupby("lane").s1_mm.shift()
    assert (previous.isna() | (rows.s0_mm >= previous)).all()
    assert result.counts["inside_cells"] == result.inside_count.sum()
    assert result.counts["unresolved_cells"] == result.unresolved_count.sum()


def test_axial_and_transverse_full_ring():
    result = probe([[10, 0, -5], [-15, 0, 0], [10, 0, 0], [0, 0, -5]],
                   [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]], [10, 30, 5, 10])
    assert isinstance(result, TorusProbeResult)
    encloses(result, [4, 8, 2, 0])
    assert not result.counts["cap_reached"]
    assert result.unresolved_count[:3].min() > 0  # no claimed exact crossings
    assert result.upper_mm[3] == 0


def test_hollow_torus():
    result = probe([[10, 0, -5], [-15, 0, 0], [10, 0, 0]],
                   [[0, 0, 1], [1, 0, 0], [0, 0, 1]], [10, 30, 5], rmin=1)
    encloses(result, [2, 4, 1], gap=3e-6)


def test_requested_segment_inside_is_not_extended_to_solid_exit():
    result = probe([[10, 0, 0]], [[0, 0, 1]], 0.25)
    encloses(result, 0.25, gap=1e-12)
    assert result.unresolved_count[0] == 0
    assert result.intervals.s0_mm.min() == 0
    assert result.intervals.s1_mm.max() == 0.25


def test_clear_misses_and_broad_phase_hole():
    result = probe([[100, 100, 100], [0, 0, -5], [13, 0, -5]],
                   [[1, 0, 0], [0, 0, 1], [0, 0, 1]], 10)
    encloses(result, 0)
    assert result.intervals.empty
    assert result.counts["aabb_candidates"] == 1  # central hole isn't a slab miss


def test_tangent_retains_bounded_unknown_not_a_fabricated_crossing():
    result = probe([[12, 0, -5]], [[0, 0, 1]], 10, min_width_mm=1e-6)
    encloses(result, 0, gap=0.01)
    assert result.upper_mm[0] > 0
    assert result.lower_mm[0] == 0
    assert result.unresolved_count[0] > 0
    assert not result.counts["cap_reached"]


def test_small_grazing_chords_are_not_missed():
    # Different near-tangent offsets, including a chord shorter than a coarse
    # terminal cell: upper bounds must include it even if no inside proof exists.
    x = np.array([12 - 1e-5, 12 - 1e-8, 12 - 1e-11])
    result = probe(np.column_stack((x, np.zeros(3), np.full(3, -5))),
                   np.tile([0, 0, 1], (3, 1)), 10, min_width_mm=1e-5)
    truth = 2 * np.sqrt((2 - (x - 10)) * (2 + (x - 10)))
    encloses(result, truth, gap=0.02)
    assert (result.upper_mm > 0).all()


@pytest.mark.parametrize("start,delta,angles,expected", [
    (0, np.pi / 2, [np.pi / 4, 3*np.pi / 4, -np.pi / 4], [4, 0, 0]),
    (7*np.pi / 4, np.pi / 2, [0, np.pi / 2, np.pi], [4, 0, 0]),
    (3*np.pi / 4, 3*np.pi / 2, [0, np.pi / 2, np.pi, 3*np.pi / 2], [4, 0, 4, 4]),
    (-np.pi / 2, np.pi, [0, np.pi], [4, 0]),
])
def test_quarter_wrapped_reflex_and_half_sectors(start, delta, angles, expected):
    angles = np.asarray(angles)
    o = np.column_stack((10 * np.cos(angles), 10 * np.sin(angles), np.full(len(angles), -5)))
    result = probe(o, np.tile([0, 0, 1], (len(o), 1)), 10, startphi=start, deltaphi=delta)
    encloses(result, expected)


def test_sector_plane_crossing_splits_actual_solid_and_axis_is_not_inside():
    # At y=1 the +x lobe only is in the first quadrant. The angular-plane
    # crossing x=0 is inside the radial hole and must not become material.
    result = probe([[-15, 1, 0], [0, 0, -5]], [[1, 0, 0], [0, 0, 1]],
                   [30, 10], deltaphi=np.pi / 2)
    encloses(result, [np.sqrt(12**2 - 1) - np.sqrt(8**2 - 1), 0])


def test_boundary_plane_parallel_is_unknown_not_guessed_inside():
    result = probe([[10, 0, -5]], [[0, 0, 1]], 10, deltaphi=np.pi / 2,
                   max_nodes=1000)
    encloses(result, 4, gap=None)
    assert result.lower_mm[0] == 0
    assert result.counts["cap_reached"]
    assert result.unresolved_count[0] > 0


def test_parent_rigid_transform_then_local_probe():
    angle = 0.37
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([200., -80., 33.])
    local_o = np.array([[10., 0., -5.], [-15., 0., 0.]])
    local_d = np.array([[0., 0., 1.], [1., 0., 0.]])
    world_o, world_d = local_o @ rotation.T + translation, local_d @ rotation.T
    result = probe((world_o - translation) @ rotation, world_d @ rotation, [10, 30])
    encloses(result, [4, 8])


def test_empty_zero_and_zero_thickness():
    empty = probe([], [], [])
    assert empty.lower_mm.shape == empty.upper_mm.shape == (0,)
    assert empty.intervals.empty
    zero = probe([[10, 0, 0]], [[0, 0, 1]], 0)
    encloses(zero, 0)
    assert zero.intervals.empty
    surface = probe([[10, 0, -5]], [[0, 0, 1]], 10, rmin=2, min_width_mm=1e-5)
    encloses(surface, 0, gap=1e-4)
    assert surface.lower_mm[0] == 0
    assert surface.unresolved_count[0] > 0


@pytest.mark.parametrize("limit,reason", [
    ({"max_nodes": 0}, "max_nodes"), ({"max_nodes": 1}, "max_nodes"),
    ({"max_rounds": 0}, "max_rounds"), ({"max_rounds": 1}, "max_rounds"),
    ({"max_active": 1}, "max_active"),
])
def test_work_caps_never_mean_transmission(limit, reason):
    result = probe([[-15, 0, 0]], [[1, 0, 0]], 30, **limit)
    encloses(result, 8, gap=None)
    assert result.counts["cap_reached"]
    assert reason in result.counts["unresolved_reasons"]
    assert result.upper_mm[0] > 0
    assert result.counts["evaluated_nodes"] <= limit.get("max_nodes", 1_000_000)
    assert result.counts["peak_active_cells"] <= limit.get("max_active", 131072)


def test_initial_frontier_cap_preserves_every_lane():
    result = probe([[10, 0, -5], [10, 0, -5]], [[0, 0, 1], [0, 0, 1]], 10, max_active=1)
    encloses(result, [4, 4], gap=None)
    assert (result.unresolved_count == 1).all()
    assert result.counts["evaluated_nodes"] == 0


def test_distant_origins_clip_before_adaptive_work_and_degrade_honestly():
    result = probe([[-1e8, 0, 0]], [[1, 0, 0]], 1e8 + 20, min_width_mm=1e-4)
    encloses(result, 8, gap=0.01)
    assert result.counts["rounds"] < 30
    # Cancellation guard dominates at this distance. No precise length claim.
    huge = probe([[-1e15, 0, 0]], [[1, 0, 0]], 1e15 + 20, max_nodes=100)
    encloses(huge, 8, gap=None)
    assert huge.unresolved_count[0] > 0


def test_extreme_parallel_slabs_are_conservative():
    # Tiny nonzero transverse component can overflow a slab quotient; that
    # merely makes the corresponding halfspace nonrestrictive on this ray.
    result = probe([[10, 0, -5]], [[1e-320, 0, 1]], 10)
    encloses(result, 4)


@pytest.mark.parametrize("kwargs", [
    {"rtor": 2}, {"rmin": -1}, {"rmin": 3}, {"rmax": -1},
    {"rtor": np.inf}, {"rtor": 1e151}, {"startphi": np.nan},
    {"startphi": 1e7}, {"deltaphi": 0}, {"deltaphi": -1},
    {"deltaphi": 2*np.pi + 1e-10}, {"min_width_mm": 0},
    {"min_width_mm": np.nan}, {"max_nodes": -1}, {"max_nodes": 1.5},
    {"max_nodes": True}, {"max_rounds": -1}, {"max_active": 0}, {"max_lanes": 0},
])
def test_invalid_geometry_and_limits(kwargs):
    with pytest.raises(ValueError):
        probe([[10, 0, -5]], [[0, 0, 1]], 10, **kwargs)


@pytest.mark.parametrize("o,d,length", [
    ([[0, 0, 0]], [[0, 0, 2]], 1), ([[0, 0, 0]], [[0, 0, 0]], 1),
    ([[np.inf, 0, 0]], [[0, 0, 1]], 1), ([[0, 0, 0]], [[np.nan, 0, 1]], 1),
    ([[0, 0, 0]], [[0, 0, 1]], -1), ([[0, 0, 0]], [[0, 0, 1]], np.inf),
    ([[0, 0, 0]], [[0, 0, 1]], [1, 2]), ([[0, 0]], [[0, 1]], 1),
    ([[1e151, 0, 0]], [[0, 0, 1]], 1), ([[0, 0, 0]], [[0, 0, 1]], 1e151),
])
def test_invalid_rays(o, d, length):
    with pytest.raises(ValueError):
        torus_intervals(o, d, length, **BASE)


@pytest.mark.parametrize("start,delta", [(0, np.pi/2), (np.pi/2, 3*np.pi/2)])
def test_sector_boundary_crossing_within_material(start, delta):
    result = probe([[10, -10, 0]], [[0, 1, 0]], 20,
                   startphi=start, deltaphi=delta)
    encloses(result, np.sqrt(12**2 - 10**2), gap=3e-6)
    assert result.lower_mm[0] > 0
    assert result.unresolved_count[0] > 0


def test_vectorized_random_axial_analytic_hollow_chords():
    rng = np.random.default_rng(1497)
    radial = rng.uniform(7.5, 12.5, 128)
    angles = rng.uniform(-np.pi, np.pi, len(radial))
    offsets = np.abs(radial - 10)
    truth = 2 * (np.sqrt(np.maximum(0, 4 - offsets**2))
                 - np.sqrt(np.maximum(0, 1 - offsets**2)))
    o = np.column_stack((radial * np.cos(angles), radial * np.sin(angles),
                         np.full(len(radial), -5.)))
    result = probe(o, np.tile([0, 0, 1.], (len(o), 1)), 10, rmin=1)
    # Analytic values use radial values before sin/cos conversion; this differs
    # by only ~1e-13 mm and is contained in the retained numerical brackets.
    encloses(result, truth, gap=8e-6)
    assert not result.counts["cap_reached"]


def test_vectorized_random_transverse_analytic_chords():
    rng = np.random.default_rng(491)
    z = rng.uniform(-1.99, 1.99, 128)
    y = rng.uniform(-13, 13, len(z))
    outer = np.sqrt(4 - z*z)
    width = lambda r: 2 * (np.sqrt(np.maximum(0, (10+r)**2 - y*y))
                           - np.sqrt(np.maximum(0, (10-r)**2 - y*y)))
    truth = width(outer)
    result = probe(np.column_stack((np.full(len(z), -15), y, z)),
                   np.tile([1., 0, 0], (len(z), 1)), 30)
    encloses(result, truth, gap=8e-6)
    assert not result.counts["cap_reached"]


def test_angular_roundoff_never_clears_a_boundary_neighbour():
    angles = np.array([-1e-14, 0., 1e-14])
    o = np.column_stack((10 * np.cos(angles), 10 * np.sin(angles), np.full(3, -5.)))
    result = probe(o, np.tile([0, 0, 1.], (3, 1)), 10,
                   deltaphi=np.pi/2, max_nodes=1000)
    encloses(result, [0, 4, 4], gap=None)
    assert (result.upper_mm >= 4).all()
    assert (result.unresolved_count > 0).all()


@pytest.mark.parametrize("guard", [-1., np.inf, np.nan])
def test_caller_transform_guard_validation(guard):
    with pytest.raises(ValueError):
        probe([[10., 0., -3.]], [[0., 0., 1.]], 6., coordinate_guard_mm=guard)


def test_transform_guard_preserves_unknown_coordinate_error():
    unguarded = probe([[10., 0., -3.]], [[0., 0., 1.]], 6.)
    guarded = probe([[10., 0., -3.]], [[0., 0., 1.]], 6., coordinate_guard_mm=.01, max_nodes=10000)
    assert guarded.lower_mm[0] <= unguarded.lower_mm[0]
    assert guarded.upper_mm[0] >= unguarded.upper_mm[0]
    assert guarded.counts["coordinate_guard_mm"] == .01


def test_near_full_sector_not_silently_promoted_to_full():
    delta = 2*np.pi - 1e-6
    angle = 2*np.pi - 0.5e-6
    result = probe([[10*np.cos(angle), 10*np.sin(angle), -5]], [[0, 0, 1]], 10,
                   deltaphi=delta)
    encloses(result, 0)
    assert result.intervals.empty
