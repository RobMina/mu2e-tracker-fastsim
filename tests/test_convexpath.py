"""Synthetic stored-halfspace checks, not GDML or production transport validation."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim.materials import AIR, ALUMINIUM
from mu2efastsim.world import PieceSet, SignedPieceSet, column
from trkfastsim.convexpath import convex_intervals


def box(material="Al", x0=1., x1=5., y0=-1., y1=1., z0=-1., z1=1., cut=None):
    lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
    planes = np.column_stack([np.vstack([np.eye(3), -np.eye(3)]), np.r_[hi, -lo]])
    if cut is not None:
        planes = np.vstack([planes, cut])
    return material, planes, lo, hi


def signed(*pieces):
    return SignedPieceSet(pieces, cell=2.)


def ray(geometry, o=(0., 0., 0.), d=(1., 0., 0.), length=6.):
    return convex_intervals(geometry, [o], [d], length)


def assert_spans(frame, expected):
    assert frame.material.tolist() == [m for _, _, m in expected]
    np.testing.assert_allclose(frame[["s0_mm", "s1_mm"]], [[a, b] for a, b, _ in expected], atol=1e-12)


def assert_partition(frame, lengths):
    lane = frame.lane.to_numpy()
    first = np.r_[True, lane[1:] != lane[:-1]]
    last = np.r_[lane[:-1] != lane[1:], True]
    np.testing.assert_array_equal(frame.s0_mm.to_numpy()[first], 0.)
    np.testing.assert_array_equal(frame.s1_mm.to_numpy()[last], np.asarray(lengths)[lane[last]])
    np.testing.assert_array_equal(frame.s1_mm.to_numpy()[:-1][~first[1:]],
                                  frame.s0_mm.to_numpy()[1:][~first[1:]])
    assert (frame.s1_mm > frame.s0_mm).all()
    assert frame.lane.is_monotonic_increasing


def test_positive_disjoint_pieces_and_same_material_gap():
    g = PieceSet([box("Al", 1, 2), box("Cu", 3, 4), box("Al", 4, 5)])
    result = ray(g)
    assert result.columns.tolist() == ["lane", "s0_mm", "s1_mm", "material"]
    assert result.lane.tolist() == [0] * 6
    assert_spans(result, [(0, 1, "vacuum"), (1, 2, "Al"), (2, 3, "vacuum"),
                          (3, 4, "Cu"), (4, 5, "Al"), (5, 6, "vacuum")])


def test_mother_minus_daughter_with_two_existing_material_keys():
    mother, daughter = box(AIR.name), box(ALUMINIUM.name, 2, 3)
    g = signed((1, *mother), (-1, *box(AIR.name, 2, 3)), (1, *daughter))
    assert_spans(ray(g), [(0, 1, "vacuum"), (1, 2, AIR.name), (2, 3, ALUMINIUM.name),
                          (3, 5, AIR.name), (5, 6, "vacuum")])
    assert_spans(ray(g, o=(2.5, 0, 0), length=1), [(0, .5, ALUMINIUM.name), (.5, 1, AIR.name)])


def test_hollow_box_and_extra_halfspace_cut():
    g = signed((1, *box()), (-1, *box(x0=2, x1=4, y0=-.5, y1=.5)))
    assert_spans(ray(g), [(0, 1, "vacuum"), (1, 2, "Al"), (2, 4, "vacuum"),
                          (4, 5, "Al"), (5, 6, "vacuum")])
    assert_spans(ray(g, o=(0, .75, 0)), [(0, 1, "vacuum"), (1, 5, "Al"), (5, 6, "vacuum")])
    cut = PieceSet([box(cut=[1., 1., 0., 3.])])
    assert_spans(ray(cut, o=(0, .5, 0)), [(0, 1, "vacuum"), (1, 2.5, "Al"), (2.5, 6, "vacuum")])


def test_repeated_coincident_events_aggregate_before_cumulative_sum():
    g = signed((1, *box()), (-1, *box(x0=2, x1=4)),
               (1, *box(x0=2, x1=3)), (1, *box(x0=3, x1=4)))
    assert_spans(ray(g), [(0, 1, "vacuum"), (1, 2, "Al"), (2, 3, "Al"),
                          (3, 4, "Al"), (4, 5, "Al"), (5, 6, "vacuum")])
    cancel = signed((1, *box()), (-1, *box()))
    assert_spans(ray(cancel), [(0, 6, "vacuum")])


def test_touching_positive_pieces_are_not_multiplicity_two():
    g = PieceSet([box(x0=1, x1=3), box(x0=3, x1=5)])
    assert_spans(ray(g), [(0, 1, "vacuum"), (1, 3, "Al"), (3, 5, "Al"), (5, 6, "vacuum")])


@pytest.mark.parametrize("geometry", [
    PieceSet([box(), box(x0=2, x1=3)]),
    signed((1, *box()), (1, *box())),
    signed((-1, *box())),
    signed((1, *box(x0=1, x1=3)), (-1, *box(x0=2, x1=4))),
    signed((1, *box(x0=1, x1=5)), (-1, *box(x0=0, x1=.5))),
    signed((1, *box("Cu")), (-1, *box("Al"))),
])
def test_invalid_multiplicities_rejected_before_column_clipping(geometry):
    with pytest.raises(ValueError, match="occupancy|multiplicity"):
        ray(geometry)


@pytest.mark.parametrize("geometry", [PieceSet([box(), box("Cu", 2, 3)]),
                                       signed((1, *box()), (1, *box("Cu", 2, 3)))])
def test_cross_material_overlap_rejected(geometry):
    with pytest.raises(ValueError, match="cross-material"):
        ray(geometry)


def test_pointwise_negative_is_not_hidden_by_positive_total_column():
    g = signed((1, *box(x0=1, x1=4)), (-1, *box(x0=4, x1=5)))
    np.testing.assert_array_equal(g.column([[0, 0, 0]], [[1, 0, 0]], 6)["Al"], [2.])
    with pytest.raises(ValueError, match="occupancy"):
        ray(g)
    assert_spans(ray(g, length=3), [(0, 1, "vacuum"), (1, 3, "Al")])
    assert_spans(ray(g, o=(0, 2, 0)), [(0, 6, "vacuum")])


def test_parallel_rays_closed_faces_and_zero_measure_contacts():
    g = PieceSet([box()])
    assert_spans(ray(g, o=(0, 1, 0)), [(0, 1, "vacuum"), (1, 5, "Al"), (5, 6, "vacuum")])
    assert_spans(ray(g, o=(0, 1.01, 0)), [(0, 6, "vacuum")])
    assert_spans(ray(g, o=(3, -2, 0), d=(0, 1, 0), length=4),
                 [(0, 1, "vacuum"), (1, 3, "Al"), (3, 4, "vacuum")])
    assert_spans(ray(g, length=1), [(0, 1, "vacuum")])
    assert_spans(ray(g, o=(5, 0, 0)), [(0, 6, "vacuum")])
    assert_spans(ray(g, o=(6, 0, 0), d=(-1, 0, 0)),
                 [(0, 1, "vacuum"), (1, 5, "Al"), (5, 6, "vacuum")])
    tangent = PieceSet([box(cut=[1, 1, 0, 1])])
    assert_spans(ray(tangent), [(0, 6, "vacuum")])


def test_oblique_rotated_translated_convex_piece():
    angle = .73
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    shift = np.array([13., -27., 6.])
    mat, planes, lo, hi = box(cut=[1, 1, 0, 3])
    normals = planes[:, :3] @ rotation.T
    moved = np.column_stack([normals, planes[:, 3] + normals @ shift])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = corners @ rotation.T + shift
    g = PieceSet([(mat, moved, corners.min(axis=0), corners.max(axis=0))])
    assert_spans(ray(g, o=shift, d=rotation @ [1, 0, 0]),
                 [(0, 1, "vacuum"), (1, 3, "Al"), (3, 6, "vacuum")])


def test_empty_zero_length_and_no_pieces_have_stable_schema():
    for g in (PieceSet([]), signed(), PieceSet([box()])):
        empty = convex_intervals(g, np.empty((0, 3)), np.empty((0, 3)), [])
        assert empty.empty and ray(g, length=0).empty
        assert empty.dtypes.to_dict() == ray(g).dtypes.to_dict()
        assert convex_intervals(g, np.empty((0, 3)), np.empty((0, 3)), 1.).empty
    assert_spans(ray(PieceSet([])), [(0, 6, "vacuum")])
    result = convex_intervals(PieceSet([box()]), [[0, 0, 0]] * 3, [[1, 0, 0]] * 3, [0, 2, .5])
    assert result.lane.tolist() == [1, 1, 2]
    assert_spans(result, [(0, 1, "vacuum"), (1, 2, "Al"), (0, .5, "vacuum")])


@pytest.mark.parametrize("length", [-1, np.inf, np.nan, [1, 2], [[1]]])
def test_invalid_lengths(length):
    with pytest.raises(ValueError, match="length"):
        ray(PieceSet([]), length=length)


@pytest.mark.parametrize("direction", [(0, 0, 0), (2, 0, 0), (1.000001, 0, 0), (np.nan, 0, 0)])
def test_invalid_directions(direction):
    with pytest.raises(ValueError, match="unit|finite"):
        ray(PieceSet([]), d=direction)


def test_invalid_shapes_origins_and_wrapper_types():
    for o, d in [([0, 0, 0], [1, 0, 0]), ([[0, 0, 0]], [[1, 0]]), ([], [])]:
        with pytest.raises(ValueError, match="shapes"):
            convex_intervals(PieceSet([]), o, d, 1)
    with pytest.raises(ValueError, match="finite"):
        ray(PieceSet([]), o=(np.inf, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        convex_intervals(PieceSet([]), np.empty((0, 3)), np.empty((0, 3)), np.nan)
    for unsupported in ([], [box()], object()):
        with pytest.raises(TypeError, match="PieceSet"):
            ray(unsupported)


def test_packing_omissions_and_invalid_geometry_rejected():
    with pytest.raises(ValueError, match="omitted"):
        ray(signed((2, *box())))
    g = PieceSet([box()])
    g.P[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ray(g)


@pytest.mark.parametrize("use_signed", [False, True])
def test_grid_equivalence_and_core_columns_for_clean_geometry(use_signed, monkeypatch):
    import trkfastsim.convexpath as navigator
    monkeypatch.setattr(navigator, "_PAIR_CHUNK", 37)
    pieces = [box("Al", 1, 3), box("Cu", 4, 5), box("Al", -4, -2)]
    if use_signed:
        g = signed(*[(1, *p) for p in pieces], (-1, *box("Al", 1.5, 2.5)),
                   (1, *box("Cu", 1.5, 2.5)))
        sets = [ps for _, ps in g.sets]
    else:
        g = PieceSet(pieces)
        sets = [g]
    rng = np.random.default_rng(901)
    o = rng.uniform(-6, 6, size=(250, 3))
    d = rng.normal(size=(250, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    o = np.vstack([o, [-6, 0, 0], [6, 0, 0], [0, 0, 0]])
    d = np.vstack([d, [1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    length = np.r_[rng.uniform(0, 20, 250), 12, 12, 0]
    for ps in sets:
        ps.grid = None
    plain = convex_intervals(g, o, d, length)
    for ps in sets:
        ps.build_grid(cell=1.)
    calls = []
    original = PieceSet._candidates

    def candidates(self, origins, directions, lengths):
        calls.append(len(origins))
        return original(self, origins, directions, lengths)

    monkeypatch.setattr(PieceSet, "_candidates", candidates)
    gridded = convex_intervals(g, o, d, length)
    assert calls
    pd.testing.assert_frame_equal(plain, gridded)
    expected = g.column(o, d, length) if use_signed else column(g, o, d, length)
    for material, values in expected.items():
        spans = plain[plain.material == material]
        actual = np.bincount(spans.lane, weights=spans.s1_mm - spans.s0_mm, minlength=len(o))
        np.testing.assert_allclose(actual, values, atol=1e-11, rtol=1e-11)
    assert_partition(plain, length)
    assert 252 not in plain.lane.to_numpy()
