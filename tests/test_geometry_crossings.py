"""Unit tests: ray-cylinder straw crossings and rigid alignment on a synthetic one-panel geometry."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trkfastsim import crossings as X                                   # noqa: E402
from trkfastsim import geometry as G                                    # noqa: E402

R = 2.5235


def _geom(n=3, pitch=10.0, h=500.0):
    """n straws along x (axis = x), centred at z = 0, spaced in y by `pitch`; one panel."""
    mid = np.c_[np.zeros(n), pitch * np.arange(n), np.zeros(n)]
    z = np.zeros(n, int)
    return G.StrawGeometry(plane=z, panel=z, straw=np.arange(n), mid=mid, dir=np.tile([1.0, 0, 0], (n, 1)),
                           halflen=np.full(n, h), radius=R, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
                           plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def test_vertical_track_through_axis_has_diameter_chord():
    g = _geom()
    c = X.straw_crossings(np.array([[0.0, 0.0, 50.0]]), np.array([[0.0, 0.0, -1.0]]), g)
    assert len(c) == 1 and c[0, 1] == 0 and np.isclose(c[0, 3] - c[0, 2], 2 * R)


def test_offset_track_chord_length():
    g = _geom()
    b = 1.5
    c = X.straw_crossings(np.array([[0.0, b, 50.0]]), np.array([[0.0, 0.0, -1.0]]), g)
    assert np.isclose(c[0, 3] - c[0, 2], 2 * np.sqrt(R * R - b * b))


def test_slanted_track_scales_with_angle_and_hits_several_straws():
    g = _geom(pitch=4.0)
    th = 1.2   # 4 mm pitch: the neighbour axis is 4 cos(th) = 1.45 mm < R from the track
    d = np.array([[0.0, np.sin(th), -np.cos(th)]])
    c = X.straw_crossings(np.array([[0.0, -50 * np.tan(th), 50.0]]) , d, g)
    assert len(c) >= 2 and np.all(np.diff(c[:, 2]) > 0)


def test_track_along_axis_is_clipped_by_the_ends():
    g = _geom(h=100.0)
    c = X.straw_crossings(np.array([[-500.0, 0.0, 0.5]]), np.array([[1.0, 0.0, 0.0]]), g)
    assert len(c) == 1 and np.isclose(c[0, 3] - c[0, 2], 200.0)


def test_miss_beyond_the_end():
    g = _geom(h=100.0)
    c = X.straw_crossings(np.array([[150.0, 0.0, 50.0]]), np.array([[0.0, 0.0, -1.0]]), g)
    assert len(c) == 0


def test_tracker_translation_moves_every_straw():
    g = _geom()
    a = G.apply_alignment(g, tracker=(1.0, 2.0, 3.0, 0, 0, 0))
    assert np.allclose(a.mid - g.mid, [1, 2, 3]) and np.allclose(a.dir, g.dir)


def test_tracker_rotation_about_origin():
    g = _geom()
    a = G.apply_alignment(g, tracker=(0, 0, 0, 0, 0, np.pi / 2))
    assert np.allclose(a.dir[0], [0, 1, 0]) and np.allclose(a.mid[1], [-10, 0, 0])


def test_straw_id_packing():
    assert G.straw_id(35, 5, 95) == (35 << 10) | (5 << 7) | 95


def test_point_straw_inside_wall_and_outside():
    g = _geom(pitch=10.0, h=100.0)
    pts = np.array([[0.0, 0.0, 1.0], [50.0, 10.0, -2.0], [0.0, 0.0, 2.6], [0.0, 5.0, 0.0], [150.0, 0.0, 0.0]])
    assert X.point_straw(pts, g).tolist() == [0, 1, -1, -1, -1]


def test_nearest_straw_snaps_gap_points_and_rejects_far_ones():
    g = _geom(pitch=10.0, h=100.0)
    pts = np.array([[0.0, 4.0, 0.0], [0.0, 6.0, 1.0], [0.0, 0.0, 30.0], [500.0, 0.0, 0.0]])
    js, d = X.nearest_straw(pts, g, pad=10.0)
    assert js.tolist() == [0, 1, -1, -1] and np.isclose(d[0], 4.0)


def test_interior_loss_accumulates_along_the_muon():
    """TrackerInterior.loss_to_straws: segment columns between consecutive straws, cumulative per muon."""
    from mu2efastsim import world as Wd
    from trkfastsim import steps as S
    interior = S.TrackerInterior.__new__(S.TrackerInterior)
    # one copper slab-like shell: coaxial with z, rmin 100, rmax 110 (10 mm wall), long in z
    tube = [("G4_Cu", np.eye(3), np.array([0.0, 0.0, 0.0]), 100.0, 110.0, 1000.0, 0.0, 2 * np.pi)]
    interior.shells = Wd.CoaxialShells(tube, axis_point=[0.0, 0.0, 0.0])
    interior.nsub = 8
    rng = np.random.default_rng(1)
    o = np.array([[0.0, 200.0, 0.0]])
    d = np.array([[0.0, -1.0, 0.0]])
    p = np.array([3000.0])
    # straws at 50 mm (y = 150: before the wall), 95 mm (y = 105: in the upper wall, 5 mm crossed), 320 mm
    # (y = -120: past the lower wall, both walls crossed)
    it = np.array([0, 0, 0])
    s_in = np.array([50.0, 95.0, 320.0])
    lost, stats = interior.loss_to_straws(rng, o, d, p, it, s_in)
    # 10 mm Cu = 8.96 g/cm2: Bethe mean ~13 MeV, one Urban sample per wall sits around it (MPV lower)
    assert lost[0] == 0 and 2 < lost[1] < 15 and lost[2] > lost[1] and 10 < lost[2] < 50
    assert abs(stats["G4_Cu"] - 20.0) < 0.5
