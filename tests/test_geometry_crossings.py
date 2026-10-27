"""Unit tests: ray-cylinder straw crossings and rigid alignment on a synthetic one-panel geometry."""
import sys
from pathlib import Path

import numpy as np
import pytest

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


def _plate_interior(thick=10.0):
    """TrackerInterior stub: a copper shell (rmin 100, rmax 100 + thick) about the z axis."""
    from mu2efastsim import world as Wd
    from trkfastsim import steps as S
    interior = S.TrackerInterior.__new__(S.TrackerInterior)
    tube = [("G4_Cu", np.eye(3), np.array([0.0, 0.0, 0.0]), 100.0, 100.0 + thick, 1000.0, 0.0, 2 * np.pi)]
    interior.shells = Wd.CoaxialShells(tube, axis_point=[0.0, 0.0, 0.0])
    interior.nsub = 8
    return interior


def _layers(ys, h=5000.0, n=401, pitch=6.0):
    """Straw layers along x at heights ys, spanning |z| <= pitch * n / 2 (one panel per layer)."""
    zs = pitch * (np.arange(n) - n // 2)
    mid = np.array([[0.0, y, z] for y in ys for z in zs])
    m = len(mid)
    up = np.repeat(np.arange(len(ys)), n)
    return G.StrawGeometry(plane=np.zeros(m, int), panel=up, straw=np.tile(np.arange(n), len(ys)), mid=mid,
                           dir=np.tile([1.0, 0, 0], (m, 1)), halflen=np.full(m, h), radius=R,
                           panel_R=np.tile(np.eye(3), (len(ys), 1, 1)), panel_t=np.zeros((len(ys), 3)),
                           plane_R=np.tile(np.eye(3), (len(ys), 1, 1)), plane_t=np.zeros((len(ys), 3)))


@pytest.mark.slow   # statistical check on 2e4-1e5 muons: several GB, tens of seconds
def test_interior_transport_kinks_after_material_and_matches_highland():
    from mu2efastsim import eloss as E, materials as M
    interior = _plate_interior(10.0)
    g = _layers([150.0, -150.0], pitch=4.0)   # overlapping straws: no gaps that would select small angles
    n = 20_000
    rng = np.random.default_rng(2)
    o = np.tile([0.0, 200.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    p = np.full(n, 2000.0)
    out = interior.transport(rng, o, d, p, g, tail=False, kink_xx0=1e9)   # support kinks only
    first = out["pos0"][:, 1] > 0
    assert np.allclose(out["dir"][first], [0, -1, 0]) and (out["lost"][first] == 0).all()
    # each muon's first crossing below the walls
    low = ~first & (np.r_[True, (out["pos0"][:-1, 1] > 0) | (out["it"][1:] != out["it"][:-1])])
    assert low.sum() == n and (out["lost"][low] > 5).all()
    # angle below both walls: Highland of 20 mm Cu (the incremental log term)
    th = np.arctan2(out["dir"][low, 2], -out["dir"][low, 1])
    th0 = E.highland_theta0(2000.0 - 15, 20.0 / E.radiation_length_mm(M.COPPER))
    assert abs(np.std(th) / th0 - 1) < 0.05
    # displacement in the lower layer: kinks at the straw entries, lever arms from the walls to y = -150
    assert np.std(out["pos0"][low, 2]) > 0.5


def test_interior_transport_without_material_is_a_straight_line():
    interior = _plate_interior(10.0)
    g = _layers([60.0, 30.0])              # both layers inside the shell's hole (r < 100)
    rng = np.random.default_rng(3)
    o = np.tile([0.0, 90.0, 0.3], (5, 1))
    d = np.tile([0.0, -1.0, 0.0], (5, 1))
    out = interior.transport(rng, o, d, np.full(5, 1500.0), g, tail=True, kink_xx0=1e9)
    xs = X.straw_crossings(o, d, g)
    assert len(out["it"]) == len(xs) and np.allclose(out["path"], xs[:, 2]) and (out["lost"] == 0).all()
    assert interior.last_stats["kinks"] == 0


@pytest.mark.slow   # statistical check on 2e4-1e5 muons: several GB, tens of seconds
def test_interior_transport_tail_adds_large_angles():
    interior = _plate_interior(10.0)
    g = _layers([150.0, -150.0])
    n = 100_000
    o = np.tile([0.0, 200.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    p = np.full(n, 2000.0)
    a = interior.transport(np.random.default_rng(4), o, d, p, g, tail=False, kink_xx0=1e9)
    b = interior.transport(np.random.default_rng(4), o, d, p, g, tail=True, kink_xx0=1e9)
    ta = np.abs(np.arctan2(a["dir"][a["pos0"][:, 1] < 0, 2], -a["dir"][a["pos0"][:, 1] < 0, 1]))
    tb = np.abs(np.arctan2(b["dir"][b["pos0"][:, 1] < 0, 2], -b["dir"][b["pos0"][:, 1] < 0, 1]))
    s = np.median(ta) / 0.6745
    assert abs(np.median(tb) / np.median(ta) - 1) < 0.03
    assert (tb > 5 * s).sum() > 10 * max((ta > 5 * s).sum(), 1)


def test_interior_transport_scatters_in_straws_and_air():
    """No support material: the straws alone scatter (Highland of their summed x/X0); air in the gaps adds to it."""
    from mu2efastsim import eloss as E, materials as M
    from trkfastsim import steps as S
    interior = _plate_interior(10.0)
    ys = np.linspace(95.0, -95.0, 10)                 # ten layers inside the shell's hole (r < 100)
    # 41 straws per layer (+-82 mm, the tracks start at z = 0 and deflect < 10 mm) and 5,000 tracks: the 401-straw,
    # 20,000-track version held every (track, straw) candidate in memory and was OOM-killed at 10 GB on a GPVM
    g = _layers(ys, n=41, pitch=4.0)
    g.wall_rmax, g.metal_t = R + 0.015, 5e-5
    n = 5_000
    o = np.tile([0.0, 99.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    p = np.full(n, 300.0)
    out = interior.transport(np.random.default_rng(6), o, d, p, g, tail=False)
    assert interior.last_stats["kinks"] >= n and (out["lost"] == 0).all()     # column x/X0 ~ 1.4e-3: >= one kink each
    lastrow = np.r_[out["it"][1:] != out["it"][:-1], True]
    th = np.arctan2(out["dir"][lastrow, 2], -out["dir"][lastrow, 1])
    # material up to the last kink: at most ~10 straws (gas + two walls), the last one behind the last kink
    xmax = 10 * (5.0 / E.radiation_length_mm(S.STRAW_GAS) + 0.03 / E.radiation_length_mm(M.MYLAR))
    th_max = E.highland_theta0(300.0, xmax)
    assert 0.6 * th_max < np.std(th) < 1.05 * th_max
    out_air = interior.transport(np.random.default_rng(6), o, d, p, g, tail=False, gap=M.AIR)
    th_air = np.arctan2(out_air["dir"][lastrow, 2], -out_air["dir"][lastrow, 1])
    assert np.std(th_air) > 1.05 * np.std(th)        # 190 mm air adds ~40 % to x/X0: ~+9 % in theta0
    # straight with kink_xx0 huge (no support crossed): no kinks
    interior.transport(np.random.default_rng(6), o[:100], d[:100], p[:100], g, kink_xx0=1e9)
    assert interior.last_stats["kinks"] == 0


@pytest.mark.slow   # statistical check on 2e4-1e5 muons: several GB, tens of seconds
def test_interior_transport_wire_hits_add_large_angles():
    """A thick sense wire on the straw axes: tracks that cross it scatter far more than those that miss it."""
    interior = _plate_interior(10.0)
    g = _layers([60.0, 20.0, -20.0, -60.0], pitch=6.0)
    g.wall_rmax, g.metal_t = R + 0.015, 5e-5
    g.wire_r, g.wire_plate_t = 0.5, 0.01               # 1 mm tungsten wire: hit by ~1/5 of tracks through a straw
    n = 20_000
    rng = np.random.default_rng(8)
    o = np.c_[np.zeros(n), np.full(n, 95.0), rng.uniform(-2.5, 2.5, n)]
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    out = interior.transport(np.random.default_rng(9), o, d, np.full(n, 1000.0), g, tail=False, wire=True)
    ref = interior.transport(np.random.default_rng(9), o, d, np.full(n, 1000.0), g, tail=False, wire=False)
    assert interior.last_stats["wire_hits"] == 0
    last = lambda r: np.r_[r["it"][1:] != r["it"][:-1], True]
    a = np.abs(np.arctan2(out["dir"][last(out), 2], -out["dir"][last(out), 1]))
    b = np.abs(np.arctan2(ref["dir"][last(ref), 2], -ref["dir"][last(ref), 1]))
    assert np.quantile(a, 0.95) > 3 * np.quantile(b, 0.95)


def test_interior_transport_does_not_truncate_chords_after_a_kink():
    """A kink displaces the track at the next straw's entry; the restart must not begin inside that straw's gas."""
    interior = _plate_interior(10.0)
    g = _layers(np.linspace(95.0, -95.0, 12), n=41, pitch=5.3)
    g.wall_rmax, g.metal_t = R + 0.015, 5e-5
    n = 3000
    rng = np.random.default_rng(12)
    o = np.c_[rng.uniform(-50, 50, n), np.full(n, 99.0), rng.uniform(-30, 30, n)]
    th, ph = rng.uniform(0, 0.6, n), rng.uniform(0, 2 * np.pi, n)
    d = np.c_[np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph)]
    out = interior.transport(np.random.default_rng(1), o, d, np.full(n, 300.0), g, tail=True)
    assert interior.last_stats["kinks"] > n
    L = np.linalg.norm(out["pos1"] - out["pos0"], axis=1)
    js, u = out["js"], out["dir"]
    a, w = g.dir[js], out["pos0"] - g.mid[js]
    cx = np.cross(u, a)
    b = np.abs(np.einsum("ij,ij->i", w, cx)) / np.maximum(np.linalg.norm(cx, axis=1), 1e-12)
    ua = np.sqrt(1 - np.einsum("ij,ij->i", u, a) ** 2)
    chord = 2 * np.sqrt(np.clip(R ** 2 - b ** 2, 0, None)) / ua
    # each step is the full chord of its own straight line (a few clipped by the straw ends: none here)
    assert np.mean(L < chord - 0.01) < 0.002 and abs(L.mean() / chord.mean() - 1) < 5e-4


def test_to_plane_marks_muons_outside_the_margin_not_alive(monkeypatch):
    """Muons whose line misses the footprint margin are not transported and must not reach a straw untouched."""
    import pandas as pd
    from trkfastsim import steps as S
    g = _layers([0.0], n=11, pitch=6.0)
    monkeypatch.setattr(S, "plane_y", lambda geom: 100.0)
    monkeypatch.setattr(S.tr, "world_loss", lambda rng, o, d, p, w, e, **k: (
        p, np.zeros(len(p)), np.ones(len(p), bool), np.zeros((len(p), 1)), np.zeros((len(p), 1)), p[:, None]))
    monkeypatch.setattr(S.tr, "msc_to_plane", lambda rng, o, d, p, y, *a, **k: (
        o + d * ((o[:, 1] - y) / -d[:, 1])[:, None], d, (o[:, 1] - y) / -d[:, 1]))
    # muon 0 straight down onto the straws; muon 1 far outside (z = 5 m at the plane: the straws span |z| <= 33 mm and
    # 10 m in x); muon 2 upward
    gen = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 2, 3], "pdg": 13,
                        "x": [0.0, 0.0, 0.0], "y": [1000.0, 1000.0, 1000.0], "z": [0.0, 5000.0, 0.0],
                        "px": [0.0, 0.0, 0.0], "py": [-1000.0, -1000.0, 1000.0], "pz": [0.0, 0.0, 0.0], "t": 0.0})
    _, o, d, p, L, alive = S.to_plane(gen, g, np.random.default_rng(1), world=None, near_margin=100.0, msc_tail=False)
    assert alive.tolist() == [True, False, False]


def test_near_tracker_accepts_shallow_lines_that_cross_the_stack_below_the_plane():
    from trkfastsim import steps as S
    g = _layers([0.0, -300.0], n=11, pitch=6.0)        # stack y -300..0, |z| <= 33 mm
    # shallow muon: at the plane y = 100 it is at z = 1000 (outside a 100 mm margin), at y = -300 it is at z = 0
    d = np.array([[0.0, -0.3713907, -0.9284767]])        # dz/dy = 2.5
    o = np.array([[0.0, 100.0, 1000.0]]) - d * 1000.0
    assert S.near_tracker(o, d, g, 100.0, margin=100.0).tolist() == [True]
    o2 = np.array([[0.0, 100.0, 3000.0]]) - d * 1000.0    # crosses y = -300 at z = 2000: never near
    assert S.near_tracker(o2, d, g, 100.0, margin=100.0).tolist() == [False]
