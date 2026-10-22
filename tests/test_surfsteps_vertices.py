import numpy as np
import pandas as pd

from trkfastsim import surfaces as S


def _verts(points, p=2000.0):
    """One event: a polyline through `points` (detector frame) with a kink at each interior point."""
    pts = np.asarray(points, float)
    d = np.diff(pts, axis=0)
    path = np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]
    u = np.r_[d / np.linalg.norm(d, axis=1)[:, None], d[-1:] / np.linalg.norm(d[-1])]
    v = pd.DataFrame({"run": 1400, "subrun": 1, "event": 3, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
                      "ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2], "p": p, "path": path, "t0": 5.0, "beta": 1.0})
    return v


def _steps():
    return pd.DataFrame({"run": [1400], "subrun": [1], "event": [3], "sp": [1], "t": [10.0], "x0": [600.0], "y0": [100.0],
                         "z0": [500.0], "x1": [600.0], "y1": [95.0], "z1": [500.0], "px": [0.0], "py": [-2000.0], "pz": [0.0]})


def test_kinked_path_crossings_follow_the_kink():
    # straight down to y = 800 (inside TT_Outer at x = 600? r = 1000 > 850: outside), kink towards x, then down again
    v = _verts([(600, 3000, 500), (600, 700, 500), (300, -700, 500), (300, -3000, 500)])
    o = S.surface_steps(_steps(), v)
    assert sorted(o.surf.tolist()) == [4, 4]
    c = np.c_[[(o[f"{a}0"] + o[f"{a}1"]) / 2 for a in "xyz"]].T
    assert np.allclose(np.hypot(c[:, 0], c[:, 1]), S.R_OUT, atol=1e-3)
    # the exit crossing lies on the post-kink segment (x between 300 and 600), not on the pre-kink straight line x = 600
    assert np.all(c[:, 0] < 600.0)
    assert (o.t.diff().dropna() > 0).all()


def test_events_without_straw_steps_are_dropped():
    v = _verts([(600, 3000, 500), (600, -3000, 500)])
    v2 = v.assign(event=4)
    o = S.surface_steps(_steps(), pd.concat([v, v2]))
    assert set(o.event) == {3}
