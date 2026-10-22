import numpy as np
import pandas as pd

from trkfastsim import surfaces as S


def _steps(points, p=3000.0):
    """One event, primary muon steps along a straight line through the given (entry, exit) detector-frame points."""
    rows = []
    for k, (a, b) in enumerate(points):
        u = (np.asarray(b) - np.asarray(a)) / np.linalg.norm(np.asarray(b) - np.asarray(a))
        rows.append({"run": 1400, "subrun": 1, "event": 7, "sp": 1, "t": 10.0 + k,
                     "x0": a[0], "y0": a[1], "z0": a[2], "x1": b[0], "y1": b[1], "z1": b[2],
                     "px": u[0] * p, "py": u[1] * p, "pz": u[2] * p})
    return pd.DataFrame(rows)


def _on_surface(o):
    c = np.c_[[(o[f"{a}0"] + o[f"{a}1"]) / 2 for a in "xyz"]].T
    r = np.hypot(c[:, 0], c[:, 1])
    s = o.surf.to_numpy()
    zs = np.select([s == 0, s == 1, s == 2], [S.Z_FRONT, S.Z_MID, S.Z_BACK], np.nan)
    rs = np.where(s == 3, S.R_IN, S.R_OUT)
    return np.where(s <= 2, np.abs(c[:, 2] - zs), np.abs(r - rs))


def test_vertical_muon_crosses_outer_twice_and_no_plane():
    # vertical (along -y) at x = 600 mm, z = 500 mm: outside the inner cylinder, inside the outer
    st = _steps([((600, 300, 500), (600, 295, 500)), ((600, -295, 500), (600, -300, 500))])
    o = S.surface_steps(st)
    assert sorted(o.surf.tolist()) == [4, 4]
    assert np.all(_on_surface(o) < 1e-3)
    assert (o.t.diff().dropna() > 0).all()


def test_vertical_muon_through_the_hole_crosses_inner_too():
    st = _steps([((100, 300, 500), (100, 295, 500))])
    o = S.surface_steps(st)
    assert sorted(o.surf.tolist()) == [3, 3, 4, 4]


def test_inclined_muon_crosses_planes():
    # along z with a slope in y, from z = -3000 to +3000 at x = 600: crosses front, mid, back planes (r < 810 at mid)
    a, b = np.array([600.0, 200.0, -1000.0]), np.array([600.0, 100.0, 1000.0])
    st = _steps([(a, a + 0.01 * (b - a)), (b - 0.01 * (b - a), b)])
    o = S.surface_steps(st)
    assert {0, 1, 2}.issubset(set(o.surf.tolist()))
    assert np.all(_on_surface(o) < 1e-3)


def test_only_primary_and_columns():
    st = _steps([((600, 300, 500), (600, 295, 500))])
    st2 = st.assign(sp=1000)
    o = S.surface_steps(pd.concat([st, st2]))
    assert list(o.columns) == S.COLS and (o.sp == 1).all()
