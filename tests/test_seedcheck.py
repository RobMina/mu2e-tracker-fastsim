import numpy as np
import pandas as pd

from mu2efastsim import eloss
from trkfastsim.analyses import seedcheck as sc


def test_crv_pass_needs_visible_edep_and_momentum_window():
    s = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 2, 3, 3], "edep": [0.2, 0.05, 0.2, 0.3],
                      "px0": 0.0, "py0": [-100.0, -100.0, -40.0, -2e6], "pz0": 0.0})
    assert sc.crv_pass_from(s).event.tolist() == [1]


def test_projected_over_theta0_is_angle_over_highland():
    u = np.array([[0.0, -1.0, 0.0]])
    th = 0.01
    v = np.array([[np.sin(th), -np.cos(th), 0.0]])
    r = sc.projected_over_theta0(v, u, np.array([1000.0]), np.array([1.0]))
    t0 = eloss.highland_theta0(np.array([1000.0]), np.array([1.0]))[0]
    assert np.isclose(np.abs(r).max(), np.tan(th) / t0) and np.isclose(np.abs(r).min(), 0.0, atol=1e-12)


def test_width_table_reports_g4_over_fast():
    n = 200
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"xx0": 0.5, "p_seed": 500.0, "rg1": rng.normal(0, 0.7, n), "rg2": rng.normal(0, 0.7, n),
                       "rf1": rng.normal(0, 1.0, n), "rf2": rng.normal(0, 1.0, n)})
    (row,) = sc.width_table(df)
    assert row["n"] == n and 0.55 < row["g4_over_fast_sigma"] < 0.85
