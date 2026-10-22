"""Support-scattering calibration of TrackerInterior (steps.interior_msc_scale / interior_msc_var_factor / slab_group)."""
import numpy as np

from mu2efastsim import materials
from trkfastsim import steps as st

CAL = {"points": [(10.0, (0.8, 0.0, 0.0)), (24.0, (0.9, 0.02, 0.0))], "clip": [0.5, 1.2]}


def test_scale_interpolates_in_ln_x0_and_clamps():
    s_cu = st.interior_msc_scale(CAL, materials.BRONZE_C938, 1.0, 1000.0)   # X0 10.9 g/cm2: near the first point
    s_al = st.interior_msc_scale(CAL, materials.ALUMINIUM_6061, 1.0, 1000.0)
    s_pb = st.interior_msc_scale(CAL, materials.LEAD, 1.0, 1000.0)         # below the first X0: clamped
    assert 0.8 < s_cu < 0.82 and abs(s_al - 0.9) < 2e-3 and abs(s_pb - 0.8) < 1e-12
    big = {"points": [(10.0, (5.0, 0.0, 0.0))], "clip": [0.5, 1.2]}
    assert st.interior_msc_scale(big, materials.LEAD, 1.0, 1000.0) == 1.2


def test_var_factor_weights_by_xx0_share():
    X = np.array([2.0, 2.0])
    xm = {"BronzeC938": np.array([1.0, 0.0])}
    f = st.interior_msc_var_factor(CAL, xm, X, np.array([500.0, 500.0]))
    s = st.interior_msc_scale(CAL, materials.BRONZE_C938, 1.0, 500.0)
    assert np.allclose(f, [0.5 * s * s + 0.5, 1.0])


def test_slab_group_width_scales_and_off_is_unchanged():
    p = np.full(40000, 1000.0)
    one = {"points": [(10.0, (0.75, 0.0, 0.0))], "clip": [0.5, 1.2]}
    a0 = st.slab_group(np.random.default_rng(3), materials.LEAD, 10.0, p, tail=False)
    a1 = st.slab_group(np.random.default_rng(3), materials.LEAD, 10.0, p, tail=False, msc_cal=one)
    r = np.std(a1[2]) / np.std(a0[2])
    assert abs(r - 0.75) < 1e-6                     # same draws, the core width scaled exactly
    b0 = st.slab_group(np.random.default_rng(3), materials.LEAD, 10.0, p, tail=False)
    assert np.array_equal(a0[2], b0[2])
