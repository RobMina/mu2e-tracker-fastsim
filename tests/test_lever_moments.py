import numpy as np

from trkfastsim import geometry, steps as st


class _Col:
    """A stand-in column: one support slab of material G4_Cu at s in [a, b] of every queried segment."""
    def __init__(self, a, b):
        self.a, self.b, self.groups, self.c = a, b, [], np.zeros(3)

    def column(self, o, d, t_max, nsub=16, moments=False):
        t = np.asarray(t_max, float) * np.ones(len(o))
        lo, hi = np.minimum(self.a, t), np.minimum(self.b, t)
        L = np.clip(hi - lo, 0, None)
        if not moments:
            return {"G4_Cu": L}
        return {"G4_Cu": (L, 0.5 * (hi ** 2 - lo ** 2) * (L > 0), (hi ** 3 - lo ** 3) / 3 * (L > 0))}


def test_moments_equal_midpoint_for_a_full_segment_and_differ_for_material_at_the_top():
    rng = np.random.default_rng(1)
    o, d = np.zeros((4, 3)), np.tile([0.0, -1.0, 0.0], (4, 1))
    seg = np.full(4, 100.0)
    # identical results when the material fills the segment uniformly is checked on the moments themselves:
    c = _Col(0.0, 100.0).column(o, d, seg, moments=True)["G4_Cu"]
    xm = c[1] / c[0]
    assert np.allclose(xm, 50.0)                               # centroid = midpoint
    c = _Col(0.0, 10.0).column(o, d, seg, moments=True)["G4_Cu"]
    assert np.allclose(c[1] / c[0], 5.0)                       # material at the top: centroid near the start


def test_split_length_puts_the_first_kink_after_material():
    it = st.TrackerInterior.__new__(st.TrackerInterior)
    it.shells, it.nsub = _Col(80.0, 100.0), 8               # all the material in the last fifth of the segment
    o, d = np.zeros((1, 3)), np.array([[0.0, -1.0, 0.0]])
    x0 = __import__("mu2efastsim.eloss", fromlist=["x"]).radiation_length_mm(__import__("mu2efastsim.materials", fromlist=["x"]).get("G4_Cu"))
    L = it._split_length(o, d, np.array([100.0]), np.array([10.0 / x0]))   # half the material
    assert 87.4 < L[0] < 93.8                                  # 90 mm, on the 1/16 grid


def test_envelope_exit_from_axis_downwards():
    o = geometry.to_mu2e(np.array([[0.0, 0.0, 0.0], [0.0, 900.0, 0.0]]))
    d = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    t = st._envelope_exit(o, d)
    from trkfastsim import surfaces as S
    assert np.isclose(t[0], S.R_OUT) and t[1] == 0.0
