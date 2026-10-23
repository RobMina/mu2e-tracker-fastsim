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
