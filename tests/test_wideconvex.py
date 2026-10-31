"""Local-plane numerical arm keeps owners, finite intervals and explicit shape padding."""
import numpy as np

from mu2efastsim import world
from trkfastsim import wideconvex as W


def test_original_local_planes_with_stored_translation_and_mixed_plane_counts():
    local = np.array([[1., 0., 0., 1.], [-1., 0., 0., 1.], [0., 1., 0., 1.], [0., -1., 0., 1.], [0., 0., 1., 1.], [0., 0., -1., 1.]])
    center = np.array([-3904., 0., 24171.])
    global_planes = local.copy()
    global_planes[:, 3] += local[:, :3] @ center
    ps = world.PieceSet([("0", global_planes, center - 1., center + 1.),
                         ("1", np.vstack([global_planes, global_planes[-1:]]), center - 1., center + 1.)])
    terms = {0: (local, np.eye(3), center), 1: (np.vstack([local, local[-1:]]), np.eye(3), center)}
    rows = list(W.local_hits(ps, np.array([center + [-2., 0., 0.]]), np.array([[1., 0., 0.]]), np.array([4.]), terms))
    assert sum(len(r[0]) for r in rows) == 2
    for _, _, a, b in rows:
        np.testing.assert_allclose(a, 1.)
        np.testing.assert_allclose(b, 3.)
