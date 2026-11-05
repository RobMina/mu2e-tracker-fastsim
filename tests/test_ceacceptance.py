"""Vacuum first-hit and fixed hybrid sensitivities are diagnostics, not physics candidates."""
import numpy as np

from trkfastsim.validation import ceacceptance as ca
from test_fieldsteps import _geometry, _start


def test_first_hit_full_search_and_chord_index_agree_without_field():
    o = np.array([[0, 0, -40.0], [1000.0, 0, -40.0], [0, 300.0, -40.0]])
    u = np.tile([0.0, 0, 1], (3, 1))
    start = _start(o, u, 105.0, pdg=11)
    out, check = ca.first_gas(start, _geometry(), np.zeros(3), np.zeros(3))
    assert out.vacuum_hit.tolist() == [True, False, False]
    assert out.status.tolist() == [3, 0, 0]
    assert check["chord_index_full_search_hit_disagreements"] == 0
    assert (out.path <= 30000).all()


def test_fixed_hybrid_menu_preserves_id_and_changes_only_named_components():
    f = _start(np.array([[0.0, 0, -40]]), np.array([[0.0, 0, 1]]), 80.0, pdg=11)
    f["event"] = np.array([2**54 + 1], np.uint64)
    g = f.copy()
    g[ca.XYZ] = [[1.0, 2, 3]]
    g[ca.MOM] = [[100.0, 0, 0]]
    h = ca.hybrids(f, g)
    np.testing.assert_allclose(h["g4_xu_fast_p"][ca.MOM], [[80, 0, 0]])
    np.testing.assert_allclose(h["fast_xu_g4_p"][ca.MOM], [[0, 0, 100]])
    np.testing.assert_allclose(h["fast_x_g4_u_fast_p"][ca.XYZ], f[ca.XYZ])
    np.testing.assert_allclose(h["g4_x_fast_u_fast_p"][ca.MOM], f[ca.MOM])
    assert all(d.event.iloc[0] == 2**54 + 1 for d in h.values())


def test_empty_vacuum_start():
    s = _start(np.empty((0, 3)), np.empty((0, 3)), 105.0, pdg=11)
    out, check = ca.first_gas(s, _geometry(), np.array([0.0, 0, 1]), np.zeros(3))
    assert out.empty and check["tested_chords"] == 0
