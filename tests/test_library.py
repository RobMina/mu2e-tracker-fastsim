"""Unit tests: the in-straw secondary sampler on a synthetic library."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trkfastsim import library as L                                    # noqa: E402


def _lib():
    # 4 crossings in the same (|p|, path) bin: 0 and 2 have secondaries (1 and 2 steps)
    ins = pd.DataFrame({"xing": [0, 1, 2, 3], "p": 3000.0, "mu_len": 4.8, "nsec": [1, 0, 2, 0]})
    ins["pbin"] = np.digitize(ins.p, L.P_BINS) - 1
    ins["lbin"] = np.digitize(ins.mu_len, L.LEN_BINS) - 1
    st = pd.DataFrame({"xing": [0, 2, 2], "eion": [1e-3, 2e-3, 3e-3], "len": [1.0, 2.0, 3.0], "ion": 1, "shape": 0,
                       "pdg": 11})
    return {"instraw": ins, "instraw_steps": st}


def test_draw_frequency_and_expansion():
    s = L.InStrawSampler(_lib(), np.random.default_rng(3))
    n = 40_000
    xing, st = s.draw(np.full(n, 3000.0), np.full(n, 4.8))
    assert set(np.unique(xing)) == {0, 1, 2, 3}
    assert abs(len(st) / n - (1 + 2) / 4) < 0.02          # mean secondaries per crossing
    assert np.all(np.diff(st.i.to_numpy()) >= 0) and st.i.max() < n


def test_empty_bin_falls_back_to_a_neighbour():
    s = L.InStrawSampler(_lib(), np.random.default_rng(4))
    xing, _ = s.draw(np.array([3000.0]), np.array([2.5]))   # path bin with no library rows
    assert xing[0] in (0, 1, 2, 3)


def test_offaxis_sampler_draws_whole_events_with_empty_ones():
    ev = pd.DataFrame({"run": 1400, "subrun": 1, "event": [1, 2, 3], "p": 3000.0, "ux": 0.0, "uy": -1.0, "uz": 0.0})
    ev["pbin"] = np.digitize(ev.p, L.P_BINS) - 1
    off = pd.DataFrame({"run": 1400, "subrun": 1, "event": [2, 2, 2], "eion": 1e-3, "len": 2.0, "ion": 1, "shape": 0,
                        "pdg": 11, "sp": [5, 5, 6], "k": 0, "dpl": 0, "dpa": 0, "dst": [1, 2, 3], "dt": 0.1})
    s = L.OffAxisSampler({"events": ev, "offaxis": off}, np.random.default_rng(5))
    evi, st = s.draw(np.full(30_000, 3000.0), np.full(30_000, -0.99))
    assert abs(len(st) / 30_000 - 1.0) < 0.03                        # 3 steps in 1 of 3 events
    assert set(st.groupby("i").size().unique()) == {3}


def test_offaxis_paste_lattice_offsets():
    """A fast muon crossing straws (plane 4, panel 2, straw 50) then (plane 5, panel 2, straw 51); library steps
    anchored on k = 0 at +1 straw, on k = 1 at +1 plane, +7 straws, and one off the lattice (plane + 80: beyond one
    mirror)."""
    from trkfastsim import geometry as G, steps as S
    n = 36 * 6 * 96
    up = np.repeat(np.arange(36 * 6), 96)
    g = G.StrawGeometry(plane=up // 6, panel=up % 6, straw=np.tile(np.arange(96), 216), mid=np.zeros((n, 3)),
                        dir=np.tile([1.0, 0, 0], (n, 1)), halflen=np.full(n, 500.0), radius=2.5085,
                        panel_R=np.tile(np.eye(3), (216, 1, 1)), panel_t=np.zeros((216, 3)),
                        plane_R=np.tile(np.eye(3), (36, 1, 1)), plane_t=np.zeros((36, 3)))
    def row(pl, pa, s_, t):
        return {"run": 1400, "subrun": 1, "event": 1, "nsgs": 2, "sid": int(G.straw_id(pl, pa, s_)), "plane": pl,
                "panel": pa, "straw": s_, "upanel": pl * 6 + pa, "shape": 0, "ion": 0, "sp": 1, "eion": 1e-3,
                "len": 4.0, "width": 0.0, "t": t, "x0": 0.0, "y0": 0.0, "z0": 0.0, "x1": 0.0, "y1": 0.0, "z1": 0.0,
                "px": 0.0, "py": -3000.0, "pz": 0.0}
    steps = pd.DataFrame([row(4, 2, 50, 10.0), row(5, 2, 51, 10.2)])
    parts = pd.DataFrame({"run": [1400], "subrun": [1], "event": [1], "key": [1], "parent": [-1], "pdg": [13],
                          "isprim": [1], "ismcprim": [1]})
    ev = pd.DataFrame({"run": [1400], "subrun": [1], "event": [7], "p": [3000.0], "ux": [0.0], "uy": [-1.0], "uz": [0.0]})
    ev["pbin"] = np.digitize(ev.p, L.P_BINS) - 1
    off = pd.DataFrame({"run": 1400, "subrun": 1, "event": [7, 7, 7], "eion": [2e-3, 3e-3, 4e-3], "len": 2.0, "ion": 1,
                        "shape": 0, "pdg": 11, "sp": [9, 9, 10], "k": [0, 1, 1], "dpl": [0, 1, 80], "dpa": 0,
                        "dst": [1, 7, 0], "dt": [0.5, 0.7, 0.1]})
    ev["plane_min"], ev["plane_max"] = 4, 5
    s = L.OffAxisSampler({"events": ev, "offaxis": off}, np.random.default_rng(1))
    out, pp, st = S.add_offaxis_secondaries(steps, parts, s, g)
    assert (st["drawn"], st["off_lattice"], st["in_muon_straw"], st["placed"]) == (3, 1, 0, 2)
    assert st["draw_level_counts"][0] == 1                        # exact (|p|, angle, plane extent) key
    new = out[out.sp != 1].sort_values("t")
    assert new[["plane", "panel", "straw"]].values.tolist() == [[4, 2, 51], [6, 2, 58]]
    assert np.allclose(new.t, [10.5, 10.9]) and (out.nsgs == 4).all()
    assert sorted(pp[pp.key != 1].key.tolist()) == [100000] and pp[pp.key != 1].parent.tolist() == [1]



def test_mirror_reflects_once():
    from trkfastsim.steps import _mirror
    assert _mirror(np.array([-1, -3, 0, 35, 36, 38, 80, -40]), 36).tolist() == [1, 3, 0, 35, 34, 32, 80, -40]


def test_offaxis_sampler_prefers_matching_plane_extent():
    ev = pd.DataFrame({"run": 1400, "subrun": 1, "event": [1, 2], "p": 3000.0, "ux": 0.0, "uy": -1.0, "uz": 0.0,
                       "plane_min": [0, 30], "plane_max": [5, 35]})
    ev["pbin"] = np.digitize(ev.p, L.P_BINS) - 1
    off = pd.DataFrame({"run": 1400, "subrun": 1, "event": [1, 2], "eion": 1e-3, "len": 2.0, "ion": 1, "shape": 0,
                        "pdg": 11, "sp": [5, 6], "k": 0, "dpl": 0, "dpa": 0, "dst": [1, 2], "dt": 0.1})
    s = L.OffAxisSampler({"events": ev, "offaxis": off}, np.random.default_rng(2))
    _, st = s.draw(np.full(200, 3000.0), np.full(200, -0.99), np.full(200, 31), np.full(200, 34))
    assert (st.dst == 2).all() and (s.last_level == 0).all()
    _, st = s.draw(np.full(200, 3000.0), np.full(200, -0.99), np.full(200, 14), np.full(200, 20))
    assert (s.last_level == 2).all() and set(st.dst) == {1, 2}   # no plane match nearby: any plane extent



def test_instraw_pasting_uses_muon_steps_only():
    """Regression: in-straw secondaries are drawn for muon crossings (sp == 1) only, never for pasted secondaries."""
    from trkfastsim import steps as S
    lib = _lib()
    lib["instraw"]["nsec"] = 1
    lib["instraw_steps"] = pd.DataFrame({"xing": [0, 1, 2, 3], "eion": 1e-3, "len": 1.0, "ion": 1, "shape": 0, "pdg": 11})
    base = {"run": 1400, "subrun": 1, "event": 1, "nsgs": 2, "plane": 0, "panel": 0, "straw": 0, "upanel": 0,
            "shape": 0, "ion": 0, "eion": 1e-3, "len": 4.8, "width": 0.0, "t": 1.0, "x0": 0.0, "y0": 0.0, "z0": 0.0,
            "x1": 0.0, "y1": 0.0, "z1": 0.0, "px": 0.0, "py": -3000.0, "pz": 0.0}
    steps = pd.DataFrame([{**base, "sid": 0, "sp": 1}, {**base, "sid": 5, "sp": 100000}])
    parts = pd.DataFrame({"run": [1400, 1400], "subrun": 1, "event": 1, "key": [1, 100000], "parent": [-1, 1],
                          "pdg": [13, 11], "isprim": [1, 0], "ismcprim": [1, 0]})
    out, pp = S.add_instraw_secondaries(steps, parts, L.InStrawSampler(lib, np.random.default_rng(1)))
    new = out[(out.sp >= 1000) & (out.sp < 100000)]
    assert len(new) == 1 and new.sid.tolist() == [0] and len(out) == 3


def test_offaxis_edge_anchor_keeps_outside_steps_outside():
    """Fast muon over planes 10-12 (ranks 0-2 in plane 10, 3 in 11, 4-5 in 12); library muon crossed one plane, its set
    anchored on k = 0: one step in the same plane (dpl 0), one 2 planes below (dpl -2), one 1 plane above (dpl +1). By
    relative rank all anchor on rank 0 (plane 10), so dpl +1 lands in plane 11, which the fast muon crosses; with
    edge_anchor the dpl +1 step anchors on plane 12 and lands in plane 13, dpl -2 in plane 8, dpl 0 unchanged."""
    from trkfastsim import geometry as G, steps as S
    n = 36 * 6 * 96
    up = np.repeat(np.arange(36 * 6), 96)
    g = G.StrawGeometry(plane=up // 6, panel=up % 6, straw=np.tile(np.arange(96), 216), mid=np.zeros((n, 3)),
                        dir=np.tile([1.0, 0, 0], (n, 1)), halflen=np.full(n, 500.0), radius=2.5085,
                        panel_R=np.tile(np.eye(3), (216, 1, 1)), panel_t=np.zeros((216, 3)),
                        plane_R=np.tile(np.eye(3), (36, 1, 1)), plane_t=np.zeros((36, 3)))

    def row(pl, s_, t):
        return {"run": 1400, "subrun": 1, "event": 1, "nsgs": 6, "sid": int(G.straw_id(pl, 2, s_)), "plane": pl,
                "panel": 2, "straw": s_, "upanel": pl * 6 + 2, "shape": 0, "ion": 0, "sp": 1, "eion": 1e-3,
                "len": 4.0, "width": 0.0, "t": t, "x0": 0.0, "y0": 0.0, "z0": 0.0, "x1": 0.0, "y1": 0.0, "z1": 0.0,
                "px": 0.0, "py": -3000.0, "pz": 0.0}
    steps = pd.DataFrame([row(10, 40, 1.0), row(10, 41, 1.1), row(10, 42, 1.2), row(11, 43, 1.3), row(12, 44, 1.4),
                          row(12, 45, 1.5)])
    parts = pd.DataFrame({"run": [1400], "subrun": [1], "event": [1], "key": [1], "parent": [-1], "pdg": [13],
                          "isprim": [1], "ismcprim": [1]})
    ev = pd.DataFrame({"run": [1400], "subrun": [1], "event": [7], "p": [3000.0], "ux": [0.0], "uy": [-1.0], "uz": [0.0],
                       "plane_min": [10], "plane_max": [12], "nstraw_mu": [6]})
    ev["pbin"] = np.digitize(ev.p, L.P_BINS) - 1
    off = pd.DataFrame({"run": 1400, "subrun": 1, "event": [7, 7, 7], "eion": [2e-3, 3e-3, 4e-3], "len": 2.0, "ion": 1,
                        "shape": 0, "pdg": 11, "sp": [9, 10, 11], "k": 0, "dpl": [0, -2, 1], "dpa": 0,
                        "dst": [10, 0, 0], "dt": [0.1, 0.2, 0.3]})
    planes = {}
    for edge in (False, True):
        s = L.OffAxisSampler({"events": ev, "offaxis": off}, np.random.default_rng(1), edge_anchor=edge)
        out, _, st = S.add_offaxis_secondaries(steps, parts, s, g)
        new = out[out.sp != 1].sort_values("t")
        planes[edge] = new[["plane", "straw"]].values.tolist()
    assert planes[False] == [[10, 50], [8, 40], [11, 40]]
    assert planes[True] == [[10, 50], [8, 40], [13, 44]]
