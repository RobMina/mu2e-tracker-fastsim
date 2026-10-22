"""Unit tests: library format v2 (straw-local placement) built on a synthetic straw and re-pasted around fast muons."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trkfastsim import geometry as G, library as L, steps as S        # noqa: E402

R = 2.5085


def _geom():
    """Full lattice of straws along x (halflen 500), straw j at y = 10 * straw, z = 50 * (plane * 6 + panel)."""
    n = 36 * 6 * 96
    up = np.repeat(np.arange(216), 96)
    st = np.tile(np.arange(96), 216)
    mid = np.c_[np.zeros(n), 10.0 * st, 50.0 * up] + G.DETECTOR_ORIGIN_EXTRACTED
    return G.StrawGeometry(plane=up // 6, panel=up % 6, straw=st, mid=mid, dir=np.tile([1.0, 0, 0], (n, 1)),
                           halflen=np.full(n, 500.0), radius=R, panel_R=np.tile(np.eye(3), (216, 1, 1)),
                           panel_t=np.zeros((216, 3)), plane_R=np.tile(np.eye(3), (36, 1, 1)), plane_t=np.zeros((36, 3)))


def _row(g, j, p0, p1, pvec, sp, t, eion=1e-3):
    """sgsteps row in straw j (detector frame), ends given in the Mu2e frame."""
    a, b = G.to_detector(p0), G.to_detector(p1)
    return {"run": 1, "subrun": 1, "event": 1, "nsgs": 0, "sid": int(g.sid[j]), "plane": int(g.plane[j]),
            "panel": int(g.panel[j]), "straw": int(g.straw[j]), "upanel": int(g.upanel[j]), "shape": 0, "ion": 0,
            "sp": sp, "eion": eion, "len": float(np.linalg.norm(b - a)), "width": 0.0, "t": t,
            "x0": a[0], "y0": a[1], "z0": a[2], "x1": b[0], "y1": b[1], "z1": b[2],
            "px": pvec[0], "py": pvec[1], "pz": pvec[2]}


def _chord(g, j, d, b, u0):
    """Gas chord through straw j of the line with direction d passing the wire at offset b (along z) and along-wire u0."""
    m = g.mid[j]
    c = m + np.array([u0, 0.0, b])
    t = np.sqrt(R ** 2 - b ** 2) / np.hypot(d[1], d[2])     # transverse part of d is in y
    return c - t * d, c + t * d


def test_local_roundtrip_any_direction():
    rng = np.random.default_rng(1)
    n = 500
    a = np.tile([1.0, 0, 0], (n, 1))
    m = rng.normal(size=(n, 3))
    d = L._unit(rng.normal(size=(n, 3)))
    x = m + rng.uniform(-400, 400, n)[:, None] * a + np.c_[np.zeros(n), rng.uniform(-1.7, 1.7, (n, 2))]
    s, e1, e2 = L.straw_frame(a, d)
    uref = rng.uniform(-100, 100, n)
    du, r, ph = L.to_local(x, m, a, s, e1, e2, uref)
    assert np.allclose(L.from_local(du, r, ph, m, a, s, e1, e2, uref, np.full(n, 1e9)), x)


def test_line_u_closest_approach_and_clip():
    a = np.array([[1.0, 0, 0]] * 2)
    m = np.zeros((2, 3))
    p = np.array([[30.0, 5.0, -20.0], [900.0, 5.0, -20.0]])
    d = np.array([[0.0, 0.0, 1.0]] * 2)
    assert np.allclose(L.line_u(m, a, np.full(2, 500.0), p, d), [30.0, 500.0])


def test_v2_build_and_repaste_at_another_angle():
    """A G4 muon crosses straw A going -y (wire offset +0.5); a delta ray starts on its chord and ends 0.3 mm from the
    wire. A second step sits in straw B (+3 straws, off-axis). Rebuilt around a fast muon at another angle and wire
    offset, both steps keep (along-wire offset, r, phi) in the fast muon's frame."""
    g = _geom()
    jA, jB = 10 * 96 + 40, 10 * 96 + 43
    dg = L._unit(np.array([[0.3, -1.0, 0.0]]))[0]
    q0, q1 = _chord(g, jA, dg, 0.5, 100.0)
    xc = L.chord_u(g.mid[[jA]], g.dir[[jA]], q0[None], q1[None])[0]
    e0 = q0 + 0.4 * (q1 - q0)
    e1 = g.mid[jA] + np.array([xc + 2.0, 0.3 * np.cos(1.0), 0.3 * np.sin(1.0)])
    # off-axis step in straw B (3 straws away)
    fb0 = g.mid[jB] + np.array([140.0, 1.0, 0.0])
    fb1 = g.mid[jB] + np.array([145.0, 0.0, -1.5])
    steps = pd.DataFrame([_row(g, jA, q0, q1, 3000 * dg, 1, 10.0), _row(g, jA, e0, e1, [0, 0, 0], 2, 10.01),
                          _row(g, jB, fb0, fb1, [0, 0, 0], 3, 10.1)])
    steps["nsgs"] = 3
    parts = pd.DataFrame({"run": 1, "subrun": 1, "event": 1, "key": [1, 2, 3], "pdg": [13, 11, 11],
                          "ismcprim": [1, 0, 0]})
    lib = _build(steps, parts, g)
    assert L.version(lib) == 2
    ins, off = lib["instraw_steps"].iloc[0], lib["offaxis"].iloc[0]
    assert np.isclose(ins.r1, 0.3) and np.isclose(ins.du1, 2.0) and np.isclose(ins["dt"], 0.01)
    assert np.isclose(ins.du0, L.to_local(e0[None], g.mid[[jA]], g.dir[[jA]], *L.straw_frame(g.dir[[jA]], dg[None]), xc)[0][0])
    # the off-axis reference: where the G4 muon line comes closest to B's axis
    uB = L.line_u(g.mid[[jB]], g.dir[[jB]], np.array([500.0]), q0[None], dg[None])[0]
    assert np.isclose(off.du0, 140.0 - uB) and np.isclose(off.r1, np.hypot(0.0, 1.5))
    # fast muon in the same straws, steeper, the other way along the wire and on the other side of the wire
    df = L._unit(np.array([[-0.8, -1.0, 0.0]]))[0]
    f0, f1 = _chord(g, jA, df, 0.7, -50.0)
    fast = pd.DataFrame([_row(g, jA, f0, f1, 2000 * df, 1, 20.0)])
    fast["nsgs"] = 1
    fparts = parts.iloc[[0]].assign(parent=-1, isprim=1)[["run", "subrun", "event", "key", "parent", "pdg", "isprim", "ismcprim"]]
    smp = L.InStrawSampler(lib, np.random.default_rng(1))
    out, _ = S.add_instraw_secondaries(fast, fparts, smp, g)
    new = out[out.sp >= S.SECONDARY_KEY0]
    assert len(new) == 1
    x0, x1 = (G.to_mu2e(new[[f"x{e}", f"y{e}", f"z{e}"]].to_numpy(float)) for e in "01")
    xf = L.chord_u(g.mid[[jA]], g.dir[[jA]], f0[None], f1[None])
    s, ea, eb = L.straw_frame(g.dir[[jA]], df[None])
    flip = -1.0                                            # in its own frame the fast muon passes the wire on the other side
    bf = L._dot(f0[None] - g.mid[[jA]], eb)
    for x, e in ((x0, "0"), (x1, "1")):
        du, cn, h = L.to_chord(x, g.mid[[jA]], g.dir[[jA]], s, ea, eb, xf, bf, R)
        assert np.allclose([du[0], cn[0], h[0]], [ins[f"du{e}"], ins[f"cn{e}"], flip * ins[f"h{e}"]])
    assert s[0] == -1.0 and np.isclose(x1[0, 0] - g.mid[jA][0], xf[0] - 2.0)   # along-wire offset follows the muon
    # the delta ray's start was on the G4 chord: it lands on the fast chord (transverse plane); its end keeps r = 0.3 when |b| agrees
    tr = lambda v: v - v[0] * np.array([1.0, 0, 0])            # transverse part (straw axis = x)
    assert np.linalg.norm(np.cross(tr(x0[0] - f0), tr(df))) < 1e-9 * np.linalg.norm(tr(df)) + 1e-9
    g0, g1 = _chord(g, jA, df, -0.5, -50.0)
    out, _ = S.add_instraw_secondaries(pd.DataFrame([{**_row(g, jA, g0, g1, 2000 * df, 1, 20.0), "nsgs": 1}]), fparts, smp, g)
    y1 = G.to_mu2e(out[out.sp >= S.SECONDARY_KEY0][["x1", "y1", "z1"]].to_numpy(float))
    assert np.isclose(np.linalg.norm(np.cross(y1[0] - g.mid[jA], [1.0, 0, 0])), 0.3)
    # off-axis repaste: fast muon in straw A (k = 0), target B at dst = +3
    ev = lib["events"]
    osmp = L.OffAxisSampler(lib, np.random.default_rng(2))
    out, _, stt = S.add_offaxis_secondaries(fast, fparts, osmp, g)
    assert stt["placed"] == 1
    nb = out[out.sp >= S.OFFAXIS_KEY0]
    assert nb.sid.tolist() == [int(g.sid[jB])]
    y0 = G.to_mu2e(nb[["x0", "y0", "z0"]].to_numpy(float))
    sB, eB1, eB2 = L.straw_frame(g.dir[[jB]], df[None])
    uBf = L.line_u(g.mid[[jB]], g.dir[[jB]], np.array([500.0]), f0[None], df[None])
    du, r, ph = L.to_local(y0, g.mid[[jB]], g.dir[[jB]], sB, eB1, eB2, uBf)
    fl = -1.0 if np.sign(L._dot(f0[None] - g.mid[[jB]], eB2))[0] * np.sign(off.mub) < 0 else 1.0
    assert np.allclose([du[0], r[0]], [off.du0, off.r0]) and np.isclose(np.cos(ph[0] - fl * off.phi0), 1.0)
    assert len(ev) == 1


def _build(steps, parts, g, tmp=None):
    """library.build on in-memory tables (T.read_many patched)."""
    from trkfastsim import tables as T
    orig = T.read_many
    T.read_many = lambda paths, tree, cols=None: (steps if tree == "sgsteps" else parts)[cols or slice(None)].copy()
    try:
        return L.build(["mem"], g)
    finally:
        T.read_many = orig


def test_v1_library_still_pastes_on_chord_and_axis():
    """A v1 library (no du0 columns) keeps the v1 placement: in-straw on the muon chord, off-axis on the straw axis."""
    g = _geom()
    j = 5 * 96 + 20
    d = np.array([0.0, -1.0, 0.0])
    q0, q1 = _chord(g, j, d, 0.2, 10.0)
    fast = pd.DataFrame([_row(g, j, q0, q1, 3000 * d, 1, 5.0)])
    fast["nsgs"] = 1
    parts = pd.DataFrame({"run": [1], "subrun": 1, "event": 1, "key": [1], "parent": [-1], "pdg": [13], "isprim": [1],
                          "ismcprim": [1]})
    ins = pd.DataFrame({"xing": [0], "p": 3000.0, "mu_len": 4.8, "nsec": [1]})
    ins["pbin"], ins["lbin"] = np.digitize(ins.p, L.P_BINS) - 1, np.digitize(ins.mu_len, L.LEN_BINS) - 1
    lib = {"instraw": ins, "instraw_steps": pd.DataFrame({"xing": [0], "eion": 1e-3, "len": 1.0, "ion": 1, "shape": 0,
                                                          "pdg": 11}),
           "offaxis": pd.DataFrame({"run": 9, "subrun": 1, "event": [1], "eion": 1e-3, "len": 2.0, "ion": 1, "shape": 0,
                                    "pdg": 11, "sp": [5], "k": 0, "dpl": 0, "dpa": 0, "dst": [2], "dt": 0.1}),
           "events": pd.DataFrame({"run": [9], "subrun": 1, "event": 1, "p": 3000.0, "ux": 0.0, "uy": -1.0, "uz": 0.0,
                                   "pbin": np.digitize([3000.0], L.P_BINS) - 1})}
    assert L.version(lib) == 1
    out, _ = S.add_instraw_secondaries(fast, parts, L.InStrawSampler(lib, np.random.default_rng(1)), g)
    new = out[out.sp >= S.SECONDARY_KEY0]
    assert np.allclose(new[["x0", "y0", "z0", "x1", "y1", "z1"]].to_numpy(), fast[["x0", "y0", "z0", "x1", "y1", "z1"]].to_numpy())
    out, _, _ = S.add_offaxis_secondaries(fast, parts, L.OffAxisSampler(lib, np.random.default_rng(1)), g)
    nb = out[out.sp >= S.OFFAXIS_KEY0]
    x0 = G.to_mu2e(nb[["x0", "y0", "z0"]].to_numpy(float))[0]
    assert np.allclose(x0, g.mid[j + 2] - np.array([1.0, 0, 0]))
