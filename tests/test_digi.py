"""Unit tests: fast tracker digitisation (trkfastsim.digicond, trkfastsim.digi) on the stock job's conditions excerpt
and a synthetic 36 x 6 x 96 straw geometry; table round trip through the hit-reco reader."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trkfastsim import digi as DG                                       # noqa: E402
from trkfastsim import digicond as DC                                   # noqa: E402
from trkfastsim import geometry as G                                    # noqa: E402

FIX = Path(__file__).resolve().parent / "data" / "digi_job_excerpt.fcl"
HL = 500.0


def _geom():
    """Straws along x; straw s of uniquePanel p at y = 10 s, z = 60 p (Mu2e frame, i.e. detector frame + origin)."""
    n = DC.NUSTRAWS
    up = np.repeat(np.arange(216), 96)
    s = np.tile(np.arange(96), 216)
    mid = np.c_[np.zeros(n), 10.0 * s, 60.0 * up] + G.DETECTOR_ORIGIN_EXTRACTED
    z = np.zeros((216, 3, 3))
    return G.StrawGeometry(plane=up // 6, panel=up % 6, straw=s, mid=mid, dir=np.tile([1.0, 0, 0], (n, 1)),
                           halflen=np.full(n, HL), radius=2.5085, panel_R=z, panel_t=np.zeros((216, 3)),
                           plane_R=np.zeros((36, 3, 3)), plane_t=np.zeros((36, 3)), wall_rmax=2.5235, wire_r=0.0125)


@pytest.fixture(scope="module")
def cond():
    d = DC.parse_fhicl_dump(FIX.read_text())
    pr = np.zeros((DC.NUSTRAWS, 6))
    pr[:, 0] = np.arange(DC.NUSTRAWS)
    pr[:, 3], pr[:, 4], pr[:, 5] = 11.996, 11.632, 1.5e6
    return DC.assemble(d, pr, _geom())


def _step(c, us, b=1.0, zpos=100.0, t=1000.0, eion=2.0e-3, ion=0):
    """A vertical (along z) chord through straw us at distance b from the wire, along-wire position zpos (toward cal)."""
    mid, sdir = c.straw_mid[us], c.straw_dir[us]
    r = c.strawInnerRadius
    h = np.sqrt(r * r - b * b)
    yv = np.array([0.0, 1.0, 0.0])
    p0 = mid + zpos * sdir + b * yv + np.array([0, 0, h])
    p1 = mid + zpos * sdir + b * yv - np.array([0, 0, h])
    sid = DG.unique_straw_to_sid(us)
    return dict(sid=sid, shape=0, ion=ion, eion=eion, len=2 * h, width=0.0, t=t, x0=p0[0], y0=p0[1], z0=p0[2],
                x1=p1[0], y1=p1[1], z1=p1[2])


def _steps(rows, ev=None):
    keys = rows[0].keys()
    st = {k: np.array([r[k] for r in rows]) for k in keys}
    st["ev"] = np.zeros(len(rows), np.int64) if ev is None else np.asarray(ev)
    return st


def test_conditions_match_stock_printout(cond):
    c = cond
    # makeSD printLevel 1 / strawElectronics verbose 1 printout of the stock job (Offline v13_39_00)
    assert np.isclose(c.EAverage, 2.60898e-05, rtol=1e-5) and np.isclose(c.NAverage, 1.97061, rtol=1e-5)
    assert np.isclose(c.intNProb[0], 0.663632, rtol=1e-5) and np.isclose(c.EIonize[1], 4.33893e-05, rtol=1e-5)
    assert np.allclose(c.current_pulse[2][-2:], [2.82994e-07, 2.82938e-07], rtol=1e-5)
    assert np.allclose(c.resp_thresh[2][-2:], [-9.79291e-09, -9.78568e-09], rtol=1e-5)
    assert np.allclose(c.resp_adc[2][-2:], [-4.54564e-06, -4.54248e-06], rtol=1e-5)
    assert np.allclose(c.resp_sat[2][-2:], [-9.79291e-09, -9.78568e-09], rtol=1e-5)
    assert np.allclose(c.resp_imp[-2:], [0.00105868, 0.00105771], rtol=1e-5)
    assert c.adc_ped[0] == 391 and c.digitizationStartTDC == 12800 and c.digitizationEndTDC == 77056
    assert c.eventWindowLength == 100000.0 and c.nADCSamples == 15 and c.ttrunc == 500.0
    assert c.allPlanes == [0, 1] and c.allStraw == 91


def test_drift_table(cond):
    c = cond
    t = DC.d2t(np.array([0.0, 0.5, 1.0, 2.0, 2.5]), np.zeros(5), c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi)
    assert t[0] == 0.0 and np.all(np.diff(t) > 0) and 35.0 < t[-1] < 45.0
    assert np.all(c.drift_times == c.drift_times[:, :1])      # no field: no Lorentz-angle dependence


def test_single_muon_step_gives_one_digi(cond):
    c = cond
    us = 5 * 96 + 40
    z = 120.0
    d = DG.digitize(_steps([_step(c, us, b=1.0, zpos=z)]), 1, c, np.random.default_rng(3))
    assert len(d["sid"]) == 1 and d["sid"][0] == DG.unique_straw_to_sid(us)
    t = d["tdc"][0] * c.TDCLSB
    assert np.isclose(t[1] - t[0], 2 * z / c.vprop, atol=0.6)  # cal end at +halflength along the straw direction
    tcal = 1000.0 + (HL - z) / c.vprop
    assert tcal < t[0] < tcal + 30.0
    assert 1 <= d["tot"][0, 0] <= 15 and d["adc"].shape == (1, 15) and d["pmp"][0] > 20
    assert d["step"][0, 0] == 0 and d["step"][0, 1] == 0


def test_window_and_energy_cut(cond):
    c = cond
    us = 7 * 96 + 10
    rows = [_step(c, us, eion=1.0e-6), _step(c, us + 2, t=100.0)]   # below minstepE; before the digitisation window
    d = DG.digitize(_steps(rows), 1, c, np.random.default_rng(1))
    assert len(d["sid"]) == 0


def test_read_all_straws_keep_early_hits(cond):
    c = cond
    early = 100.0
    d = DG.digitize(_steps([_step(c, 0 * 96 + 92, t=early)]), 1, c, np.random.default_rng(1))   # plane 0, straw >= 91
    assert len(d["sid"]) == 1


def test_digital_dead_time(cond):
    c = cond
    us = 9 * 96 + 50
    near = DG.digitize(_steps([_step(c, us, t=1000.0), _step(c, us, t=1200.0)]), 1, c, np.random.default_rng(2))
    far = DG.digitize(_steps([_step(c, us, t=1000.0), _step(c, us, t=1600.0)]), 1, c, np.random.default_rng(2))
    assert len(near["sid"]) == 1 and len(far["sid"]) == 2
    assert np.all(np.diff(far["tdc"][:, 0]) > 0)


def test_reproducible_and_empty(cond):
    c = cond
    rows = [_step(c, 3 * 96 + k, b=0.3 * k % 2.4, zpos=10.0 * k) for k in range(20)]
    a = DG.digitize(_steps(rows), 1, c, np.random.default_rng(7))
    b = DG.digitize(_steps(rows), 1, c, np.random.default_rng(7))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    e = DG.digitize({k: v[:0] for k, v in _steps(rows).items()}, 1, c, np.random.default_rng(7))
    assert len(e["sid"]) == 0 and e["adc"].shape == (0, 15)


def test_minion_cluster_budget(cond):
    c = cond
    rng = np.random.default_rng(5)
    etot = np.array([2.0e-3, 5.0e-5, 3.0e-2])
    nc = np.array([14, 2, 30])
    ne = DG._minion_clusters(rng, c, etot, nc)
    assert len(ne) == nc.sum() and ne.min() >= 1
    off = np.r_[0, np.cumsum(nc)]
    tot = np.array([ne[off[i]:off[i + 1]].sum() for i in range(3)])
    assert np.all(tot * c.EIonize[0] <= etot)                  # every electron costs at least the first ionisation
    assert 50 < tot[0] < 140                                   # about eIon / 26 eV for a 2 keV step


def test_table_round_trip(cond, tmp_path):
    from trkfastsim.analyses import fastdigi
    from trkfastsim.reco import hitreco
    c = cond
    rows = [_step(c, 3 * 96 + k, b=0.2 * k, zpos=5.0 * k) for k in range(8)]
    d = DG.digitize(_steps(rows, ev=[0, 0, 0, 2, 2, 2, 2, 2]), 3, c, np.random.default_rng(9))
    keys = np.array([[1400, 1001, 5], [1400, 1001, 6], [1400, 1001, 9]])
    p = tmp_path / "fd.root"
    fastdigi.write_digis(str(p), keys, d, c.nADCSamples)
    r = hitreco.read_digis([str(p)])
    assert np.array_equal(r["sid"], d["sid"]) and np.array_equal(r["tdc0"], d["tdc"][:, 0])
    assert np.array_equal(r["tot1"], d["tot"][:, 1]) and np.all(r["nadc"] == 15)
    assert np.array_equal(r["adc"], d["adc"].ravel()) and np.array_equal(r["evkey"], keys)
    assert np.array_equal(r["ev"], d["ev"])


def test_lattice_waveform_equals_pointwise(cond):
    c = cond
    rng = np.random.default_rng(4)
    rows = [_step(c, 11 * 96 + k, b=0.25 * k, zpos=-40.0 * k, t=2000.0 + 3 * k) for k in range(6)]
    st = _steps(rows)
    st["sid"] = np.asarray(st["sid"], np.int64)
    st["_us"] = DG.unique_straw(st["sid"])
    cs, pos, ne = DG.make_clusters(rng, c, st, np.arange(len(rows)))
    us = st["_us"][cs]
    rho, zw, _ = DG.wire_coordinates(c, us, pos)
    t = 2000.0 + rho * 15.0
    q = np.full(len(cs), 0.01)
    wd = c.wire_halflen[us] - zw
    inst = np.unique(us, return_inverse=True)[1]
    wf = DG.Waveforms(c, 2 * inst, t, q, wd, us, cs)
    r = np.unique(2 * inst)
    t0 = wf.t[wf.s[r]] - 3.0
    for path, step, K in (("thresh", 50, 14), ("adc", 200, 15), ("sat", 25, 60)):
        lat = wf.volt_lattice(r, t0, step, K, path)
        pt = wf.volt(r, t0[:, None] + np.arange(K)[None, :] * step / c.sampleRate, path)
        assert np.allclose(lat, pt, rtol=1e-5, atol=1e-6 * np.abs(pt).max())


def test_sim_wire_offsets_in_panel_uvw():
    g = _geom()
    a = np.zeros((DC.NUSTRAWS, 9))
    a[:, 0] = np.arange(DC.NUSTRAWS)
    a[7, 1:5] = [0.2, -0.1, 0.4, 0.3]                          # wire cal dV, dW, hv dV, dW of straw 7
    s = DC.sim_straws(g, a)
    cal = s["wire_mid"][7] + s["wire_halflen"][7] * s["wire_dir"][7]
    hv = s["wire_mid"][7] - s["wire_halflen"][7] * s["wire_dir"][7]
    nom = s["straw_mid"][7]
    # u = +x (straw direction, HV -> cal), v = +y (towards straw 94), w = u x v = +z
    assert np.allclose(cal - (nom + [HL, 0, 0]), [0, 0.2, -0.1], atol=1e-9)
    assert np.allclose(hv - (nom - [HL, 0, 0]), [0, 0.4, 0.3], atol=1e-9)
    assert np.allclose(s["wire_mid"][8], s["straw_mid"][8]) and np.isclose(s["wire_halflen"][8], HL)


def test_event_clustered_and_paired_variance():
    from trkfastsim.analyses import digicmp as DC2
    bins = np.arange(4) - 0.5
    # arm 1: event 0 puts 3 entries in bin 1; event 1 one in bin 0. arm 2: event 0 two in bin 1, event 1 one in bin 2
    h1 = DC2.ev_bin_counts(np.array([1, 1, 1, 0.0]), np.array([0, 0, 0, 1]), bins)
    h2 = DC2.ev_bin_counts(np.array([1, 1, 2.0]), np.array([0, 0, 1]), bins)
    assert np.array_equal(h1[0], [1, 3, 0]) and np.array_equal(h1[1], [1, 9, 0])
    assert np.array_equal(h2[0], [0, 2, 1]) and np.array_equal(h2[1], [0, 4, 1])
    assert np.array_equal(DC2.paired_var(h1, h2, 3), [1, 1, 1])
    assert np.allclose(np.histogram([1, 1, 1, 0.0], bins)[0], h1[0])
    c, n = DC2.chi2ndf(h1[0], h2[0], h1[1] + h2[1])
    assert n == 3 and np.isclose(c, (1 / 1 + 1 / 13 + 1 / 1) / 3)
