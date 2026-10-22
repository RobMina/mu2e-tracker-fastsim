import numpy as np
import pytest

jax = pytest.importorskip("jax")

from trkfastsim.reco import hitreco, hitreco_jax  # noqa: E402


def _cond(nstraw, rng):
    st = {"midx": rng.uniform(-700, 700, nstraw), "midy": rng.uniform(-700, 700, nstraw), "midz": rng.uniform(-1500, 1500, nstraw),
          "hlen": rng.uniform(200, 600, nstraw), "toff_hv": rng.normal(0, 1, nstraw), "toff_cal": rng.normal(0, 1, nstraw),
          "dead": (rng.uniform(size=nstraw) < 0.02).astype(float), "noisy": (rng.uniform(size=nstraw) < 0.02).astype(float),
          "halfvp": rng.uniform(80, 120, nstraw), "pmpscale": np.ones(nstraw)}
    phi = rng.uniform(0, 2 * np.pi, nstraw)
    st.update(dirx=np.cos(phi), diry=np.sin(phi), dirz=np.zeros(nstraw))
    return {"scalar": {"tdcLSB": 0.0390625, "adcLSB": 0.4883, "totLSB": 4.0, "strawGain": 7000.0, "eionPerQ": 0.0277,
                       "pmpScaleAvg": 1.3, "strawOuterRadius": 2.5, "nADCPreSamples": 3},
            "halfvpscale": 1.0 + 0.002 * np.arange(59), "centres": 40.0 + 0.5 * np.arange(59), "resslope": np.full(59, 0.1),
            "totdtime": rng.uniform(0, 20, 80), "totderror": rng.uniform(1, 5, 80), "tot_bins": [16, 5.0, 5, 0.0005],
            "straw": st}


def _digis(n, nstraw, rng):
    ev = np.sort(rng.integers(0, max(n // 20, 1), n))
    plane, panel, straw = rng.integers(0, 36, n), rng.integers(0, 6, n), rng.integers(0, 96, n)
    sid = ((plane << 10) | (panel << 7) | straw).astype(np.uint16)
    us = hitreco.unique_straw(sid.astype(np.int64))
    assert us.max() < nstraw
    nadc = rng.integers(8, 16, n)
    adc = rng.integers(0, 1024, nadc.sum())
    t0 = rng.integers(1000, 2_000_000, n)
    return {"ev": ev, "sid": sid, "tdc0": t0, "tdc1": t0 + rng.integers(-300, 300, n), "tot0": rng.integers(0, 20, n),
            "tot1": rng.integers(0, 20, n), "nadc": nadc, "adc": adc, "pbt": rng.uniform(0, 50, ev.max() + 1),
            "evkey": np.zeros((ev.max() + 1, 3), int)}


@pytest.mark.parametrize("settings", ["mc", "data"])
def test_straw_hits_jax_matches_numpy_on_cpu(settings):
    rng = np.random.default_rng(7)
    nstraw = 36 * 6 * 96
    cond = _cond(nstraw, rng)
    d = _digis(5000, nstraw, rng)
    ref = hitreco.straw_hits(d, cond, settings)
    out = hitreco_jax.straw_hits(d, cond, settings)
    assert set(ref) == set(out)
    for k in ref:
        a, b = np.asarray(ref[k]), np.asarray(out[k])
        assert a.shape == b.shape, k
        if jax.default_backend() == "cpu":
            assert np.array_equal(a, b), (k, np.flatnonzero(a != b)[:5])
        elif a.dtype.kind == "f":
            assert np.all(np.abs(a.astype(np.float64) - b) <= 2 * np.spacing(np.abs(a).astype(np.float32))), k
        else:
            assert np.mean(a == b) >= 0.9999, k


def test_padding_does_not_change_results():
    rng = np.random.default_rng(3)
    cond = _cond(36 * 6 * 96, rng)
    d = _digis(700, 36 * 6 * 96, rng)
    dc = hitreco_jax.device_conditions(cond)
    a = hitreco_jax.straw_hits(d, cond, dcond=dc, packed=hitreco_jax.pack_digis(d, dc["npre"], n_pad=1024, ns_pad=16))
    b = hitreco_jax.straw_hits(d, cond, dcond=dc, packed=hitreco_jax.pack_digis(d, dc["npre"], n_pad=4096, ns_pad=32))
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def _combo_case(seed, settings):
    from trkfastsim.reco import combo
    rng = np.random.default_rng(seed)
    nstraw = 36 * 6 * 96
    cond = _cond(nstraw, rng)
    d = _digis(6000, nstraw, rng)
    n = len(d["sid"])
    d["ev"] = np.sort(rng.integers(0, 150, n))                               # ~40 hits per event
    d["pbt"] = rng.uniform(0, 50, 150)
    plane, panel = rng.integers(0, 2, n), rng.integers(0, 2, n)               # ~10 hits per panel run
    d["sid"] = ((plane << 10) | (panel << 7) | rng.integers(0, 12, n)).astype(np.uint16)
    d["tdc0"] = rng.integers(1000, 2500, n)                                    # within ~60 ns
    d["tdc1"] = d["tdc0"] + rng.integers(-50, 50, n)
    sh = hitreco.straw_hits(d, cond, settings)
    sh["flag"] = sh["flag"] | np.uint32(combo.SEL)          # force the selection bits so most hits combine
    return combo, sh


@pytest.mark.parametrize("settings", ["mc", "data"])
def test_panel_hits_jax_matches_numpy(settings):
    from trkfastsim.reco import combo_jax
    combo, sh = _combo_case(11, settings)
    ref = combo.panel_hits(sh, settings)
    out = combo_jax.panel_hits(sh, settings)
    assert (ref["ncombo"] > 1).sum() > 100
    for k in ("ev", "sid", "ncombo", "flag", "ind", "ind_file", "ind_combo"):
        assert np.array_equal(ref[k], out[k]), k
    for k in ("edep", "et0", "et1", "time", "wdist", "x", "y", "z", "uvar"):
        a, b = ref[k], out[k]
        if jax.default_backend() == "cpu":
            assert np.array_equal(a, b), (k, np.flatnonzero(a != b)[:5])
        else:
            assert np.all(np.abs(a.astype(np.float64) - b) <= 2 * np.spacing(np.abs(a))), k


def test_panel_hits_jax_chunking_is_invisible():
    from trkfastsim.reco import combo_jax
    combo, sh = _combo_case(5, "data")
    a = combo_jax.panel_hits(sh, "data", chunk=64)
    b = combo_jax.panel_hits(sh, "data", chunk=1 << 20)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def _toy_tracks(ntrk, nhit, rng):
    """Straight muon tracks crossing toy wires (planes along y, wires along x or z), hits generated from the F2 model
    itself with small noise, so the fit has a known answer."""
    from trkfastsim.reco import linefit
    xt = np.stack([rng.uniform(-200, 200, ntrk), rng.uniform(-200, 200, ntrk), rng.uniform(-0.5, 0.5, ntrk),
                   rng.uniform(-0.5, 0.5, ntrk), rng.uniform(1000, 2000, ntrk)], 1)
    ys = np.linspace(400, -400, nhit)
    mid = np.zeros((ntrk, nhit, 3)); wdir = np.zeros((ntrk, nhit, 3))
    along = np.arange(nhit) % 2 == 0
    pos = xt[:, None, [0, 1]] + xt[:, None, [2, 3]] * (-ys[None, :, None])     # (x, z) at plane y
    mid[..., 0] = np.where(along, 0.0, pos[..., 0] + rng.uniform(-2, 2, (ntrk, nhit)))
    mid[..., 1] = ys
    mid[..., 2] = np.where(along, pos[..., 1] + rng.uniform(-2, 2, (ntrk, nhit)), 0.0)
    wdir[..., 0] = np.where(along, 1.0, 0.0); wdir[..., 2] = np.where(along, 0.0, 1.0)
    H = {"active": np.ones((ntrk, nhit), bool), "lr": np.zeros((ntrk, nhit)), "tmeas": np.zeros((ntrk, nhit)),
         "hlen": np.full((ntrk, nhit), 600.0), "esign": np.where(rng.uniform(size=(ntrk, nhit)) < 0.5, 1.0, -1.0),
         "vsig": np.full((ntrk, nhit), 200.0), "mid": mid, "wdir": wdir, "totdt": np.zeros((ntrk, nhit)),
         "tvar": np.full((ntrk, nhit), 25.0), "uvar": np.full((ntrk, nhit), 900.0), "wdist": np.zeros((ntrk, nhit)),
         "udir_dot_w": np.ones((ntrk, nhit)), "sh": np.zeros((ntrk, nhit), int)}
    dt = np.linspace(-20, 80, 2001)
    drift = {"dt": dt, "rdrift": np.clip(0.065 * dt, 0, 2.5), "dvel": np.full_like(dt, 0.065),
             "sderr": np.full_like(dt, 0.15), "uderr": np.full_like(dt, 0.2)}
    # make the measurements consistent with xt: drift side from the true DOCA, drift time from the true distance
    r, doca, dT, _ = linefit._model(xt, H, drift)
    H["lr"] = np.sign(doca)
    H["tmeas"] = np.abs(doca) / 0.065 - dT + rng.normal(0, 1.0, doca.shape)  # dT_true = |doca|/v (+ noise)
    r, doca, dT, _ = linefit._model(xt, H, drift)
    H["totdt"] = dT + rng.normal(0, 5.0, dT.shape)
    H["wdist"] = r[..., 2] * 0 + (np.where(H["esign"] < 0, -1, 1) * 0)                  # time division neutral
    x0 = xt + np.array([5.0, 5.0, 0.01, 0.01, 2.0]) * rng.normal(size=xt.shape)
    return xt, x0, H, drift


def test_linefit_jax_matches_numpy():
    from trkfastsim.reco import linefit, linefit_jax
    rng = np.random.default_rng(4)
    xt, x0, H, drift = _toy_tracks(64, 20, rng)
    ref = linefit.fit(x0, H, drift, niter=8, use_long=False)
    out = linefit_jax.fit(x0, H, drift, niter=8, use_long=False)
    assert np.all(np.isfinite(out["x"]))
    # numerical (numpy) vs exact (JAX) Jacobian: same minimum within the parameter errors' 1e-3
    err = np.sqrt(np.diagonal(ref["cov"], axis1=1, axis2=2))
    assert np.max(np.abs(out["x"] - ref["x"]) / err) < 1e-3
    assert np.allclose(np.sqrt(np.diagonal(out["cov"], axis1=1, axis2=2)), err, rtol=1e-4)
    assert np.allclose(out["chi2"], ref["chi2"], rtol=1e-4, atol=1e-6)
    # padding to a longer hit length changes nothing
    out2 = linefit_jax.fit(x0, H, drift, niter=8, use_long=False, n_pad=64)
    assert np.allclose(out2["x"], out["x"], rtol=0, atol=1e-9)


def test_line_seeds_jax_matches_numpy():
    """Toy clusters: hits on random lines through random straws (no Mu2e input); seeds from both ports must agree."""
    from trkfastsim.reco import linefind, linefind_jax
    rng = np.random.default_rng(9)
    ncl = 300
    nh = rng.integers(10, 40, ncl)
    nh[:3] = [300, 12, 70]                                  # one oversize cluster takes the numpy path
    n = int(nh.sum())
    ev = np.repeat(np.arange(ncl), nh)
    nstraw = 36 * 6 * 96
    st = {"midx": np.zeros(nstraw), "midy": np.zeros(nstraw), "midz": np.zeros(nstraw), "dirx": np.ones(nstraw),
          "diry": np.zeros(nstraw), "dirz": np.zeros(nstraw), "hlen": np.full(nstraw, 600.0)}
    us = rng.choice(nstraw, n, replace=False)
    a0, b0 = rng.uniform(-300, 300, ncl), rng.uniform(-800, 800, ncl)
    a1, b1 = rng.uniform(-0.6, 0.6, ncl), rng.uniform(-0.6, 0.6, ncl)
    y = rng.uniform(-500, 500, n)
    x = a0[ev] - a1[ev] * y + rng.normal(0, 3, n)
    z = b0[ev] - b1[ev] * y + rng.normal(0, 2, n)
    phi = rng.uniform(0, np.pi, n)
    st["dirx"][us], st["dirz"][us] = np.cos(phi), np.sin(phi)
    st["midx"][us] = x - 50 * np.cos(phi); st["midy"][us] = y; st["midz"][us] = z - 50 * np.sin(phi)
    noise = rng.uniform(size=n) < 0.15                        # off-track hits
    st["midz"][us[noise]] += rng.uniform(50, 300, noise.sum())
    plane, rem = us // 576, us % 576
    sid = ((plane << 10) | ((rem // 96) << 7) | (rem % 96)).astype(np.uint16)
    f32 = np.float32
    sh = {"ev": ev, "sid": sid, "x": x.astype(f32), "y": y.astype(f32), "z": z.astype(f32),
          "wdist": np.full(n, 50.0, f32), "uvar": np.full(n, 900.0, f32), "eend": np.zeros(n, np.int64),
          "et0": rng.uniform(1000, 1100, n).astype(f32), "et1": np.zeros(n, f32), "dtime": np.full(n, 10.0, f32),
          "ptime": np.full(n, 2.0, f32)}
    tc = {"ev": np.arange(ncl), "n": nh, "member": np.arange(n), "member_cl": ev}
    cond = {"straw": st}
    ref = linefind.line_seeds(sh, tc, cond)
    out = linefind_jax.line_seeds(sh, tc, cond)
    assert len(ref["n"]) > 200
    for k in ("tc", "ev", "n", "member", "ind"):
        assert np.array_equal(ref[k], out[k]), k
    for k in ("t0", "a0", "b0", "a1", "b1"):
        assert np.allclose(ref[k], out[k], rtol=1e-9, atol=1e-9), k
