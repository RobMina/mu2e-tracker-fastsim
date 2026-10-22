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
