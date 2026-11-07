"""recoresp: reco response tables (NumPy) and their device application (synthetic reference sample, no data files)."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.recoresp import table as RT


def test_entry_truth_of_a_known_circle():
    # e- in +Bz, at (axis + 300, 0) moving +y: centre on the -x side at distance R; r_c = |300 - R|
    bz, pt = 1.0, 60.0
    R = pt / (RT.KB * bz)
    x = np.array([[-3904.0 + 300.0, 0.0, 0.0]])
    p = np.array([[0.0, pt, 60.0]])
    pm, td, d0, rmax = RT.entry_truth(x, p, (-3904.0, 0.0), bz)
    np.testing.assert_allclose(pm, np.hypot(pt, 60.0))
    np.testing.assert_allclose(td, 60.0 / pt)
    np.testing.assert_allclose(d0, abs(abs(300.0 - R) - R))
    np.testing.assert_allclose(rmax, abs(300.0 - R) + R)


def _reference(n, rng):
    """Synthetic reference events: fit probability rising with nrow, residuals Gaussian with a low tail."""
    nrow = rng.integers(0, 90, n)
    td = rng.uniform(0.5, 1.2, n)
    pt = 104.0 / np.sqrt(1 + td ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = np.c_[-3904.0 + rng.uniform(380, 600, n) * np.cos(phi), rng.uniform(380, 600, n) * np.sin(phi), np.zeros(n)]
    p = np.c_[pt * np.cos(phi + 1.3), pt * np.sin(phi + 1.3), pt * td]
    ptrue, tdt, d0, rmax = RT.entry_truth(x, p, (-3904.0, 0.0), 1.0)
    fit = rng.uniform(size=n) < np.clip((nrow - 15) / 15, 0, 0.95)
    dp = np.where(rng.uniform(size=n) < 0.05, -rng.exponential(1.5, n), rng.normal(-0.05, 0.2, n))
    d = pd.DataFrame(dict(has_entry=True, nrow=nrow, p_true=ptrue, tandip_t=tdt, d0_t=d0, rmax_t=rmax, fit=fit,
                          e_x=x[:, 0], e_y=x[:, 1], e_z=x[:, 2], e_px=p[:, 0], e_py=p[:, 1], e_pz=p[:, 2]))
    d["p_reco"] = np.where(fit, ptrue + dp, np.nan)
    d["pz"] = np.where(fit, p[:, 2], np.nan)
    d["tandip"] = np.where(fit, tdt + rng.normal(0, 0.01, n), np.nan)
    d["d0"] = np.where(fit, np.abs(d0 + rng.normal(0, 10, n)), np.nan)
    d["rmax"] = np.where(fit, rmax + rng.normal(-3, 8, n), np.nan)
    d["pdg"] = np.where(fit, 11.0, np.nan)
    d["fitcon"] = np.where(fit, rng.uniform(size=n), np.nan)
    d["chi2ndf"] = np.where(fit, rng.gamma(20, 0.05, n), np.nan)
    d["nactive"] = np.where(fit, nrow * 0.9, np.nan)
    return d


def test_tables_reproduce_the_reference_efficiencies_and_residuals():
    rng = np.random.default_rng(1)
    d = _reference(40000, rng)
    tab = RT.build(d)
    v = _reference(20000, np.random.default_rng(2))
    o = RT.apply_numpy(tab, np.random.default_rng(3), v.has_entry.to_numpy(), v.nrow.to_numpy(),
                       v[["e_x", "e_y", "e_z"]].to_numpy(), v[["e_px", "e_py", "e_pz"]].to_numpy())
    assert abs(o["fit"].mean() - v.fit.mean()) < 0.01
    ref_ce = RT.CE_CUTS
    qual = v.fit & (v.fitcon > ref_ce["fitcon"])
    ce = qual & v.tandip.between(*ref_ce["tandip"]) & v.d0.between(*ref_ce["d0"]) & v.rmax.between(*ref_ce["rmax"])
    assert abs(o["ce"].mean() - ce.mean()) < 0.012
    res = (o["p_reco"] - o["p_true"])[o["qual"]]
    rr = (v.p_reco - v.p_true)[qual].to_numpy()
    assert abs(np.median(res) - np.median(rr)) < 0.02
    assert abs(np.mean(res < -1.0) - np.mean(rr < -1.0)) < 0.01
    assert not np.isfinite(o["p_reco"][~o["qual"]]).any()


def test_device_apply_matches_the_numpy_response():
    jax = pytest.importorskip("jax")
    from trkfastsim.recoresp import device as RD
    d = _reference(20000, np.random.default_rng(4))
    tab = RT.build(d)
    v = _reference(20000, np.random.default_rng(5))
    x, p = v[["e_x", "e_y", "e_z"]].to_numpy(), v[["e_px", "e_py", "e_pz"]].to_numpy()
    on = RT.apply_numpy(tab, np.random.default_rng(6), v.has_entry.to_numpy(), v.nrow.to_numpy(), x, p)
    od = RD.apply(RD.to_device(tab), jax.random.PRNGKey(7), jax.numpy.asarray(v.has_entry.to_numpy()),
                  jax.numpy.asarray(v.nrow.to_numpy()), jax.numpy.asarray(x), jax.numpy.asarray(p))
    od = {k: np.asarray(a) for k, a in od.items()}
    np.testing.assert_allclose(od["p_true"], on["p_true"], rtol=1e-12)
    for k in ("fit", "qual", "ce"):
        assert abs(od[k].mean() - on[k].mean()) < 0.012, k
    a, b = od["p_reco"][od["qual"]] - od["p_true"][od["qual"]], on["p_reco"][on["qual"]] - on["p_true"][on["qual"]]
    assert abs(np.median(a) - np.median(b)) < 0.02
    assert abs(np.quantile(a, 0.05) - np.quantile(b, 0.05)) < 0.1
