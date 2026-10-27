"""Actual hit/seeding/F2 integration preserves full identities under event permutations."""
import numpy as np
import pytest

pytest.importorskip("jax")

from trkfastsim.analyses import digicmp
from test_reco_jax import _cond, _digis


def test_real_reco_permutation_variable_adc_and_empty_events():
    rng = np.random.default_rng(1701)
    c = _cond(20736, rng)
    d = _digis(600, 20736, rng)
    ne = len(d["evkey"])
    row = np.arange(len(d["ev"]))
    us = row
    d["sid"] = ((us // 576) << 10) | (((us // 96) % 6) << 7) | (us % 96)
    rank = row - np.searchsorted(d["ev"], d["ev"])
    along = rank % 2 == 0
    st = c["straw"]
    st["midx"][us], st["midy"][us], st["midz"][us] = 400., 400. - rank * 40., 0.
    st["dirx"][us], st["diry"][us], st["dirz"][us] = along, 0., ~along
    for k in ("dead", "noisy", "toff_hv", "toff_cal"):
        st[k][:] = 0.
    st["hlen"][:], st["halfvp"][:] = 500., 100.
    c["totdtime"][:], c["totderror"][:] = 20., 5.
    d["pbt"][:] = 0.
    d["tdc0"] = np.rint((1000. + d["ev"] * 1000. + rank * 40. / 299.9 + 22.5) / c["scalar"]["tdcLSB"]).astype(int)
    d["tdc1"] = d["tdc0"].copy()
    d["evkey"] = np.column_stack([np.full(ne, 1400), np.full(ne, 3), np.arange(ne) + 2**20])
    d["evkey"] = np.vstack([d["evkey"], [1400, 3, 9]])
    d["pbt"] = np.r_[d["pbt"], 0.]
    drift = {"dt": np.array([-100., 0., 100.]), "rdrift": np.array([0., 0., 2.5]),
             "sderr": np.full(3, .15), "uderr": np.full(3, .2)}
    perm = rng.permutation(len(d["evkey"]))
    mixed = digicmp._restrict(d, perm)
    restored = digicmp._restrict(mixed, np.argsort(perm))
    for k in d:
        np.testing.assert_array_equal(restored[k], d[k])
    a = digicmp.f2_yield(d, c, drift, chunk=11, fixed_shapes=True)
    b = digicmp.f2_yield(mixed, c, drift, chunk=7, fixed_shapes=True)
    assert a["seeds"] > 0 and a["good_fits"] > 0
    np.testing.assert_array_equal(a["fits_per_event"], b["fits_per_event"][np.argsort(perm)])
    for k in ("seeds", "good_fits", "events_with_fit"):
        assert a[k] == b[k]
    for k in ("nactive_q", "chi2ndf_q"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-8, atol=1e-8)
