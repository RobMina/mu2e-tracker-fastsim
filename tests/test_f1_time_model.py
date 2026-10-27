"""F1 time_model='stock': KinKal time semantics (time order, immutable constraint time, owned old trajectory)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from test_kalman_jax import _toy_track, _anns_available  # noqa: E402
from trkfastsim.reco import kalman_jax as K  # noqa: E402


def test_nearest_index_matches_piecewise_trajectory():
    ranges = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [9.0, 9.0]])
    t = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.9, 4.0, 7.0])
    idx = np.asarray(K.nearest_index(ranges, jnp.array(2), t))
    # clamp at the ends, earlier piece at an exact boundary (lower_bound on range end < t); padding never selected
    np.testing.assert_array_equal(idx, [0, 0, 0, 0, 1, 1, 2, 2, 2])


def _stock_iteration(t_cons, elo=1.0, sxscale=1.0, legacy=(), prior=None):
    xt, h, hj, cj, d, dj = _toy_track(3)
    N = len(h["lr"])
    x0 = jnp.asarray(np.r_[xt + np.array([1.0, -1.0, 0.003, 0.002, 0.5]), K.MOM])
    C0 = K.seed_covariance(x0)
    xe, Ce = jnp.tile(x0, (2 * N, 1)), jnp.tile(C0, (2 * N, 1, 1))
    hs = {"active": hj["active"], "lr": hj["lr"], "dvar": jnp.full(N, 0.15 ** 2), "s_cons": jnp.array(0.0),
          "mom_seed": jnp.array(K.MOM)}
    owner = K.seed_owner(x0, C0, jnp.array(-1e3), jnp.array(1e3), N)
    keep = K.ELOSS
    K.ELOSS = elo
    try:
        o = K.iterate_once(xe, Ce, hj, cj, dj, hs, jnp.ones(N, bool), 1.0, 1.0, sxscale, prior, 1e6,
                           tm={"t_cons": jnp.asarray(t_cons), "owner": owner, "legacy": legacy})
    finally:
        K.ELOSS = keep
    return o, N


def test_constraint_outside_hits_extends_bounds_and_owns_the_front_piece():
    """A ParameterHit earlier than every hit is the first active measurement (Track::setBounds): the fit range
    starts at its time and the front piece carries the 1e-4 MeV/c momentum constraint."""
    o, N = _stock_iteration(-1e2)
    assert float(o["s_front"]) == pytest.approx(-1e2)
    assert int(o["owner"]["count"]) == int(np.asarray(o["inbounds"]).sum())
    assert np.sqrt(float(o["Cfront"][5, 5])) < 1.5e-4
    assert np.sqrt(float(o["Cback"][5, 5])) > 3 * K.MOM_CONSTRAINT
    r = np.asarray(o["owner"]["ranges"])[: int(o["owner"]["count"]) + 1]
    assert np.all(np.diff(r[:, 0]) > 0) and np.all(r[:-1, 1] == r[1:, 0])
    assert o["query"][0] == pytest.approx(0.5 * (r[0, 0] + r[0, 1]))


def test_constraint_time_moves_its_piece_without_material_change_of_geometry():
    """Without material the pieces are one line: the constraint's time position cannot change the hit fit."""
    a, N = _stock_iteration(-1e2, elo=0.0, sxscale=0.0)
    b, _ = _stock_iteration(1e2, elo=0.0, sxscale=0.0)
    sig = np.sqrt(np.diagonal(np.asarray(a["C"][:N]), axis1=1, axis2=2))
    assert np.max(np.abs(np.asarray(a["x"][:N]) - np.asarray(b["x"][:N]))[:, :5] / sig[:, :5]) < 1e-4


@pytest.mark.skipif(not _anns_available(), reason="KinKal ANN weights not found (set MU2E_SEARCH_PATH)")
def test_full_schedule_stock_time_model_converges_on_toy_tracks():
    xt, h, hj, cj, d, dj = _toy_track(3)
    x0 = jnp.asarray(xt + np.array([3.0, -2.0, 0.01, 0.005, 2.0]))
    cf, hist, good = jax.jit(lambda x: K.fit_track(x, hj, cj, dj, K.stock_stages(), K.load_anns(),
                                                   time_model="stock"))(x0)
    assert bool(good)
    xt0, Ct0, _ = K.t0_piece(cf)
    sig = np.sqrt(np.diag(np.asarray(Ct0)))
    assert np.all(np.abs(np.asarray(xt0)[:5] - xt) / sig[:5] < 4)


def test_leave_one_out_parts():
    """bounds: hit-only bounds ignore an early constraint; statusref: query at the bounds; transit: last range ends at
    the bound; cons: the constraint time follows the local-s lift, not t_cons; prior: the passed prior is used."""
    a, N = _stock_iteration(-1e2)
    b, _ = _stock_iteration(-1e2, legacy=("bounds", "statusref", "transit"))
    assert float(b["s_front"]) > float(a["s_front"]) + 1.0
    np.testing.assert_allclose(np.asarray(b["query"]), [float(b["s_front"]), float(b["s_back"])])
    rb = np.asarray(b["owner"]["ranges"])
    assert rb[int(b["owner"]["count"]), 1] == pytest.approx(float(b["s_back"]))
    c, _ = _stock_iteration(-1e2, legacy=("cons",))
    assert float(c["s_front"]) > float(a["s_front"]) + 1.0
    x0 = np.asarray(a["xfront"])
    far = {"xf": jnp.asarray(x0 + np.r_[50.0, 0, 0, 0, 0, 0]), "Cf": jnp.eye(6) * 1e-6,
           "xb": jnp.asarray(x0), "Cb": jnp.eye(6)}
    d, _ = _stock_iteration(-1e2, legacy=("prior",), prior=far)
    assert not np.allclose(np.asarray(d["x"]), np.asarray(a["x"]))


def test_option_validation():
    xt, h, hj, cj, d, dj = _toy_track(3)
    x0 = jnp.asarray(xt)
    with pytest.raises(ValueError):
        K.fit_track(x0, hj, cj, dj, K.stock_stages(), None, xing_activity="hit")
    with pytest.raises(ValueError):
        K.fit_track(x0, hj, cj, dj, K.stock_stages(), None, time_legacy=("cons",))
    with pytest.raises(ValueError):
        K.fit_track(x0, hj, cj, dj, K.stock_stages(), None, time_model="stock", time_legacy=("order",))


def test_gas_scatter_fraction_p109_is_larger():
    """p109 DetMaterial uses the gas fraction for gas; the static (first-material) solid fraction gives less variance."""
    from trkfastsim.reco import kkmaterial as KM
    a = float(KM.scatter_angle_var("straw-gas", 2000.0, 4.0, scatterfrac=KM.SCATTERFRAC["solid"]))
    b = float(KM.scatter_angle_var("straw-gas", 2000.0, 4.0, scatterfrac=KM.SCATTERFRAC["gas"]))
    assert 1.05 < b / a < 1.3
    assert KM.GAS_SCATTERFRAC == KM.STATIC_SCATTERFRAC
